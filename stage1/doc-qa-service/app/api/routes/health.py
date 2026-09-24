"""健康检查接口 —— 告诉运维系统「我还活着」和「我能干活了」。

这两件事必须分成两个接口，是容器化部署的标准做法。

**存活探针（liveness probe）/healthz**
    问的是「进程还在不在」。
    答不上来（超时或非 200），编排系统会**杀掉容器重启**。
    所以它必须极轻，不能依赖任何外部资源。

**就绪探针（readiness probe）/readyz**
    问的是「现在能不能接流量」。
    答不上来，编排系统**把流量切走但不重启**，等它自己好。
    所以它要真的去检查依赖：模型加载完没有、知识库里有没有数据。

两者混成一个会怎样？我们的服务启动要花十几秒加载模型。
如果存活探针也去查知识库，那么启动期间它会一直失败，
编排系统以为进程挂了，杀掉重启，重启又要加载十几秒，再被杀掉。
结果是无限重启循环，服务永远起不来。这是真实发生过的事故。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    router          一个 APIRouter 对象，装着这个文件里的两个接口。
                    在 app/main.py 里用 app.include_router(router) 挂上去。
    healthz()       吃：无 -> 吐 {"status": "ok"}。被编排系统定时调用。
    readyz(request) 吃请求对象 -> 吐 {"status": "ready", "documents": 条数}。
                    知识库为空时抛异常，转成 503。

阅读地图
    必读：两个探针的区别、readyz 里为什么要用 to_thread
"""

import anyio.to_thread
from fastapi import APIRouter, Request

from app.core.errors import KnowledgeBaseNotReadyError
from app.rag.store import count_documents

# [脚手架] APIRouter 是「一组接口」的容器。
#          按功能分成多个 router，再在 main.py 里统一挂载，
#          比把所有接口写在一个文件里好维护。
#          tags 决定它们在 /docs 文档页里被分到哪一组。
router = APIRouter(tags=["健康检查"])


@router.get("/healthz", summary="存活探针")
async def healthz() -> dict[str, str]:
    """只要这个函数能返回，就说明进程没死、事件循环没卡住。"""
    # 步骤 1：直接返回一个常量。
    #         这里绝对不能查数据库、不能调模型、不能读磁盘。
    #         任何外部依赖出问题，都会导致容器被误杀重启。
    return {"status": "ok"}


@router.get("/readyz", summary="就绪探针")
async def readyz(request: Request) -> dict[str, object]:
    """检查模型加载完了没有、知识库里有没有数据。"""
    # 步骤 1：取出启动时挂上去的向量库对象。
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise KnowledgeBaseNotReadyError("服务尚未初始化完成")

    # 步骤 2：数一下库里有多少条。
    #         count_documents 是同步函数，内部要读 SQLite 文件。
    #         直接在 async 函数里调它，磁盘 IO 会把整个事件循环卡住，
    #         这期间所有其它请求都在排队。
    #
    #         anyio.to_thread.run_sync 把这个同步函数丢到线程池里跑，
    #         事件循环在等待期间可以去处理别的请求。
    #         anyio 是 FastAPI 底层用的异步库，装 FastAPI 时会一并装上。
    total = await anyio.to_thread.run_sync(count_documents, store)

    # 步骤 3：库是空的就报 503。
    #         这时候进程是好的，只是还没入库，所以要「切走流量」而不是「重启进程」。
    if total == 0:
        raise KnowledgeBaseNotReadyError("知识库为空，请先执行入库命令")

    return {"status": "ready", "documents": total}
