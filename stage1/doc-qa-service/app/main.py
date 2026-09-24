"""应用装配 —— 把前面所有零件组装成一个能跑的服务。

这个文件不含任何业务逻辑。
它只做四件事：启动时准备好资源、装中间件、装异常处理器、挂路由。

启动方式（在项目根目录下）：
    uv run fastapi dev app/main.py      开发模式，改代码自动重启
    uv run fastapi run app/main.py      生产模式，不自动重启

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    lifespan(app)
        一个异步上下文管理器。yield 之前的代码在服务启动时跑一次，
        yield 之后的代码在服务关闭时跑一次。由 FastAPI 自动调用。
    create_app()
        吃：无 -> 吐一个配置好的 FastAPI 实例。被下面的 app = create_app() 调用。
    app
        模块级变量，就是服务本体。uvicorn 通过 "app.main:app" 这个路径找到它。

阅读地图
    必读：lifespan 为什么存在、中间件和异常处理器的装配顺序
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.middleware import RequestIdMiddleware
from app.api.routes import ask as ask_routes
from app.api.routes import health as health_routes
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.rag.embeddings import build_embeddings
from app.rag.pipeline import RagPipeline, build_model
from app.rag.prompt import build_prompt
from app.rag.store import build_store, count_documents

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务的开机和关机流程。

    **lifespan（生命周期）** 是 FastAPI 现行的启动/关闭钩子写法。
    它是一个异步上下文管理器：yield 前面是开机，后面是关机。

    为什么这些东西必须放在这里，而不是放在路由函数里？
    加载嵌入模型要十几秒、占几百兆显存。
    如果每个请求都加载一次，第一个用户要等十几秒，第十个用户会把显存撑爆。
    放在 lifespan 里，整个进程只加载一次，所有请求共用。

    这是 FastAPI 服务性能上最重要的一条原则：**贵的资源只初始化一次**。
    """
    # ---- 开机 --------------------------------------------------------------
    # 步骤 1：读配置，配好日志。日志必须最先配，否则后面的日志都打不出来。
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("服务启动中", extra={"app": settings.app_name})

    # 步骤 2：加载嵌入模型。这是启动里最慢的一步，十几秒。
    embeddings = build_embeddings(settings)

    # 步骤 3：连接向量库。注意这里是只读用法，服务进程从不写库。
    store = build_store(settings, embeddings)

    # 步骤 4：加载聊天模型，准备好提示词模板。
    model = build_model(settings)
    prompt = build_prompt()

    # 步骤 5：把四个零件组装成 pipeline，挂到 app.state 上。
    #         app.state 是应用级的共享空间，路由函数通过 deps.get_pipeline 取用。
    app.state.store = store
    app.state.pipeline = RagPipeline(store, model, prompt, settings)

    # 步骤 6：报一下知识库里有多少条，方便启动时一眼看出有没有忘记入库。
    total = count_documents(store)
    logger.info("服务就绪", extra={"documents": total})
    if total == 0:
        logger.warning("知识库为空，请先执行 uv run python -m app.ingest")

    # ---- 服务运行中 ---------------------------------------------------------
    # yield 这一句之后，服务开始接收请求，一直停在这里直到收到关闭信号。
    yield

    # ---- 关机 --------------------------------------------------------------
    # 步骤 7：释放资源。
    #         Chroma 的本地客户端没有需要显式关闭的连接，所以这里只记一条日志。
    #         将来接了 Redis、数据库连接池，关闭动作就写在这里。
    logger.info("服务关闭")


def create_app() -> FastAPI:
    """造出配置好的 FastAPI 实例。

    为什么包成一个函数，而不是在模块里直接写 app = FastAPI(...)？
    因为测试时需要造一个干净的、可以塞假依赖的应用实例。
    有了这个函数，测试里调一次就能得到一个全新的 app。
    这种写法叫「应用工厂（application factory）」。
    """
    settings = get_settings()

    # 步骤 1：创建应用，并把文档页的标题写清楚。
    app = FastAPI(
        title="文档问答服务",
        description="基于电商客服知识库的检索增强问答接口",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 步骤 2：装中间件。
    #         中间件是「洋葱」结构：后装的在外层，请求先穿过外层。
    #         追踪编号要在最外层生成，这样连异常处理器打的日志也能带上它。
    app.add_middleware(RequestIdMiddleware)

    # 步骤 3：装异常处理器，让所有错误响应长成同一个形状。
    register_exception_handlers(app)

    # 步骤 4：挂路由。
    app.include_router(health_routes.router)
    app.include_router(ask_routes.router)

    logger.info("应用装配完成", extra={"app": settings.app_name})
    return app


# [核心] 模块级的 app 变量是服务的入口。
#        uvicorn 通过 "app.main:app" 这个字符串找到它：
#        前半段是模块路径，冒号后面是变量名。
app = create_app()
