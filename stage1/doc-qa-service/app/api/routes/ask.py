"""问答接口 —— POST /v1/ask。

注意这个文件有多短。
HTTP 层的职责只有三件事：收请求、调业务、把结果转成响应格式。
所有真正的逻辑都在 app/rag/pipeline.py 里。

这样分的好处，在写测试时最明显：
业务逻辑可以脱离 HTTP 单独测，HTTP 层可以塞一个假 pipeline 单独测。
如果逻辑写在路由函数里，测任何一样都得把整个服务起起来。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    router              APIRouter 对象，装着 /v1/ask 这个接口。
    ask(payload, pipeline)
        吃「已校验的请求体 + 注入进来的 pipeline」-> 吐 AskResponse。
        被 FastAPI 在收到 POST /v1/ask 时调用。

阅读地图
    必读：路由函数的四个步骤、为什么 URL 里要带版本号 v1
"""

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_pipeline
from app.core.logging import request_id_var
from app.rag.pipeline import RagPipeline
from app.schemas.ask import AskRequest, AskResponse, CitationOut

# [核心] prefix="/v1" 让这个 router 下所有接口的路径前面都带上 /v1。
#        为什么一开始就加版本号？
#        接口一旦有人用了，响应格式就不能随便改，改了会把调用方弄挂。
#        将来要改格式时，新开一个 /v2 并行跑一段时间，让调用方慢慢迁移。
#        没有版本号的话，你只能要么不改、要么强迫所有人同时升级。
router = APIRouter(prefix="/v1", tags=["问答"])


@router.post("/ask", response_model=AskResponse, summary="基于知识库回答问题")
async def ask(
    payload: AskRequest,
    pipeline: Annotated[RagPipeline, Depends(get_pipeline)],
) -> AskResponse:
    """收一个问题，返回带出处的答案。

    两个参数是怎么来的：
        payload   框架把请求体的 JSON 按 AskRequest 校验并转换后递进来。
                  校验不通过的请求根本进不到这个函数，直接被框架挡回 422。
        pipeline  框架调用 get_pipeline 拿到启动时造好的那一个实例。
    """
    # 步骤 1：调业务层。这一行是这个函数里唯一真正干活的地方。
    result = await pipeline.ask(payload.question)

    # 步骤 2：把业务层返回的 Citation 数据类转成响应用的 CitationOut。
    #         asdict 是 dataclasses 提供的函数，把数据类实例变成字典。
    #         两个星号把字典展开成关键字参数，等价于逐个字段手写一遍。
    #
    #         为什么要转一道？业务层的 Citation 和对外的 CitationOut 是两个东西。
    #         业务层将来可能加上「原文在第几行」这类内部字段，
    #         那是我们自己排查用的，不该出现在对外响应里。
    #         中间隔一层，内部结构变化就不会意外泄漏到接口契约上。
    citations = [CitationOut(**asdict(c)) for c in result.citations]

    # 步骤 3：把追踪编号带进响应体。
    #         此刻中间件已经把它放进上下文变量了，这里直接取。
    request_id = request_id_var.get()

    # 步骤 4：组装响应。返回 Python 对象即可，框架按 AskResponse 的声明转成 JSON。
    return AskResponse(
        answer=result.answer,
        citations=citations,
        retrieved=result.retrieved,
        elapsed_ms=result.elapsed_ms,
        request_id=request_id,
    )
