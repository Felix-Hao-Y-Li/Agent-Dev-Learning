"""问答接口的对外契约。

**契约（contract）**：服务承诺「你按这个格式发给我，我按那个格式还给你」。
在 FastAPI 里，契约就是这些 Pydantic 模型类。
写下来之后有三个好处，全部自动获得：

1. 请求进来时自动校验。question 是空字符串、或者超长，框架直接挡回去，
   业务代码不会被脏数据碰到。
2. 自动生成接口文档。启动服务后打开 /docs 就能看到，还能直接点按钮试调用。
3. 响应自动序列化。返回 Python 对象，框架按这里声明的字段转成 JSON。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    AskRequest      请求体的形状。框架收到 JSON 后自动转成这个对象。
    CitationOut     一条出处在响应里的形状。
    AskResponse     响应体的形状。路由函数返回它，框架自动转成 JSON。

阅读地图
    必读：AskRequest 的三道校验、AskResponse 为什么要带 request_id
"""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.core.config import get_settings

# [了解] 长度上限从配置里取，不写死在代码里。
#        这一行在模块被导入时执行一次，拿到的是那个全局唯一的配置对象。
_MAX_QUESTION = get_settings().max_question_chars


class AskRequest(BaseModel):
    """POST /v1/ask 的请求体。"""

    # [核心] Annotated 的意思是「这个字段是 str，另外还附加一些约束」。
    #        StringConstraints 里的三条约束各自挡住一类脏数据：
    #
    #        strip_whitespace=True  先去掉首尾空白。
    #            不去的话，用户发来一个全是空格的字符串，min_length 检查会通过，
    #            然后拿着空问题去检索，白白烧掉一次模型调用。
    #        min_length=1           去完空白还得有内容。
    #        max_length             挡住有人贴一整本书进来。
    #            没有上限会怎样？问题被原样拼进提示词，
    #            轻则超出模型的上下文长度直接报错，重则一次调用烧掉大量 token。
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=_MAX_QUESTION),
    ] = Field(
        ...,
        description="用户的问题",
        # [脚手架] examples 里的内容会出现在 /docs 的示例框里，方便手动试调用。
        examples=["退货的运费谁出？"],
    )


class CitationOut(BaseModel):
    """一条出处。"""

    no: int = Field(description="答案里标注的编号，对应 [1] [2]")
    source: str = Field(description="来源文件名")
    section: str = Field(description="章节路径")
    score: float = Field(description="检索距离，越小越相关")
    snippet: str = Field(description="原文片段，供人工核对")


class AskResponse(BaseModel):
    """POST /v1/ask 的响应体。"""

    answer: str = Field(description="模型生成的答案，句末带 [编号] 出处")
    citations: list[CitationOut] = Field(description="答案实际引用到的出处")
    retrieved: int = Field(description="本次检索回来的条数")
    elapsed_ms: int = Field(description="检索加生成的总耗时，毫秒")

    # [核心] 为什么响应里要带 request_id？
    #        用户报错时截图给你，上面有这个 id，你就能在日志里精确定位到那一次请求，
    #        看到它检索了什么、生成花了多久、错在哪一步。
    #        没有它的话，你只能凭「大概几点钟」在日志海里捞。
    request_id: str = Field(description="本次请求的追踪编号")
