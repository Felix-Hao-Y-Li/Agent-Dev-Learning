"""依赖注入 —— 路由函数怎么拿到它需要的东西。

**依赖注入（dependency injection）**：路由函数不自己去造对象，而是「声明我需要什么」，
由框架在调用它之前把东西准备好递进来。

举个对比就清楚了。不用依赖注入的写法是在路由函数里写：
    pipeline = RagPipeline(build_store(...), build_model(...), ...)
这样每来一个请求都会重新加载一遍模型，几秒钟就过去了，服务根本没法用。

用依赖注入的写法是：
    async def ask(pipeline: RagPipeline = Depends(get_pipeline)):
框架调用 get_pipeline 拿到启动时就造好的那一个实例，直接递进来。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    get_pipeline(request)
        吃请求对象 -> 吐启动时造好的那个 RagPipeline 实例。
        被 FastAPI 在调用 /v1/ask 之前自动调用。

阅读地图
    必读：为什么从 app.state 取，而不是用全局变量
"""

from fastapi import Request

from app.core.errors import KnowledgeBaseNotReadyError
from app.rag.pipeline import RagPipeline


def get_pipeline(request: Request) -> RagPipeline:
    """从应用状态里取出那个全局唯一的 RagPipeline。"""
    # 步骤 1：app.state 是 FastAPI 提供的一个「应用级共享空间」。
    #         启动时（见 app/main.py 的 lifespan）我们把造好的对象挂在上面，
    #         这里再取出来。
    #
    #         为什么不用模块级的全局变量？两个原因。
    #         一是全局变量在导入模块时就存在，而模型要到启动时才加载完，
    #           容易出现「导入了但还是 None」的时序问题。
    #         二是测试时没法替换。用 app.state 的话，
    #           测试可以造一个自己的 app、挂一个假 pipeline 上去，完全不碰真模型。
    #
    #         getattr(对象, 名字, 默认值) 在属性不存在时返回默认值，不会抛异常。
    pipeline = getattr(request.app.state, "pipeline", None)

    # 步骤 2：还没初始化完就明确报错，而不是让后面的代码对着 None 崩掉。
    #         这个异常会被统一异常处理器转成 503，告诉调用方「稍后再来」。
    if pipeline is None:
        raise KnowledgeBaseNotReadyError("服务尚未初始化完成，请稍后重试")

    return pipeline
