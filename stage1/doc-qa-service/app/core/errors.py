"""异常层 —— 统一的错误码、异常类型与 FastAPI 异常处理器。

为什么要单独一层：不做统一处理时，服务出错会有三种糟糕的表现——
  1. 直接把 Python 的 traceback（错误堆栈）返回给调用方，泄漏内部文件路径和代码结构，是安全问题；
  2. 每个接口自己拼错误 JSON，格式各不相同，前端要写一堆分支去猜；
  3. 只返回 500，运维拿不到任何线索，只能翻日志大海捞针。

本层的约定：**所有**错误响应都是同一个形状
    {"error": {"code": "LLM_TIMEOUT", "message": "生成超时", "request_id": "..."}}
  code       给机器读，前端按它决定怎么展示、要不要重试
  message    给人读，可以直接弹给用户
  request_id 给运维读，用户截图报错时把它报过来就能定位到那次请求的全部日志

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    ErrorCode               枚举，一堆错误码常量。ErrorCode.LLM_TIMEOUT 可直接当字符串用。
    AppError(message, ...)  异常类。造一个异常对象，带 message / code / http_status 三个信息。
                            业务代码里写 raise LLMTimeoutError("生成超时")。
    error_body(code, msg)   吃错误码和文案 -> 吐 dict，形如
                            {"error": {"code": ..., "message": ..., "request_id": ...}}
                            request_id 自动从上下文变量取，不用传。四个处理器都会调它。
    app_error_handler(request, exc)
                            吃「请求对象 + 异常对象」-> 吐 JSONResponse。
                            由 FastAPI 捕获到 AppError 时自动调用，不会手动调。
    validation_error_handler / http_exception_handler / unhandled_error_handler
                            同上，分别对应「参数不合法」「404 这类框架异常」「其它一切异常」。
    register_exception_handlers(app)
                            吃 FastAPI 应用对象 -> 无返回值，把上面四个处理器登记进去。
                            阶段 5 在 main.py 里调一次。

阅读地图
    必读：AppError、app_error_handler、unhandled_error_handler
    扫读：ErrorCode 取值、各子类的状态码选择
"""

from enum import StrEnum

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)


class ErrorCode(StrEnum):
    """[了解] 错误码枚举。

    **StrEnum** 是 Python 3.11 引入的枚举类型，成员本身就是字符串，
    所以 ErrorCode.LLM_TIMEOUT 可以直接当 "LLM_TIMEOUT" 用，json 序列化也不用转换。

    为什么要用枚举而不是到处写字符串字面量：拼错 "LLM_TIMEOUT" 为 "LLM_TIMEOUTS"
    不会有任何报错，客户端的判断就静默失效了；用枚举则拼错时立刻 AttributeError。
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"   # 请求参数不合法
    KB_NOT_READY = "KB_NOT_READY"           # 知识库没准备好（没入库/连不上）
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"   # 检索环节出错
    LLM_TIMEOUT = "LLM_TIMEOUT"             # 生成超时
    LLM_FAILED = "LLM_FAILED"               # 生成失败（上游报错）
    INTERNAL_ERROR = "INTERNAL_ERROR"       # 兜底：没预料到的错


class AppError(Exception):
    """本服务所有**业务异常**的基类。

    设计要点：每个子类只要声明自己的 code 和 http_status 两个类属性，
    就自动获得统一的错误响应，不用各自写处理逻辑。
    """

    # [核心] 类属性作为默认值，子类覆盖它们即可。
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    # ---- 练习 3：实现构造函数 ---------------------------------------------
    # [核心] 要求：
    #   · 接收一个必填的 message（给人读的错误描述）
    #   · 允许调用方用关键字参数临时覆盖 code 和 http_status（默认 None 表示不覆盖，用类属性）
    #   · 把 message 存到 self.message
    #   · 调用父类 Exception 的构造函数，把 message 传进去
    #     （不调的话，print(异常) 或日志里打印异常时会是空的，排查时很难受）
    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        http_status: int | None = None,
    ) -> None:
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        super().__init__(message)
    # -----------------------------------------------------------------------


# [了解] 具体的业务异常。状态码的选择是有讲究的，不是随便挑的：
class KnowledgeBaseNotReadyError(AppError):
    """知识库未就绪（还没跑 ingest，或库连不上）。"""

    code = ErrorCode.KB_NOT_READY
    # 503 Service Unavailable = 「我暂时不能服务，稍后再来」。
    # 用 503 而不是 500，是因为这通常是可恢复的（跑一次 ingest 就好了），
    # 而且 503 会让上游的负载均衡器把流量切走，500 不会。
    http_status = status.HTTP_503_SERVICE_UNAVAILABLE


class RetrievalError(AppError):
    """检索环节失败。"""

    code = ErrorCode.RETRIEVAL_FAILED
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


class LLMTimeoutError(AppError):
    """调用大模型超时。"""

    code = ErrorCode.LLM_TIMEOUT
    # 504 Gateway Timeout = 「我作为网关，等我依赖的上游超时了」。
    # 我们调 DeepSeek 的角色正是网关，所以 504 比 500 更准确地描述了故障位置。
    http_status = status.HTTP_504_GATEWAY_TIMEOUT


class LLMError(AppError):
    """大模型返回错误（限流、余额不足、上游 5xx 等）。"""

    code = ErrorCode.LLM_FAILED
    # 502 Bad Gateway = 「上游给了我一个无效响应」。
    http_status = status.HTTP_502_BAD_GATEWAY


def error_body(code: ErrorCode | str, message: str) -> dict:
    """[核心] 拼出统一的错误响应体。request_id 自动从上下文变量里取，调用方不用传。"""
    return {
        "error": {
            "code": str(code),
            "message": message,
            "request_id": request_id_var.get(),
        }
    }


# ===========================================================================
# 异常处理器：把异常翻译成 HTTP 响应
# ===========================================================================
# **异常处理器（exception handler）**：注册给框架的函数。请求处理过程中抛出了某类异常时，
# 框架不再让它冒泡成 500，而是交给这个函数，由它决定返回什么响应。

# ---- 练习 4：实现业务异常的处理器 ------------------------------------------
# [核心] 要求：
#   · 用 logger.warning 记一条日志，带上 code（用 extra= 传成独立字段，方便按字段过滤）
#   · 返回 JSONResponse，状态码取 exc.http_status，响应体用 error_body(...)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content=error_body(exc.code, exc.message),
    )
# ---------------------------------------------------------------------------


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """[了解] 请求体不合法（比如 question 是空字符串）时，FastAPI 抛的就是这个异常。

    默认响应体是 FastAPI 自己的 {"detail": [...]} 格式，和我们的约定不一致，
    所以这里覆盖掉，让**所有**错误——不管来自业务还是框架——长成同一个样子。
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=error_body(ErrorCode.VALIDATION_ERROR, "请求参数不合法"),
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """[了解] 404、405 这类框架级 HTTP 异常，同样收敛成统一格式。"""
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(f"HTTP_{exc.status_code}", str(exc.detail)),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """[核心] 兜底处理器：任何没被上面几个接住的异常，最后都会落到这里。

    两件事必须同时做到，缺一不可：
      · 对**日志**：用 exc_info=True 把完整堆栈打出来，否则线上出了怪问题你无从查起；
      · 对**调用方**：只回一句笼统的话，绝不把堆栈内容放进响应体。
        堆栈里含有文件路径、函数名、有时还有变量值，返回出去等于把内部结构送给攻击者。
    """
    logger.error("未处理的异常", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_body(ErrorCode.INTERNAL_ERROR, "服务内部错误，请稍后重试"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """[脚手架] 在应用装配阶段一次性注册全部处理器。

    注册顺序不影响匹配：框架按异常类型的继承关系找最具体的那个处理器。
    """
    app.add_exception_handler(AppError, app_error_handler)          # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)    # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_error_handler)
