"""日志层 —— 结构化日志 + request_id 全链路串联。

**结构化日志（structured logging）**：每行日志是一个 JSON 对象而不是一句人话。
为什么要这样？线上日志是给机器看的：日志系统（ELK、Loki 之类）要能按字段搜
「所有 level=ERROR 且 request_id=abc 的记录」。一行 `用户提问失败了` 这样的人话，
机器只能做全文匹配，搜不出结构。

**request_id**：给每个进来的 HTTP 请求发一个唯一编号，这次请求产生的每一条日志都带上它。
没有它的话，10 个用户并发提问，日志会交错成一团，你根本分不清哪条检索日志对应哪次生成日志。
线上排查问题的第一句话永远是「把 request_id 给我」。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    request_id_var          一个变量，不是函数。存当前请求的编号，字符串。
    _RESERVED               一个常量集合，存 LogRecord 自带的属性名。给 _extra_fields 用。
    _extra_fields(record)   吃 LogRecord -> 吐 dict，只含调用方用 extra= 塞进来的字段。
                            例：logger.info("完成", extra={"hits": 4}) -> {"hits": 4}
                            被 JsonFormatter.format 调用。
    JsonFormatter.format(record)
                            吃 LogRecord -> 吐一行 JSON 字符串。由 logging 框架自动调用，
                            业务代码永远不会直接调它。
    setup_logging(level)    吃日志级别字符串 -> 无返回值，只改全局配置。应用启动时调一次。
    get_logger(name)        吃名字 -> 吐一个 Logger 对象。业务代码里写 get_logger(__name__)。

阅读地图
    必读：request_id_var / _extra_fields / JsonFormatter.format
    扫读：setup_logging、_RESERVED
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# ===========================================================================
# 一、request_id 的存放处
# ===========================================================================
# **ContextVar（上下文变量）**：标准库 contextvars 提供的一种变量。
# 它和全局变量的区别是：每个「执行上下文」看到的值是各自独立的。
# 在 asyncio 里，每个请求的处理协程有自己的上下文，所以 A 请求设置的 request_id
# 不会被 B 请求读到——即使两个请求在同一个进程、同一个线程里交替执行。
#
# 为什么不用全局变量？并发时会串号：A 刚设成 "aaa"，B 立刻覆盖成 "bbb"，
# A 后面打的日志就带上了 B 的 id，排查时把两次请求混成一次。
# 为什么不逐层传参？那意味着 pipeline、retriever、每个工具函数都要多一个 request_id 参数，
# 业务代码被日志需求污染。contextvars 就是为了解决这个「隐式携带上下文」的问题。
# 文档：https://docs.python.org/3/library/contextvars.html

# ---- 练习 1：声明这个上下文变量 -------------------------------------------
# [核心] 要求：变量名 request_id_var，ContextVar 的名字写 "request_id"，
#        默认值用 "-"（还没进入任何请求时打的日志，比如启动日志，会显示这个）。
request_id_var: ContextVar[str] = ContextVar("request_id", default="-") 
# ---------------------------------------------------------------------------


# ===========================================================================
# 二、把一条日志记录变成一行 JSON
# ===========================================================================
# [脚手架] LogRecord 自带的标准属性名。凡是不在这个集合里的属性，
#          都是调用方通过 logger.info("...", extra={...}) 额外塞进来的业务字段。
_RESERVED = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName relativeCreated
    stack_info thread threadName taskName""".split()
)


def _extra_fields(record: logging.LogRecord) -> dict:
    """[核心] 摘出调用方通过 extra= 传进来的自定义字段。

    吃一个 LogRecord，吐一个 dict。
    例：logger.info("完成", extra={"hits": 4}) 产生的 record 传进来，返回 {"hits": 4}。
    做法是遍历 record 身上挂的所有属性，凡是不在 _RESERVED（标准属性名）里的就是自定义的。
    """
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class JsonFormatter(logging.Formatter):
    """把 LogRecord 格式化成一行 JSON。

    **Formatter（格式化器）**是标准库 logging 三层结构里的最后一层，职责只有一个：
    把日志记录对象变成最终要输出的字符串。换掉它就换掉了日志的样子，
    而业务代码里的 logger.info(...) 一行都不用改——这就是分层的价值。
    """

    # ---- 练习 2：实现格式化 -----------------------------------------------
    # [核心] 要求：返回一行 JSON 字符串，至少包含这几个键
    #          ts         事件时间，用 UTC 的 ISO 格式
    #          level      日志级别名，如 "INFO"
    #          logger     产生这条日志的 logger 名字
    #          msg        真正的日志正文
    #          request_id 从上面那个 ContextVar 里取
    #        另外两个要求：
    #          · 如果这条记录带异常信息，把格式化后的堆栈放进 "exc" 键
    #          · 把 _extra_fields(record) 的内容合并进去
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":datetime.fromtimestamp(record.created, tz = timezone.utc).isoformat(),
            "level":record.levelname,
            "logger":record.name,
            "msg":record.getMessage(),
            "request_id":request_id_var.get()
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        payload.update(_extra_fields(record))
        return json.dumps(payload, ensure_ascii=False)
    # -----------------------------------------------------------------------


# ===========================================================================
# 三、装配
# ===========================================================================
def setup_logging(level: str = "INFO") -> None:
    """[了解] 全局只在应用启动时调用一次。

    标准库 logging 的三层结构：
      Logger（记录器）    —— 业务代码调用的入口，按名字分层，如 "app.rag.pipeline"
      Handler（处理器）   —— 决定日志往哪儿去：标准输出、文件、网络……
      Formatter（格式化器）—— 决定日志长什么样

    容器化部署的惯例是**只往标准输出打**，不自己写文件：
    文件轮转、收集、归档由容器运行时和日志采集器负责，应用不该操心这些。
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    # [了解] 先清空已有 handler。不清的话，如果框架或某个库已经加过一个默认 handler，
    #        每条日志会被打印两遍（一遍 JSON、一遍人话）。
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # [核心] uvicorn（跑 FastAPI 的那个服务器进程）默认自带彩色的人话日志。
    #        清掉它自己的 handler 并让它 propagate（向上冒泡）到根 logger，
    #        它的访问日志才会一起变成 JSON。否则你的日志会是「一半 JSON 一半人话」，
    #        采集器解析时整片报错——这是真实项目里非常常见的一个坑。
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def get_logger(name: str) -> logging.Logger:
    """[脚手架] 统一的取 logger 入口，业务代码用 get_logger(__name__)。"""
    return logging.getLogger(name)
