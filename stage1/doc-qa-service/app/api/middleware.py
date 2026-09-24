"""HTTP 中间件 —— 每个请求进出时都要做的事。

**中间件（middleware）**：夹在「框架收到请求」和「路由函数被调用」之间的一层代码。
每个请求都会穿过它，进来穿一次，出去再穿一次。
适合放那些「所有接口都要做、但和业务无关」的事：发追踪编号、记访问日志、统计耗时。

不用中间件会怎样？每个路由函数开头都得手写一遍这些代码，漏掉一个接口就少一段日志。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    RequestIdMiddleware
        一个中间件类。在 app/main.py 里用 app.add_middleware(...) 装上。
    RequestIdMiddleware.dispatch(request, call_next)
        吃「请求对象 + 调用下一层的函数」-> 吐响应对象。
        框架对每个请求自动调用它，我们不会手动调。

阅读地图
    必读：dispatch 的六个步骤，尤其是 contextvar 的设置和归还
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)

# [了解] 上游传进来的追踪编号放在这个请求头里。
#        这是业界惯例的写法，网关、负载均衡器、服务网格大多认这个名字。
REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """给每个请求发一个追踪编号，并记一条访问日志。"""

    async def dispatch(self, request: Request, call_next) -> Response:
        # 步骤 1：先看上游有没有给编号。
        #         微服务架构里，一个用户操作会穿过好几个服务。
        #         网关生成一个编号，一路往下传，所有服务的日志用同一个编号串起来，
        #         这样才能追出「这次请求在哪个服务、哪一步慢了」。
        #         上游没给（比如你自己用 curl 直接调）就自己生成一个。
        #
        #         uuid4().hex 生成一个 32 位的十六进制字符串，重复概率可以忽略。
        #         [:16] 是截短一半，日志里更好读，对我们的量级来说仍然够用。
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]

        # 步骤 2：把编号放进上下文变量。
        #         set() 会返回一个 token（令牌），拿着它可以把变量还原成之前的值。
        #         为什么要还原？因为 asyncio 的任务可能复用同一个上下文，
        #         不还原的话，下一个请求可能读到上一个请求残留的编号。
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            # 步骤 3：调用下一层。
            #         call_next 就是「后面所有中间件 + 真正的路由函数」。
            #         await 它，拿到最终的响应对象。
            response = await call_next(request)

            # 步骤 4：把编号写进响应头，调用方能直接拿到。
            #         前端出错时把这个头的值报给后端，问题就能定位。
            response.headers[REQUEST_ID_HEADER] = request_id

            # 步骤 5：记一条访问日志。
            #         这几个字段是排查问题的最小集合：访问什么、结果如何、花了多久。
            #         用 extra 把它们做成独立字段，而不是拼成一句话，
            #         这样日志系统才能按 status 过滤、按 duration_ms 排序。
            #
            #         注意这一句写在 finally 之前。
            #         此刻上下文变量还没归还，JsonFormatter 会自动带上 request_id，
            #         不用在 extra 里手动再传一遍。
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "请求完成",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            return response
        finally:
            # 步骤 6：不管成功还是抛异常，都要把上下文变量还原。
            #         为什么要还原？asyncio 的任务可能复用同一个上下文，
            #         不还原的话，下一个请求可能读到上一个请求残留的编号。
            #         finally 保证这一句一定会执行。
            request_id_var.reset(token)
