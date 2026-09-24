"""HTTP 层的测试 —— 打真实的接口，但底下是假模型。

这些测试覆盖的是「协议层对不对」：状态码、响应格式、校验、错误结构。
业务逻辑的正确性由 test_pipeline.py 负责，这里不重复测。

阅读地图
    必读：test_error_response_shape_is_unified（统一错误格式的意义）
          test_blank_question_rejected（校验必须挡在业务之前）
"""

from httpx import AsyncClient


async def test_healthz_is_always_ok(client: AsyncClient):
    """存活探针必须又快又稳，不依赖任何外部资源。"""
    resp = await client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_readyz_reports_document_count(client: AsyncClient):
    """就绪探针要真的去查知识库，并报出条数。"""
    resp = await client.get("/readyz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["documents"] == 3      # conftest 里放了 3 条测试数据


async def test_ask_happy_path(client: AsyncClient):
    """正常提问：200，带答案、出处、耗时、追踪编号。"""
    resp = await client.post("/v1/ask", json={"question": "退货运费谁承担"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "运费由平台承担 [1]。"
    assert body["retrieved"] == 3
    assert len(body["citations"]) == 1

    # 出处的五个字段都要在，缺一个前端就展示不全。
    citation = body["citations"][0]
    assert set(citation) == {"no", "source", "section", "score", "snippet"}

    # 追踪编号既要在响应体里，也要在响应头里，而且必须是同一个。
    assert body["request_id"]
    assert resp.headers["X-Request-ID"] == body["request_id"]


async def test_request_id_is_propagated_from_header(client: AsyncClient):
    """上游传了追踪编号就沿用，不要自己另发一个。

    微服务架构里，一个用户操作会穿过好几个服务。
    编号一路传下去，所有服务的日志才能串成一条完整链路。
    """
    resp = await client.post(
        "/v1/ask",
        json={"question": "退货运费"},
        headers={"X-Request-ID": "trace-from-gateway"},
    )

    assert resp.headers["X-Request-ID"] == "trace-from-gateway"
    assert resp.json()["request_id"] == "trace-from-gateway"


async def test_blank_question_rejected(client: AsyncClient):
    """只有空白字符的问题要被挡在业务之前。

    如果不挡，就会拿着空问题去检索、去调模型，白烧一次钱。
    """
    resp = await client.post("/v1/ask", json={"question": "   "})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


async def test_too_long_question_rejected(client: AsyncClient):
    """超长问题要被挡住，防止有人贴一整本书进来。"""
    resp = await client.post("/v1/ask", json={"question": "退" * 5000})

    assert resp.status_code == 422


async def test_missing_field_rejected(client: AsyncClient):
    """请求体里没有 question 字段。"""
    resp = await client.post("/v1/ask", json={})

    assert resp.status_code == 422


async def test_error_response_shape_is_unified(client: AsyncClient):
    """所有错误响应必须长成同一个形状。

    形状是 {"error": {"code", "message", "request_id"}}。
    统一之后，前端写一套解析就够；不统一的话，每个接口都得写一套分支。
    """
    # 用一个不存在的路径触发 404，这是框架层的错误，不是我们业务抛的。
    resp = await client.get("/does-not-exist")

    assert resp.status_code == 404
    body = resp.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id"}


async def test_openapi_schema_is_generated(client: AsyncClient):
    """接口文档能正常生成。

    这条测试的价值在于：契约类写错了（比如引用了不存在的类型），
    /openapi.json 会直接报错，而这个问题平时跑接口是发现不了的。
    """
    resp = await client.get("/openapi.json")

    assert resp.status_code == 200
    schema = resp.json()
    assert "/v1/ask" in schema["paths"]
    assert "/healthz" in schema["paths"]
