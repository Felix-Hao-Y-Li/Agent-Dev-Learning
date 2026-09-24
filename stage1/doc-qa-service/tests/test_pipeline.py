"""业务层的测试 —— 用假模型，不联网、不用 GPU。

这些测试覆盖的是「RAG 链路本身对不对」，和 HTTP 无关。
能这样单独测，正是因为 pipeline 不认识 HTTP。

阅读地图
    必读：test_citations_only_include_referenced（引用核对是 RAG 的信任基础）
          test_timeout_maps_to_llm_timeout_error（超时必须映射成正确的异常类型）
"""

import asyncio

import pytest
from langchain_chroma import Chroma

from app.core.errors import KnowledgeBaseNotReadyError, LLMError, LLMTimeoutError
from app.rag.pipeline import RagPipeline
from app.rag.prompt import build_prompt

from .conftest import FakeChatModel, FakeEmbeddings


async def test_ask_returns_answer_and_citations(pipeline: RagPipeline):
    """正常问答：返回答案，并带上被引用的出处。"""
    result = await pipeline.ask("退货运费谁承担")

    assert result.answer == "运费由平台承担 [1]。"
    assert result.retrieved == 3          # settings 里 top_k 设的是 3
    assert len(result.citations) == 1     # 答案只引用了 [1]
    assert result.citations[0].no == 1
    assert result.elapsed_ms >= 0


async def test_prompt_actually_contains_retrieved_text(
    pipeline: RagPipeline, fake_model: FakeChatModel
):
    """检索到的原文必须真的出现在发给模型的提示词里。

    这条测试守的是 RAG 的命门。
    如果检索结果没被拼进提示词，模型就是在凭空作答，
    但表面上一切正常，你看不出任何异常。
    """
    await pipeline.ask("退货运费")

    # 步骤 1：假模型把收到的提示词记下来了，取出来看。
    sent = fake_model.calls[0].to_string()

    # 步骤 2：里面必须有检索到的原文，也必须有编号标记。
    assert "退货运费" in sent
    assert "[1]" in sent
    assert "【问题】" in sent


async def test_citations_only_include_referenced(
    store: Chroma, settings, fake_model: FakeChatModel
):
    """答案没引用的块，不能出现在出处里。

    为什么重要：检索回来 3 条，模型可能只用了 1 条。
    把没用到的也标成出处，用户点进去发现对不上，信任就没了。
    """
    # 步骤 1：让假模型返回一个只引用 [2] 的答案。
    model = FakeChatModel(answer="当日发出 [2]。")
    pipe = RagPipeline(store, model, build_prompt(), settings)

    # 步骤 2：出处里应该只有 2 号。
    result = await pipe.ask("发货时效")
    assert [c.no for c in result.citations] == [2]


async def test_invalid_citation_number_is_ignored(
    store: Chroma, settings
):
    """模型编出一个不存在的编号时，要跳过而不是崩掉。"""
    # 步骤 1：只检索 3 条，却让模型引用 [9]。
    model = FakeChatModel(answer="见资料 [9]。")
    pipe = RagPipeline(store, model, build_prompt(), settings)

    # 步骤 2：不抛异常，出处为空。
    result = await pipe.ask("发票")
    assert result.citations == []


async def test_timeout_maps_to_llm_timeout_error(store: Chroma, settings):
    """模型超时要转成 LLMTimeoutError，而不是把原始异常漏出去。

    为什么讲究：LLMTimeoutError 对应 504，其它模型错误对应 502。
    调用方靠这个区分「可以重试」和「上游坏了」。
    """
    # 步骤 1：让假模型抛超时异常。
    model = FakeChatModel(error=TimeoutError())
    pipe = RagPipeline(store, model, build_prompt(), settings)

    # 步骤 2：pytest.raises 断言「这段代码必须抛出指定类型的异常」。
    with pytest.raises(LLMTimeoutError):
        await pipe.ask("退货运费")


async def test_other_model_error_maps_to_llm_error(store: Chroma, settings):
    """模型的其它错误转成 LLMError。"""
    model = FakeChatModel(error=RuntimeError("余额不足"))
    pipe = RagPipeline(store, model, build_prompt(), settings)

    with pytest.raises(LLMError):
        await pipe.ask("退货运费")


async def test_empty_store_raises_not_ready(settings, fake_model: FakeChatModel):
    """库是空的时候要明确报「未就绪」，而不是让模型对着空资料瞎编。"""
    # 步骤 1：建一个空库。
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    empty = Chroma(
        collection_name="empty",
        embedding_function=FakeEmbeddings(),
        persist_directory=str(settings.chroma_dir),
    )
    pipe = RagPipeline(empty, fake_model, build_prompt(), settings)

    with pytest.raises(KnowledgeBaseNotReadyError):
        await pipe.ask("随便问点什么")


async def test_concurrent_requests_do_not_deadlock(pipeline: RagPipeline):
    """并发请求不会互相卡死。

    这条是在验证信号量用对了：
    信号量限制的是「同时进去几个」，不是「总共只能进去几个」，
    写错的话第 3 个请求会永远等下去。
    """
    # asyncio.gather 同时发起 5 个请求，全部完成后返回结果列表。
    results = await asyncio.gather(*[pipeline.ask("退货运费") for _ in range(5)])
    assert len(results) == 5
    assert all(r.answer for r in results)
