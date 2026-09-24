"""RAG 主链路 —— 一个问题进来，一个带出处的答案出去。

这条链有四步，全部由我们自己写，没有用任何现成的 RAG 封装：
    检索 -> 拼上下文 -> 调模型 -> 解析引用

为什么不用现成封装？
现成封装会把这四步藏起来。
出了问题，你不知道是检索没捞对，还是提示词没管住模型。
自己写这四步，每一步的输入输出都在手里，能单独调、单独测、单独换。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    Citation            数据类。一条出处：编号、来源文件、章节、距离、原文片段。
    RagAnswer           数据类。一次问答的结果：答案 + 出处列表 + 条数 + 耗时。
    build_model(settings)
        吃配置 -> 吐一个聊天模型对象。被 app/main.py 启动时调一次。
    RagPipeline(store, model, prompt, settings)
        整条链的持有者。启动时造一次，之后每个请求复用同一个实例。
    RagPipeline._retrieve(question)
        吃问题字符串 -> 吐 [(Document, 距离), ...]。被 ask 调用。
    RagPipeline._generate(context, question)
        吃「资料字符串 + 问题字符串」-> 吐模型生成的答案文本。被 ask 调用。
    RagPipeline._pick_citations(answer, hits)
        吃「答案文本 + 检索结果」-> 吐 list[Citation]，只含答案真正引用过的那几条。
        被 ask 调用。
    RagPipeline.ask(question)
        吃问题字符串 -> 吐 RagAnswer。被 HTTP 路由调用，是这个类唯一对外的方法。

阅读地图
    必读：_retrieve 里的信号量、_generate 的超时处理、ask 的六步编排
    扫读：_pick_citations、两个数据类
"""

import asyncio
import re
import time
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import Settings
from app.core.errors import (
    KnowledgeBaseNotReadyError,
    LLMError,
    LLMTimeoutError,
    RetrievalError,
)
from app.core.logging import get_logger
from app.rag.prompt import format_context

logger = get_logger(__name__)

# [了解] 用来从答案里抠出 [1] [2] 这种编号的正则表达式。
#        re 是标准库的正则模块。文档：https://docs.python.org/3/library/re.html
#        \d 表示一个数字，加号表示一个或多个，圆括号表示「把这部分抓出来」。
#        方括号在正则里有特殊含义，所以要用反斜杠转义。
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# [脚手架] 返回给调用方的原文片段截多长。全文返回会让响应体变得很大。
SNIPPET_CHARS = 120


@dataclass(slots=True)
class Citation:
    """一条出处。"""

    no: int          # 答案里标的编号，对应资料段里的 [1] [2]
    source: str      # 来自哪个文件
    section: str     # 哪一章哪一节
    score: float     # 检索时的距离，越小越相关
    snippet: str     # 原文片段，方便用户核对


@dataclass(slots=True)
class RagAnswer:
    """一次问答的完整结果。"""

    answer: str
    citations: list[Citation]
    retrieved: int      # 这次一共检索回来几条
    elapsed_ms: int     # 总耗时，毫秒


def build_model(settings: Settings) -> BaseChatModel:
    """[了解] 造一个聊天模型对象。

    init_chat_model 的第一个参数写成「厂商:模型名」的形式。
    它会自己找到对应的适配包（我们这里是 langchain-deepseek）并实例化。
    好处是换厂商只改配置里的一个字符串，代码一行不动。
    """
    logger.info("加载聊天模型", extra={"model": settings.llm_model})
    return init_chat_model(settings.llm_model, temperature=settings.llm_temperature)


class RagPipeline:
    """把检索和生成串起来。启动时造一次，所有请求共用。"""

    def __init__(
        self,
        store: Chroma,
        model: BaseChatModel,
        prompt: ChatPromptTemplate,
        settings: Settings,
    ) -> None:
        # 步骤 1：把四个依赖存起来。
        #         注意是「传进来」而不是「自己造」。
        #         这样测试时可以塞一个假模型进来，不用真的调外部 API。
        #         这种写法叫依赖注入（dependency injection）。
        self._store = store
        self._model = model
        self._prompt = prompt
        self._settings = settings

        # 步骤 2：造一个信号量，限制同时有几个请求在做向量编码。
        #         信号量（Semaphore）是一个计数器。
        #         拿到许可才能进去，出来时归还，许可用完的人排队等。
        #         这里设成 2，就是最多 2 个请求同时用 GPU 编码。
        #         不限制会怎样？100 个请求同时进来，100 份数据一起塞进显存，直接 OOM。
        self._embed_semaphore = asyncio.Semaphore(settings.embedding_concurrency)

    async def _retrieve(self, question: str) -> list[tuple[Document, float]]:
        """把问题编码成向量，在库里找最接近的 k 条。"""
        # 步骤 1：拿信号量。async with 退出这个代码块时自动归还许可。
        async with self._embed_semaphore:
            try:
                # 步骤 2：调异步检索，返回 [(Document, 距离), ...]。
                #         距离越小越相关。这一点和「相似度越大越好」是反的，容易搞反。
                #
                #         为什么用 a 开头的异步版本？
                #         编码和检索是 CPU/GPU 密集的同步计算。
                #         直接在 async 函数里调同步版本会把整个事件循环卡住，
                #         其它所有请求全部停摆。异步版本内部会把它丢到线程池里跑。
                return await self._store.asimilarity_search_with_score(
                    question, k=self._settings.retrieval_top_k
                )
            except Exception as exc:
                # 步骤 3：把底层异常翻译成我们自己的异常类型。
                #         为什么要翻译？调用方不该知道我们用的是 Chroma。
                #         哪天换成 Milvus，抛的异常类型就变了，上层代码不该跟着改。
                logger.error("检索失败", exc_info=True)
                raise RetrievalError("知识库检索失败") from exc

    async def _generate(self, context: str, question: str) -> str:
        """把资料和问题填进模板，调模型，返回答案文本。"""
        # 步骤 1：把资料和问题填进模板，得到一个可以直接喂给模型的消息列表。
        prompt_value = self._prompt.invoke({"context": context, "question": question})

        try:
            # 步骤 2：在超时保护下调模型。
            #         asyncio.timeout 给这个代码块设一个闹钟。
            #         时间到了还没执行完，就在里面抛 TimeoutError。
            #
            #         为什么必须设超时？
            #         对方卡住不返回，我们这个请求就一直挂着，占着连接和内存。
            #         并发一上来连接池耗尽，整个服务跟着挂。这叫级联故障。
            async with asyncio.timeout(self._settings.llm_timeout_seconds):
                message = await self._model.ainvoke(prompt_value)
        except TimeoutError as exc:
            # 步骤 3：超时单独处理。
            #         因为它对应的 HTTP 状态码和别的错不一样：
            #         超时是 504（我等上游超时了），其它错是 502（上游给了无效响应）。
            #         调用方看到 504 可以决定重试，看到 502 该去查上游。
            #
            #         raise ... from exc 把原始异常挂在新异常上。
            #         不写 from 的话，日志里只有「生成超时」，看不到真正的病根。
            logger.error("生成超时", exc_info=True)
            raise LLMTimeoutError("生成超时，请稍后重试") from exc
        except Exception as exc:
            # 步骤 4：限流、余额不足、网络错误全归到这一类。
            #
            #         捕获 Exception 这么宽，一般是坏习惯，但边界层是例外。
            #         这里是「我们的代码」和「外部 API」的交界，
            #         外部可能抛网络库的、SDK 的、JSON 解析的异常，穷举不完。
            #         边界层的职责就是把外面的一切意外收敛成我们自己的异常类型。
            #         关键是不能吞掉：这里既记了完整堆栈，又用 from 挂上了原始异常。
            logger.error("生成失败", exc_info=True)
            raise LLMError("生成失败") from exc

        # 步骤 5：message 是一个 AIMessage 对象，.text 取出里面的纯文本。
        return message.text

    def _pick_citations(
        self, answer: str, hits: list[tuple[Document, float]]
    ) -> list[Citation]:
        """[了解] 从答案里找出引用了哪些编号，映射回对应的块。

        为什么不把检索到的 4 条全部当出处返回？
        因为模型可能只用了其中 1 条。
        把没用到的也标成出处，用户点进去发现对不上，信任就没了。
        这一步实际上是在核对：模型说的每一句到底有没有依据。
        """
        # 步骤 1：用正则把答案里所有 [数字] 抓出来，得到字符串列表如 ["1", "3", "1"]。
        raw = CITATION_PATTERN.findall(answer)

        # 步骤 2：去重但保持出现顺序。
        #         dict.fromkeys 是个常用技巧。字典的键天然去重，
        #         而且 Python 3.7 起字典保持插入顺序。
        #         直接用 set 会丢掉顺序，出处的排列就变得随机。
        seen_numbers = list(dict.fromkeys(int(n) for n in raw))

        citations: list[Citation] = []
        for no in seen_numbers:
            # 步骤 3：编号从 1 开始，列表下标从 0 开始，所以要减 1。
            #         模型偶尔会编出一个不存在的编号（只给了 4 条却引用 [7]），
            #         所以这里要判断范围，越界就跳过并记日志。
            if not 1 <= no <= len(hits):
                logger.warning("答案引用了不存在的编号", extra={"no": no})
                continue

            doc, score = hits[no - 1]
            meta = doc.metadata
            section = " / ".join(
                x for x in (meta.get("h1"), meta.get("h2"), meta.get("h3")) if x
            )
            citations.append(
                Citation(
                    no=no,
                    source=meta.get("source", "未知来源"),
                    section=section,
                    score=float(score),
                    snippet=doc.page_content.strip()[:SNIPPET_CHARS],
                )
            )
        return citations

    async def ask(self, question: str) -> RagAnswer:
        """对外唯一的方法：问一个问题，拿一个带出处的答案。"""
        # [核心] 这个方法本身不干活。
        #        它只按顺序调用前面那三个方法，并记录耗时和日志。
        #        这种「只编排、不实现」的函数在企业代码里很常见，
        #        好处是整条业务流程一眼能看完。

        # 步骤 1：开始计时。
        #         perf_counter 是标准库 time 里专门测耗时的计时器。
        #         为什么不用 time.time()？后者是墙上时钟，系统对时会让它跳变甚至倒退，
        #         用它测时间差偶尔会算出负数。perf_counter 单调递增，只用来测差值。
        started = time.perf_counter()

        # 步骤 2：检索。
        hits = await self._retrieve(question)

        # 步骤 3：库是空的，说明还没入库。
        #         这不是用户的错，所以要用一个能让运维看懂的错误码（503）。
        if not hits:
            raise KnowledgeBaseNotReadyError("知识库为空，请先执行入库命令")

        # 步骤 4：拼上下文，调模型。
        context = format_context([doc for doc, _ in hits])
        answer = await self._generate(context, question)

        # 步骤 5：核对答案实际引用了哪几条。
        citations = self._pick_citations(answer, hits)

        # 步骤 6：记录耗时并返回。
        #         retrieved 和 cited 要分开记：
        #         前者是检索回来几条，后者是答案真正用了几条，差值就是浪费掉的 token。
        #         线上统计这个差值，就能用数据判断 k 该调大还是调小、要不要上重排。
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "问答完成",
            extra={
                "retrieved": len(hits),
                "cited": len(citations),
                "elapsed_ms": elapsed_ms,
            },
        )
        return RagAnswer(
            answer=answer,
            citations=citations,
            retrieved=len(hits),
            elapsed_ms=elapsed_ms,
        )
