"""测试的公共装置。

**conftest.py** 是 pytest 的约定文件名。
放在 tests/ 目录下，里面定义的 fixture 会自动被同目录（及子目录）的所有测试用到，
不需要 import。

**fixture（测试装置）**：一个用 @pytest.fixture 装饰的函数，
负责为测试准备好它需要的东西。
测试函数只要把 fixture 的名字写成参数，pytest 就会自动调用它并把结果传进来。

这个文件最重要的设计目标：**测试必须能在没有 GPU、没有网络、没有 API key 的机器上跑通**。
为什么？因为测试要在 CI（持续集成，代码推上去自动跑的流水线）里跑，
那种机器上没有显卡，也不该把真实 API key 放上去。
而且真调模型的话，一次测试几秒钟、还要花钱，没人会愿意频繁跑。

做法是用「假的」替身换掉两个重依赖：
    FakeEmbeddings  替掉本地 bge 模型，不加载 torch
    FakeChatModel   替掉 DeepSeek，不联网

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    FakeEmbeddings          假嵌入模型。把文字变成 8 维的确定性向量。
    FakeChatModel           假聊天模型。返回事先设定好的答案。
    settings                fixture，指向临时目录的配置对象。
    store                   fixture，装了 3 条测试数据的临时向量库。
    fake_model              fixture，一个 FakeChatModel 实例。
    pipeline                fixture，用假依赖装起来的 RagPipeline。
    client                  fixture，能直接打接口的异步 HTTP 客户端。

阅读地图
    必读：FakeChatModel 为什么能顶替真模型、client fixture 为什么不触发 lifespan
    扫读：FakeEmbeddings 的向量算法
"""

import os

# [核心] 这三行必须写在 import app.* 之前。
#        原因：app/schemas/ask.py 和 app/core/config.py 在**被导入时**就会读配置，
#        那一刻环境变量必须已经就位，否则会因为缺少 DEEPSEEK_API_KEY 直接报错。
#        setdefault 表示「本来没有才设」，所以本机有 .env 时不会被覆盖成测试值。
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-used")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402
from app.rag.prompt import build_prompt  # noqa: E402

# 测试用的三条语料，内容短、互相区别明显，方便断言。
TEST_DOCS = [
    Document(
        page_content="退货运费：质量问题由平台承担，非质量问题由买家承担。",
        metadata={"source": "policy.md", "h1": "售后", "h2": "退货运费"},
    ),
    Document(
        page_content="发货时效：每日 15:00 前付款的订单当日发出。",
        metadata={"source": "shipping.md", "h1": "物流", "h2": "发货时效"},
    ),
    Document(
        page_content="发票：电子普票在订单完成后 24 小时内开具。",
        metadata={"source": "shipping.md", "h1": "物流", "h2": "发票"},
    ),
]


class FakeEmbeddings(Embeddings):
    """[核心] 假的嵌入模型，不加载任何真实模型。

    它只需要满足 Embeddings 接口的两个方法：
        embed_documents(文本列表) -> 向量列表
        embed_query(单个文本)     -> 一个向量

    向量怎么算无所谓，只要满足两个条件：
        确定性  —— 同样的文字每次算出同样的向量，否则测试结果会随机波动。
        有区分度 —— 不同的文字算出不同的向量，否则检索结果全是并列。

    这里用最简单的办法：数几个关键词各出现了多少次，凑成 8 维向量。
    """

    KEYWORDS = ["退货", "运费", "发货", "时效", "发票", "平台", "买家", "订单"]

    def _vector(self, text: str) -> list[float]:
        # 步骤 1：对每个关键词，数它在文本里出现了几次，作为这一维的值。
        return [float(text.count(kw)) for kw in self.KEYWORDS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeChatModel:
    """[核心] 假的聊天模型，不联网。

    为什么一个这么简单的类就能顶替真模型？
    因为 RagPipeline 只用到模型的一个方法：await self._model.ainvoke(prompt_value)。
    Python 不检查类型，只要这个对象有 ainvoke 方法、返回的东西有 .text 属性，就能用。
    这叫**鸭子类型（duck typing）**：像鸭子一样走路、像鸭子一样叫，就当它是鸭子。

    它还多做两件测试里很有用的事：
        把每次收到的提示词存进 self.calls，测试可以检查「到底发了什么给模型」。
        可以设定成抛异常，用来测试超时和失败分支。
    """

    def __init__(self, answer: str = "运费由平台承担 [1]。", error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[object] = []

    async def ainvoke(self, prompt_value, **kwargs) -> AIMessage:
        # 步骤 1：把这次收到的提示词记下来，供测试断言。
        self.calls.append(prompt_value)
        # 步骤 2：设定了异常就抛出来，用来测试错误分支。
        if self.error is not None:
            raise self.error
        # 步骤 3：返回一个 AIMessage，和真模型返回的类型一致。
        return AIMessage(content=self.answer)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """一份指向临时目录的配置。

    tmp_path 是 pytest 内置的 fixture，为每个测试函数创建一个全新的空目录，
    测试结束后自动清理。用它就不会污染项目里真正的 chroma_db/。
    """
    return Settings(
        deepseek_api_key="test-key-not-used",
        chroma_dir=tmp_path / "chroma",
        chroma_collection="test",
        data_dir=tmp_path / "data",
        embedding_device="cpu",
        retrieval_top_k=3,
        llm_timeout_seconds=5.0,
    )


@pytest.fixture
def store(settings: Settings) -> Chroma:
    """一个装了 3 条测试数据的临时向量库。"""
    # 步骤 1：用假嵌入模型建库，全程不碰 torch。
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    s = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=FakeEmbeddings(),
        persist_directory=str(settings.chroma_dir),
    )
    # 步骤 2：写入测试数据，id 固定，重复跑也不会堆数据。
    s.add_documents(TEST_DOCS, ids=[f"doc-{i}" for i in range(len(TEST_DOCS))])
    return s


@pytest.fixture
def fake_model() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture
def pipeline(store: Chroma, fake_model: FakeChatModel, settings: Settings) -> RagPipeline:
    """用假依赖装起来的真 pipeline。

    注意 RagPipeline 本身是真的，被换掉的只是它的两个外部依赖。
    这正是构造函数接收依赖（依赖注入）带来的好处。
    """
    return RagPipeline(store, fake_model, build_prompt(), settings)


@pytest.fixture
async def client(pipeline: RagPipeline, store: Chroma):
    """一个能直接打接口的异步 HTTP 客户端。"""
    from app.main import create_app

    # 步骤 1：造一个全新的应用实例。
    app = create_app()

    # 步骤 2：手动把假依赖挂到 app.state 上。
    #         正常运行时这是 lifespan 干的活，见 app/main.py。
    app.state.store = store
    app.state.pipeline = pipeline

    # 步骤 3：用 ASGITransport 直接把请求送进应用，不经过网络。
    #         **关键点**：这种方式不会触发 lifespan，
    #         所以不会去加载真实的 bge 模型和 DeepSeek 客户端。
    #         如果用 starlette 的 TestClient，进入上下文时会执行 lifespan，
    #         测试就会变慢并且需要 GPU 和网络。
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
