"""嵌入层 —— 把文字变成一串数字。

**嵌入（embedding）**：把一段文字变成一个固定长度的数字列表。
比如「七天无理由退货」经过模型处理后，变成 512 个小数组成的列表。
这个列表叫**向量（vector）**。

为什么要这么做？计算机没法直接比较两句话的意思。
但它可以比较两个数字列表有多接近。
模型训练的目标就是：意思相近的两句话，算出来的两个列表也相近。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    build_embeddings(settings)
        吃配置对象 -> 吐一个 HuggingFaceEmbeddings 实例。
        被 app/ingest.py（入库时）和 app/main.py（服务启动时）各调一次。

阅读地图
    必读：build_embeddings 里文档端和查询端的区别
"""

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """加载本地 BGE 模型，返回一个可以把文字转成向量的对象。

    这个对象有两个常用方法，后面会用到：
        embed_documents(["文字1", "文字2"])  -> [[0.1, 0.2, ...], [0.3, ...]]
        embed_query("一个问题")              -> [0.5, 0.1, ...]
    """
    # ---- 练习 1：构造嵌入模型 ---------------------------------------------
    # [核心] 这里最关键的一件事：**文档和查询用的编码方式不一样**。
    #
    #        BGE 这个模型系列有个硬性要求。
    #        给它一个查询时，要在前面拼一句指令：「为这个句子生成表示以用于检索相关文章：」。
    #        给它一段文档时，不能拼这句话。
    #        这是模型训练时就定好的规矩，写在它的模型卡上。
    #
    #        两边搞反或者漏掉会怎样？不报错。检索结果会变差，但你收不到任何信号。
    #        这是 RAG 里最难查的一类问题。
    #
    # [核心] 要求写三个步骤：
    #   步骤 1  准备文档端参数 doc_kwargs = {"normalize_embeddings": True}
    #           normalize（归一化）的意思是把向量的长度缩放到 1。
    #           这样之后算相似度时，点积就直接等于余弦相似度，省一步计算。
    #   步骤 2  准备查询端参数 query_kwargs，在步骤 1 的基础上多一个键：
    #           "prompt": settings.embedding_query_prefix
    #           prompt 是 sentence-transformers 库 encode() 方法的参数。
    #           它的作用就是把这段文字拼到每条输入前面。
    #   步骤 3  构造并返回 HuggingFaceEmbeddings，传四个参数：
    #           model_name=settings.embedding_model
    #           model_kwargs={"device": settings.embedding_device}
    #           encode_kwargs=doc_kwargs            <- 作用于文档
    #           query_encode_kwargs=query_kwargs    <- 作用于查询
    # 文档端的编码参数
    doc_kwargs = {"normalize_embeddings": True}
    # 查询端的编码参数，比文档端多一个 prompt，告诉模型这是一个查询
    query_kwargs = {"normalize_embeddings": True,
                    "prompt": settings.embedding_query_prefix,}
    # 步骤 3：构造 HuggingFaceEmbeddings
    logger.info(
        "加载嵌入模型",
        extra={
            "model_name": settings.embedding_model,
            "device": settings.embedding_device,
        },
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs=doc_kwargs,
        query_encode_kwargs=query_kwargs,
    )
    # -----------------------------------------------------------------------
