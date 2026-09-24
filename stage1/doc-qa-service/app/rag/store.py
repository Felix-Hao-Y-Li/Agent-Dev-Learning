"""向量库层 —— 存向量，并支持「给我一个向量，找出最像的 K 条」。

**向量库（vector store）**：专门存向量的数据库。
它的核心能力是最近邻搜索：给它一个向量，它快速找出库里最接近的那几条。

朴素做法是拿查询向量和库里每一条都算一遍相似度。
十万条就是十万次计算。
向量库用特殊的索引结构把这件事加速几十到上百倍。

本项目用 Chroma。
离线侧（ingest 命令）往里写，在线侧（FastAPI 服务）只读。
原因写在 README 的 ADR-003 里：Chroma 的持久化客户端不支持多进程并发写。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    build_store(settings, embeddings)
        吃「配置对象 + 嵌入模型对象」-> 吐一个 Chroma 实例。
        被 app/ingest.py 和 app/main.py 各调一次。
    count_documents(store)
        吃一个 Chroma 实例 -> 吐库里有多少条记录，整数。
        被 /readyz 就绪检查和 ingest 命令调用。

阅读地图
    必读：build_store 的三个参数分别决定什么
    扫读：count_documents
"""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def build_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    """连接（或创建）本地 Chroma 库。"""
    # ---- 练习 2：连接向量库 -----------------------------------------------
    # [核心] 要求写两个步骤：
    #   步骤 1  确保目录存在：settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    #           parents=True 表示父目录不存在就一并创建。
    #           exist_ok=True 表示目录已存在时不报错。
    #   步骤 2  构造并返回 Chroma，传三个参数：
    #           collection_name=settings.chroma_collection
    #           embedding_function=embeddings
    #           persist_directory=str(settings.chroma_dir)
    #
    # [核心] 这三个参数分别决定什么：
    #   collection（集合）相当于关系数据库里的「表」。
    #       一个 Chroma 库里可以有多个集合，互相隔离，检索只在指定集合内进行。
    #   embedding_function 是嵌入模型对象。
    #       传了它之后，写入时 Chroma 自动把文字编码成向量，检索时自动编码查询。
    #       你不需要手动调 embed_documents 或 embed_query。
    #   persist_directory 是数据落在硬盘上的位置。
    #       **不传这个参数会怎样？数据只存在内存里，进程一退出就全没了。**
    # 步骤1
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    # 步骤2
    logger.info(
        "连接向量库",
        extra={
            "dir": str(settings.chroma_dir),
            "collection": settings.chroma_collection,}
    )
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    # -----------------------------------------------------------------------


def count_documents(store: Chroma) -> int:
    """数一数库里现在有多少条记录。"""
    # 步骤 1：store.get() 会把库里的记录取出来，返回一个字典。
    #         include=[] 表示「只要 id，别把正文和向量也读出来」。
    #         不加这个参数的话，库大了会把几百兆数据读进内存，非常慢。
    result = store.get(include=[])

    # 步骤 2："ids" 这个键下面是全部记录的 id 列表，数一下长度就是总条数。
    return len(result["ids"])
