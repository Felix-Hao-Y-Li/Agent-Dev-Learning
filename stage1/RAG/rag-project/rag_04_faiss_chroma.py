"""
RAG Part 2 之二：用 FAISS / Chroma 建本地向量索引

流程接上一课：加载 -> 切分 -> 【向量化 -> 存进向量库 -> 检索】

运行方式：
    uv run rag_04_faiss_chroma.py

会在当前目录生成两个索引目录（都已加进 .gitignore）：
    faiss_index/     FAISS 索引（两个文件：index.faiss + index.pkl）
    chroma_db/       Chroma 索引（一个 SQLite 数据库文件 + 若干二进制文件）
"""

# ---------------------------------------------------------------------------
# 名词先解释清楚
# ---------------------------------------------------------------------------
# **向量库 / 向量数据库（vector store / vector database）**：专门存向量、并且能
#   「给我一个向量，快速找出库里和它最像的 K 个」的存储系统。
#   为什么需要它？因为朴素做法是把查询向量和库里每一条都算一遍相似度，
#   十万条就是十万次计算。向量库用特殊的索引结构把这件事加速几十上百倍。
#
# **索引（index）**：为了加快查找而额外建立的数据结构。和数据库里给某一列建索引是同一个意思。
#
# **FAISS**：Facebook AI Research 开源的向量检索库（名字是 Facebook AI Similarity Search
#   的缩写）。它是一个**库**不是**服务**——没有服务器进程，索引就是硬盘上的两个文件，
#   全部加载进内存来查。快、轻、零运维，但不支持并发写、没有权限管理。
#
# **Chroma**：一个面向 AI 应用的开源向量数据库。相比 FAISS 多了「集合（collection）」、
#   自动持久化、按元数据过滤等数据库特性，底层用 SQLite 存元数据，用起来更像一个真数据库。
#
# **K 近邻搜索（KNN, K-Nearest Neighbors）**：给一个查询向量，返回库里最接近的 K 条。
#   RAG 检索环节干的就是这件事，那个 K 就是代码里的参数 k。

from pathlib import Path
import shutil                # 标准库，用来删除整个目录树。文档：https://docs.python.org/3/library/shutil.html

import faiss                 # FAISS 的 Python 绑定，由 faiss-cpu 这个包提供
import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# langchain_community 里的 FAISS 封装。
# 【重要的时效性提醒】langchain-community 这个包已于 2026 年被官方 sunset（停止维护、
# 仓库归档），import 时会打印一条 DeprecationWarning。官方的方向是「每个集成拆成
# 独立的包」，但 FAISS 至今没有官方独立包，所以想用 LangChain 封装的 FAISS 就只能走这里。
# 面试被问到要能说清这个现状。下面第 3 步还会演示「不依赖这个包」的原生写法。
from langchain_community.vectorstores import FAISS

# Chroma 则有官方维护的独立包 langchain-chroma，这是当前推荐的用法。
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
MD_PATH = BASE_DIR / "data" / "sample_zh.md"
FAISS_DIR = BASE_DIR / "faiss_index"
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def oneline(text: str, n: int = 44) -> str:
    """把块内容压成单行并截断，纯粹为了终端输出好读。"""
    return text.replace("\n", " ")[:n] + "..."


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# 第 0 步：准备语料（复用上一课的加载 + 切分）
# ===========================================================================
def build_chunks() -> list[Document]:
    """加载 Markdown、按标题切、再按长度切，返回可以入库的块列表。"""
    banner("第 0 步：准备语料（加载 -> 切分）")

    raw = MD_PATH.read_text(encoding="utf-8")
    base_doc = Document(page_content=raw, metadata={"source": MD_PATH.name})

    header_chunks = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    ).split_text(base_doc.page_content)
    for c in header_chunks:
        c.metadata = base_doc.metadata | c.metadata

    chunks = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=200,
        chunk_overlap=40,
        add_start_index=True,
    ).split_documents(header_chunks)

    # 沿用上一课的质量闸门：丢掉太短的碎块
    chunks = [c for c in chunks if len(c.page_content) >= 20]

    # Chroma 的元数据只接受 str / int / float / bool / None 这几种简单类型，
    # 不接受嵌套的字典或列表。这里的 metadata 恰好都是字符串，所以没问题；
    # 但如果你往里塞过一个列表（比如 tags: ["a","b"]），入库时会直接报错。
    print(f"{MD_PATH.name} -> {len(chunks)} 块可入库")
    return chunks


def build_embeddings() -> HuggingFaceEmbeddings:
    """创建嵌入模型，查询端自动加 BGE 的指令前缀。"""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cuda"},
        # encode_kwargs 作用于**文档**：只归一化，不加任何前缀
        encode_kwargs={"normalize_embeddings": True},
        # query_encode_kwargs 作用于**查询**：额外加上 BGE 要求的指令前缀。
        # prompt 是 sentence-transformers 的 encode() 参数，作用就是把这段文字
        # 拼到每条输入的前面。这样一来，向量库内部调用 embed_query 时会自动带上前缀，
        # 你在业务代码里不用手动拼字符串，也就不会忘。
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": BGE_QUERY_INSTRUCTION,
        },
    )


# ===========================================================================
# 第 1 步：不用 LangChain，直接用原生 FAISS 走一遍（理解索引到底是什么）
# ===========================================================================
def raw_faiss_demo(chunks: list[Document], emb: HuggingFaceEmbeddings) -> None:
    banner("第 1 步：原生 FAISS —— 索引里到底发生了什么")

    texts = [c.page_content for c in chunks]
    # embed_documents 一次编码所有文本，返回 list[list[float]]。
    # np.array(..., dtype="float32") 转成 NumPy 数组：FAISS 只吃 float32，
    # 传 float64 会直接报错。这是新手最常见的第一个坑。
    vecs = np.array(emb.embed_documents(texts), dtype="float32")
    dim = vecs.shape[1]     # shape 是 (条数, 维度)，[1] 取维度
    print(f"向量矩阵形状：{vecs.shape}（{vecs.shape[0]} 条 × {dim} 维）")

    # --- 索引类型一：IndexFlatL2 ---------------------------------------------
    # Flat 的意思是「不做任何压缩和近似，老老实实存原始向量、逐条比对」。
    # L2 指欧氏距离（Euclidean distance），也就是两点之间的直线距离。
    # 特点：结果 100% 精确，但查询耗时随数据量线性增长。
    # 距离越小越相似 —— 注意这和「相似度越大越相似」是反的。
    index_l2 = faiss.IndexFlatL2(dim)
    index_l2.add(vecs)      # 把所有向量灌进索引

    # --- 索引类型二：IndexFlatIP ---------------------------------------------
    # IP = Inner Product（内积，也就是点积）。分数越大越相似。
    # 上一个脚本证明过：向量归一化之后，内积在数值上等于余弦相似度。
    # 因为我们设了 normalize_embeddings=True，所以这里的 IP 就是余弦相似度。
    index_ip = faiss.IndexFlatIP(dim)
    index_ip.add(vecs)

    print(f"索引里存了多少条：L2 索引 {index_l2.ntotal} 条，IP 索引 {index_ip.ntotal} 条")

    query = "文档切成多大合适"
    # 查询向量也必须是 float32，而且必须是二维数组（一批查询），
    # 所以用 [query_vec] 包一层。只查一条也要包，这是 FAISS 的接口约定。
    qv = np.array([emb.embed_query(query)], dtype="float32")

    k = 3  # 要返回最相似的几条
    # search 返回两个数组：
    #   distances —— 距离/分数，形状 (查询条数, k)
    #   indices   —— 命中向量在索引里的**下标**，形状同上
    # 注意 FAISS 只还给你下标，它根本不知道原文是什么——
    # 「下标 → 原文」的映射得你自己维护。LangChain 封装帮你干的主要就是这件事。
    d_l2, i_l2 = index_l2.search(qv, k)
    d_ip, i_ip = index_ip.search(qv, k)

    print(f"\n查询：{query}")
    print(f"\nIndexFlatL2（欧氏距离，越小越像）：")
    for rank, (dist, idx) in enumerate(zip(d_l2[0], i_l2[0]), start=1):
        print(f"    第{rank}名 距离={dist:.4f} 下标={idx}  {oneline(texts[idx], 36)}")
    print(f"\nIndexFlatIP（内积=余弦相似度，越大越像）：")
    for rank, (score, idx) in enumerate(zip(d_ip[0], i_ip[0]), start=1):
        print(f"    第{rank}名 分数={score:.4f} 下标={idx}  {oneline(texts[idx], 36)}")

    # 数学上，对归一化向量有：L2距离² = 2 - 2×余弦相似度。
    # 所以两种索引的**排序结果必然完全一致**，只是分数的表达方式不同。
    print(f"\n两种索引的命中顺序一致吗：{list(i_l2[0]) == list(i_ip[0])}")
    print("对归一化过的向量，有恒等式 L2距离² = 2 - 2×余弦相似度，")
    print("所以两者排序必然相同，选哪个只影响你读到的分数怎么解释。")


# ===========================================================================
# 第 2 步：LangChain 封装的 FAISS —— 建库、检索、存盘、重新加载
# ===========================================================================
def langchain_faiss_demo(chunks: list[Document], emb: HuggingFaceEmbeddings) -> None:
    banner("第 2 步：LangChain 封装的 FAISS")

    # from_documents 是「类方法」（classmethod）：不需要先造对象，直接用类名调用，
    # 它内部会完成「编码所有文档 -> 建索引 -> 建立下标到 Document 的映射」一整套动作。
    store = FAISS.from_documents(chunks, emb)
    print(f"建库完成，索引里 {store.index.ntotal} 条向量")

    query = "文档切成多大合适"
    # similarity_search_with_score 返回 [(Document, 分数), ...]。
    # 对 FAISS 来说这个分数默认是 **L2 距离，越小越相关**——
    # 很多人误以为是相似度（越大越好），把排序写反了，这是高频 bug。
    hits = store.similarity_search_with_score(query, k=3)
    print(f"\n查询：{query}")
    for rank, (doc, score) in enumerate(hits, start=1):
        section = doc.metadata.get("h3") or doc.metadata.get("h2")
        print(f"    第{rank}名 L2距离={score:.4f} 章节={section}")
        print(f"          {oneline(doc.page_content)}")

    # --- 存盘（先演示官方写法，再处理它在本机踩到的坑）------------------------
    # save_local 是官方推荐写法，会写出两个文件：
    #   index.faiss —— 纯向量数据，由 FAISS 的 C++ 代码直接写文件
    #   index.pkl   —— 下标到 Document 的映射，用 Python 的 pickle 格式存
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        store.save_local(str(FAISS_DIR))
        print(f"\nsave_local 成功，写出：{sorted(p.name for p in FAISS_DIR.iterdir())}")
        saved_by = "save_local"
    except RuntimeError as e:
        # 【本机实测的坑】这个仓库的路径里有中文（"求职"、"agent开发"）。
        # FAISS 的文件读写是 C++ 实现的，在 Windows 上用的是系统 ANSI 代码页来解释路径，
        # 处理不了非 ASCII 字符，于是报「could not open ... for writing」。
        # 注意目录其实已经被 Python 建好了，失败的只是 C++ 那一侧的 fopen。
        print(f"\nsave_local 失败了：{type(e).__name__}")
        print(f"    {str(e).splitlines()[0][:110]}...")
        print("    原因：路径里有中文，FAISS 的 C++ 文件接口在 Windows 上打不开非 ASCII 路径。")
        print("    绕开办法：让 FAISS 只负责把索引变成字节，写文件交给 Python 自己做——")
        print("    Python 的文件接口是支持 Unicode 路径的。")

        # serialize_to_bytes() 把整个索引（向量 + Document 映射）打包成 bytes，
        # 全程在内存里完成，不碰文件系统，自然就绕开了上面那个编码问题。
        blob = store.serialize_to_bytes()
        # write_bytes 是 pathlib 提供的方法，用 Python 自己的 I/O 写二进制文件。
        (FAISS_DIR / "index.bytes").write_bytes(blob)
        print(f"    已改用 serialize_to_bytes() 写出 index.bytes"
              f"（{len(blob) / 1024:.1f} KB）")
        saved_by = "serialize_to_bytes"

    # --- 重新加载 -----------------------------------------------------------
    # allow_dangerous_deserialization=True 是必须显式传的，不传会直接抛异常。
    # 原因：这里用的 pickle 格式在反序列化时会执行文件内容描述的构造过程，
    # 一个被人做过手脚的文件可以借此在你机器上执行任意代码。
    # LangChain 强制你手动确认「这个文件是我自己生成的、我信任它」。
    # 加载别人给的索引文件时，这个参数是一个真实的安全风险，不要随手传 True。
    if saved_by == "save_local":
        reloaded = FAISS.load_local(
            str(FAISS_DIR), emb, allow_dangerous_deserialization=True
        )
    else:
        reloaded = FAISS.deserialize_from_bytes(
            (FAISS_DIR / "index.bytes").read_bytes(),
            emb,
            allow_dangerous_deserialization=True,
        )

    hits2 = reloaded.similarity_search(query, k=1)
    print(f"\n重新加载（{saved_by}）后再查一次，第 1 名是否相同："
          f"{hits2[0].page_content == hits[0][0].page_content}")
    print("关键点：存盘的是**向量**，不是嵌入模型本身。重新加载时必须传入**同一个模型**，")
    print("否则查询向量和库里的向量来自两个不同的空间，检索结果会完全是噪声（而且不报错）。")


# ===========================================================================
# 第 3 步：Chroma —— 自动持久化 + 按元数据过滤
# ===========================================================================
def chroma_demo(chunks: list[Document], emb: HuggingFaceEmbeddings) -> None:
    banner("第 3 步：Chroma")

    # 每次运行先删掉旧目录重建，避免重复插入（见下面的说明）。
    # shutil.rmtree 递归删除整个目录；ignore_errors=True 表示目录不存在时不报错。
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    store = Chroma(
        # collection（集合）相当于关系数据库里的「表」：同一个 Chroma 实例下可以有多个集合，
        # 互相隔离。检索只在指定的集合里进行。
        collection_name="rag_handbook",
        # embedding_function 就是嵌入模型对象。Chroma 会在 add_documents 时自动调它编码，
        # 在检索时自动编码查询——你不需要手动调用 embed_*。
        embedding_function=emb,
        # persist_directory 一给，Chroma 就把数据落到硬盘上（底层是 SQLite 数据库）。
        # 不给这个参数会怎样？数据只存在内存里，进程一退出就没了。
        persist_directory=str(CHROMA_DIR),
    )

    # add_documents 返回每条文档在库里的 id。
    # 【坑】不指定 ids 时 Chroma 会自动生成随机 id，所以**重复运行这个脚本会不断追加重复数据**。
    # 生产上的正确做法是自己指定稳定的 id（比如「文件名 + 起始偏移量」的哈希），
    # 这样重复入库会覆盖而不是追加。这里为了演示简单，直接每次删库重建。
    ids = store.add_documents(chunks)
    print(f"入库 {len(ids)} 条，自动生成的 id 示例：{ids[0]}")

    query = "文档切成多大合适"
    hits = store.similarity_search_with_score(query, k=3)
    print(f"\n查询：{query}")
    for rank, (doc, score) in enumerate(hits, start=1):
        section = doc.metadata.get("h3") or doc.metadata.get("h2")
        print(f"    第{rank}名 距离={score:.4f} 章节={section}")
        print(f"          {oneline(doc.page_content)}")

    # --- 按元数据过滤 --------------------------------------------------------
    # 这是 Chroma 相对 FAISS 最实用的能力：**先按元数据筛掉一批，再在剩下的里面做向量检索**。
    # 真实场景比如「只在 2024 年之后的文档里搜」「只搜我有权限看的部门」。
    # filter 的写法是 {字段名: 值}，也支持 {"字段": {"$in": [...]}} 这类操作符。
    print(f"\n同样的查询，但限定只在「二、检索阶段」这一章里找：")
    filtered = store.similarity_search(
        query,
        k=3,
        filter={"h2": "二、检索阶段"},
    )
    for rank, doc in enumerate(filtered, start=1):
        print(f"    第{rank}名 章节={doc.metadata.get('h3')}  {oneline(doc.page_content, 40)}")
    print("    注意结果全部落在指定章节内——这正是为什么切分时要认真维护 metadata。")

    # --- 落盘检查 -----------------------------------------------------------
    # rglob("*") 递归列出目录下所有文件；这里只是确认数据真的写到硬盘上了。
    written = [p for p in CHROMA_DIR.rglob("*") if p.is_file()]
    total_kb = sum(p.stat().st_size for p in written) / 1024
    print(f"\n落盘文件 {len(written)} 个，合计 {total_kb:.1f} KB，"
          f"不需要手动调用任何 save —— Chroma 写入时就已经持久化了。")


# ===========================================================================
# 第 4 步：统一接口 retriever
# ===========================================================================
def retriever_demo(chunks: list[Document], emb: HuggingFaceEmbeddings) -> None:
    banner("第 4 步：as_retriever —— 把向量库变成可替换的零件")

    store = FAISS.from_documents(chunks, emb)

    # **Retriever（检索器）**：LangChain 里的一个统一接口，只规定一件事——
    # 「给我一个字符串查询，还我一个 Document 列表」。
    # 它的价值是解耦：上层的 RAG 逻辑只认这个接口，
    # 底下换成 FAISS、Chroma、Milvus 还是 BM25 关键词检索，上层代码一行都不用改。
    retriever = store.as_retriever(
        # search_type="similarity" 是默认值：纯按相似度取前 k 条。
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    # invoke 是 LangChain 所有组件的统一调用方法。
    docs = retriever.invoke("文档切成多大合适")
    print(f"similarity 检索到 {len(docs)} 条：")
    for d in docs:
        print(f"    · {oneline(d.page_content)}")

    # search_type="mmr" 换一种策略。
    # **MMR（Maximal Marginal Relevance，最大边际相关性）**：在「和查询相关」之外，
    # 额外要求「返回的几条之间尽量不重复」。
    # 解决的问题：纯相似度检索经常返回三条内容几乎一样的块（尤其是有 overlap 的时候），
    # 白白浪费了三个名额。MMR 会牺牲一点相关性来换取结果的多样性。
    #   fetch_k     —— 先按相似度捞多少条candidates（候选）出来
    #   lambda_mult —— 0~1 之间，越接近 1 越看重相关性，越接近 0 越看重多样性
    mmr_retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    )
    mmr_docs = mmr_retriever.invoke("文档切成多大合适")
    print(f"\nMMR 检索到 {len(mmr_docs)} 条：")
    for d in mmr_docs:
        print(f"    · {oneline(d.page_content)}")

    same = [a.page_content for a in docs] == [b.page_content for b in mmr_docs]
    print(f"\n两种策略结果完全相同吗：{same}")


def main() -> None:
    print("正在加载嵌入模型...")
    emb = build_embeddings()

    chunks = build_chunks()
    raw_faiss_demo(chunks, emb)
    langchain_faiss_demo(chunks, emb)
    chroma_demo(chunks, emb)
    retriever_demo(chunks, emb)

    banner("全部完成")


if __name__ == "__main__":
    main()
