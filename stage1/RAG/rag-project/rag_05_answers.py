"""
RAG Part 2 自测题 —— 答案版

三道题的完整解法，每处答案下方附实测输出。
配套的挖空版是 rag_05_selftest.py，两份文件除 TODO 处之外逐字一致。

运行方式：
    uv run rag_05_answers.py            # 跑全部三题
    uv run rag_05_answers.py 1a 2 3     # 只跑指定的题（题 1b 需要额外下载模型）

阅读地图
    必读：test_1_model_mismatch()   —— 索引和嵌入模型是绑死的，换了就废
          stable_id()              —— 什么样的 id 才叫「稳定」
          test_2_stable_ids()      —— upsert 覆盖 vs 追加重复数据
    扫读：pairwise_mean_similarity() —— 多样性怎么量化成一个数
          test_3_mmr_diversity()   —— lambda_mult 的调参手感
    跳过：banner() / oneline() / main() —— 终端输出与命令行参数解析
"""

from itertools import combinations   # 标准库：从列表里取出所有「两两组合」。文档：https://docs.python.org/3/library/itertools.html
from pathlib import Path
import sys

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
MD_PATH = BASE_DIR / "data" / "sample_zh.md"
CHROMA_DIR = BASE_DIR / "chroma_db_selftest"     # 和 rag_04 的库分开，互不干扰

MODEL_NAME = "BAAI/bge-small-zh-v1.5"                        # 512 维
OTHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"       # 384 维，题 1b 的对照
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
QUERY = "文档切成多大合适"


# ===========================================================================
# 脚手架区：这一段整段都可以跳过
# ===========================================================================
def banner(title: str) -> None:
    """[脚手架] 打印分节标题。"""
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def oneline(text: str, n: int = 40) -> str:
    """[脚手架] 把块内容压成单行并截断，方便终端预览。"""
    return text.replace("\n", " ")[:n] + "..."


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """[了解] 余弦相似度：点积除以两个模长。向量已归一化时分母恒为 1。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_chunks() -> list[Document]:
    """[脚手架] 复用第 4 课的两级切分，产出 8 个可入库的块。"""
    raw = MD_PATH.read_text(encoding="utf-8")
    base = Document(page_content=raw, metadata={"source": MD_PATH.name})

    header_chunks = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    ).split_text(base.page_content)
    for c in header_chunks:
        c.metadata = base.metadata | c.metadata

    chunks = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=200,
        chunk_overlap=40,
        add_start_index=True,
    ).split_documents(header_chunks)
    return [c for c in chunks if len(c.page_content) >= 20]


def build_embeddings(with_prefix: bool = True, model: str = MODEL_NAME) -> HuggingFaceEmbeddings:
    """[核心] 造嵌入模型。with_prefix 控制查询端加不加 BGE 的指令前缀。

    文档端（encode_kwargs）永远不加前缀，这是模型卡的硬要求；
    差异只体现在查询端（query_encode_kwargs）。
    """
    query_kwargs: dict = {"normalize_embeddings": True}
    if with_prefix:
        query_kwargs["prompt"] = BGE_QUERY_INSTRUCTION
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs=query_kwargs,
    )


# ===========================================================================
# 题 1：换了模型（或换了配置）的索引不能混用
# ===========================================================================
def test_1_model_mismatch(chunks: list[Document], run_1b: bool = True) -> None:
    """[核心] 验证「索引和嵌入模型是绑死的」这件事的两种失败形态。"""
    banner("题 1 · 换了模型的索引不能混用")

    # [核心] 用「带查询前缀」的模型建索引并序列化。
    #        注意：库里存的文档向量两种配置下完全相同（文档端本来就不加前缀），
    #        变的只有**查询**的编码方式。这正是本题要隔离出来的变量。
    emb_ref = build_embeddings(with_prefix=True)
    store = FAISS.from_documents(chunks, emb_ref)
    blob = store.serialize_to_bytes()
    print(f"用 {MODEL_NAME}（带查询前缀）建好索引，序列化 {len(blob) / 1024:.1f} KB\n")

    baseline = store.similarity_search_with_score(QUERY, k=3)
    print(f"【基准】查询：{QUERY}")
    for rank, (doc, score) in enumerate(baseline, start=1):
        print(f"    第{rank}名 距离={score:.4f}  {oneline(doc.page_content)}")

    # ---- 1A：同一个模型，但查询端配置不同 --------------------------------
    # ---- 题 1-① 的答案 ---------------------------------------------------
    # [核心] 去掉查询前缀。维度完全一致，所以**不会报任何错**——
    #        这才是危险的地方：系统照常运行，只是检索质量悄悄变了。
    emb_no_prefix = build_embeddings(with_prefix=False)
    # ---------------------------------------------------------------------

    reloaded_a = FAISS.deserialize_from_bytes(
        blob,
        emb_no_prefix,
        # [了解] 反序列化用的是 pickle，会执行文件里描述的构造过程，
        #        所以 LangChain 强制你显式声明「我信任这份数据」。
        allow_dangerous_deserialization=True,
    )
    hits_a = reloaded_a.similarity_search_with_score(QUERY, k=3)

    print(f"\n【1A · 同模型，查询端去掉前缀】不报错，结果如下：")
    for rank, (doc, score) in enumerate(hits_a, start=1):
        print(f"    第{rank}名 距离={score:.4f}  {oneline(doc.page_content)}")

    same_top1 = baseline[0][0].page_content == hits_a[0][0].page_content
    drift = abs(baseline[0][1] - hits_a[0][1])
    print(f"\n    top-1 是否相同：{same_top1}")
    print(f"    top-1 距离漂移：{drift:.4f}")
    print("    结论：维度一致就不会报错。检索还能用，但打分体系已经变了——")
    print("          语料一多、区分度一小，排序就会开始出错，而你不会收到任何信号。")

    # ---- 1B：换成维度不同的模型 -----------------------------------------
    if not run_1b:
        print(f"\n【1B】已跳过（需要先下载 {OTHER_MODEL}）")
        return

    # ---- 题 1-② 的答案 ---------------------------------------------------
    # [核心] 预测：库里是 512 维向量，新模型产出 384 维查询向量，
    #        FAISS 在比对时会发现维度对不上 —— 这次应该**会**报错。
    #        用 try/except 捕获，把异常类型和首行消息打出来看清楚。
    emb_384 = build_embeddings(with_prefix=False, model=OTHER_MODEL)
    try:
        reloaded_b = FAISS.deserialize_from_bytes(
            blob, emb_384, allow_dangerous_deserialization=True
        )
        hits_b = reloaded_b.similarity_search(QUERY, k=1)
        print(f"\n【1B · 换成 384 维模型】居然没报错？第 1 名：{oneline(hits_b[0].page_content)}")
    except Exception as e:                      # noqa: BLE001 —— 这里就是要看到底抛的是什么
        print(f"\n【1B · 换成 384 维模型】抛异常了：")
        print(f"    异常类型：{type(e).__name__}")
        print(f"    首行消息：{str(e).splitlines()[0][:150]}")
    # ---------------------------------------------------------------------

    print("\n    两种失败对照：")
    print("      · 维度不同 -> 立刻报错，反而是好事，你马上就知道搞错了")
    print("      · 维度相同但配置/模型不同 -> 静默劣化，最难排查")
    print("    所以索引必须和「模型名 + 编码配置」一起版本化管理，")
    print("    换模型就得整库重建，没有增量迁移这条路。")


# ===========================================================================
# 题 2：给 Chroma 加稳定 id，让重复入库变成覆盖
# ===========================================================================
def stable_id(chunk: Document) -> str:
    """[核心] 为一个块生成稳定不变的 id。"""
    # ---- 题 2-① 的答案 ---------------------------------------------------
    # [核心] 「稳定」的含义是：同一块文本无论跑多少次，都得到完全一样的 id。
    #        所以不能用随机数（uuid）、不能用时间戳、不能用列表下标（切分参数一改就错位）。
    #
    # [核心] 这里有个真实踩到的坑：只写 f"{source}#{start_index}" 是**不够**的，
    #        会抛 chromadb.errors.DuplicateIDError: found duplicates of sample_zh.md#0。
    #        原因回到第 1 课那条结论——start_index 是相对于「传给切分器的那个 Document」
    #        的偏移，而我们是先按标题切、再对每个标题块递归切，
    #        所以**每个标题块内的第一个小块 start_index 都是 0**，8 块里有好几个 0。
    #        补上章节路径作为区分，才真正唯一。
    meta = chunk.metadata
    # [了解] .get(key, "") 而不是 meta[key]：不是每个块都有 h3（比如只到二级标题的那些），
    #        直接下标访问会抛 KeyError。
    section = "/".join(meta.get(k, "") for k in ("h1", "h2", "h3"))
    return f"{meta['source']}#{section}#{meta['start_index']}"
    # ---------------------------------------------------------------------


def _run_three_rounds(chunks: list[Document], emb: HuggingFaceEmbeddings,
                      use_stable_ids: bool) -> None:
    """[脚手架] 连续入库三轮，每轮打印库里的总条数。"""
    label = "传稳定 id" if use_stable_ids else "不传 id（默认随机 uuid）"
    print(f"\n【{label}】")

    # [核心] 清库不能用 shutil.rmtree —— 这是本次实测踩到的坑。
    #        Windows 上前一组实验创建的 Chroma 对象仍持有 chroma.sqlite3 的文件句柄，
    #        被占用的文件删不掉；而 ignore_errors=True 会把这个失败**静默吞掉**。
    #        表现就是第二组的计数从 24 起跳变成 32/32/32，看上去像 upsert 没生效，
    #        实际是上一组的残留数据从来没被删掉。
    #        用 Chroma 自己的 reset_collection() 才可靠：它走数据库层删除，不碰文件系统。
    Chroma(
        collection_name="selftest",
        embedding_function=emb,
        persist_directory=str(CHROMA_DIR),
    ).reset_collection()

    for round_no in (1, 2, 3):
        store = Chroma(
            collection_name="selftest",
            embedding_function=emb,
            persist_directory=str(CHROMA_DIR),
        )
        if use_stable_ids:
            # ---- 题 2-② 的答案 -------------------------------------------
            # [核心] 把稳定 id 列表传给 add_documents。
            #        langchain_chroma 的 add_texts 内部走的是 collection.upsert()，
            #        所以 id 相同的记录是**覆盖**而不是追加。
            store.add_documents(chunks, ids=[stable_id(c) for c in chunks])
            # -------------------------------------------------------------
        else:
            store.add_documents(chunks)

        # [了解] Chroma.get() 返回一个字典，"ids" 键下是库里全部记录的 id。
        #        不传参数就是全量取，小库可以这么数，大库要用 count 类接口。
        total = len(store.get()["ids"])
        print(f"    第 {round_no} 轮入库 {len(chunks)} 条后，库里共 {total} 条")


def test_2_stable_ids(chunks: list[Document], emb: HuggingFaceEmbeddings) -> None:
    """[核心] 对照：不传 id 会不断追加重复数据，传稳定 id 则覆盖。"""
    banner("题 2 · 给 Chroma 加稳定 id")

    print(f"每轮都入库同样的 {len(chunks)} 块，连做三轮，看库里最后有多少条。")

    # [核心] 入库前先自检 id 是否唯一。Chroma 会对重复 id 抛 DuplicateIDError，
    #        但那个报错发生在入库那一刻，离「id 是怎么拼出来的」已经隔了好几层调用栈。
    #        在这里断言一次，问题就暴露在你写 id 的地方。
    ids = [stable_id(c) for c in chunks]
    duplicates = len(ids) - len(set(ids))
    print(f"id 唯一性自检：{len(ids)} 个 id，重复 {duplicates} 个")
    assert duplicates == 0, f"id 不唯一，重复 {duplicates} 个：{ids}"

    _run_three_rounds(chunks, emb, use_stable_ids=False)
    _run_three_rounds(chunks, emb, use_stable_ids=True)

    print(f"\n    id 示例：{stable_id(chunks[0])}")
    print("    结论：不传 id 时 Chroma 自动生成随机 uuid，同一块文本每轮都是「新记录」，")
    print("          库里堆满重复内容，检索时前几名可能全是同一段的副本。")
    print("          传稳定 id 之后重复入库变成覆盖，这也是增量更新知识库的正确姿势——")
    print("          文档改了就重新入一次，id 不变，自动替换旧版本。")


# ===========================================================================
# 题 3：量化 MMR 到底带来了什么
# ===========================================================================
def pairwise_mean_similarity(docs: list[Document], emb: HuggingFaceEmbeddings) -> float:
    """[核心] 返回一组文档「两两之间余弦相似度」的平均值，越低说明结果越多样。"""
    # ---- 题 3-① 的答案 ---------------------------------------------------
    # [核心] 三步：① 把这几条文本一次性编码成向量；
    #              ② combinations(range(n), 2) 列出所有不重复的两两配对，
    #                 3 条就是 (0,1)(0,2)(1,2) 共 3 对；
    #              ③ 求这些配对的余弦相似度平均值。
    vecs = [np.array(v) for v in emb.embed_documents([d.page_content for d in docs])]
    pairs = list(combinations(range(len(vecs)), 2))
    return sum(cosine(vecs[i], vecs[j]) for i, j in pairs) / len(pairs)
    # ---------------------------------------------------------------------


def test_3_mmr_diversity(chunks: list[Document], emb: HuggingFaceEmbeddings) -> None:
    """[了解] 扫 lambda_mult，观察相关性与多样性的取舍。"""
    banner("题 3 · 量化 MMR 的多样性")

    store = FAISS.from_documents(chunks, emb)
    k = 3

    base_docs = store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    ).invoke(QUERY)
    base_div = pairwise_mean_similarity(base_docs, emb)

    print(f"查询：{QUERY}    k={k}    候选池 fetch_k={len(chunks)}\n")
    print(f"{'策略':>22}  {'两两相似度均值':>14}   命中章节")
    print("-" * 74)
    print(f"{'similarity（纯相关性）':>22}  {base_div:>14.4f}   "
          f"{[d.metadata.get('h3') or d.metadata.get('h2') for d in base_docs]}")

    # ---- 题 3-② 的答案 ---------------------------------------------------
    # [了解] lambda_mult 越接近 1 越看重相关性，越接近 0 越看重多样性。
    #        用 range(1, 10) 生成 1~9 再除以 10，避免浮点数累加的精度误差
    #        （0.1+0.1+0.1 在浮点里不等于 0.3）。
    for lam in [i / 10 for i in range(1, 10)]:
        # ---------------------------------------------------------------------
        mmr_docs = store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": len(chunks), "lambda_mult": lam},
        ).invoke(QUERY)
        div = pairwise_mean_similarity(mmr_docs, emb)
        sections = [d.metadata.get("h3") or d.metadata.get("h2") for d in mmr_docs]
        print(f"{'mmr lambda=' + f'{lam:.1f}':>22}  {div:>14.4f}   {sections}")

    print("\n    怎么读这张表：数值越低，返回的三条彼此越不像，覆盖面越广。")
    print("    lambda 调低会把内容相近的块挤掉，换上话题不同的块；")
    print("    代价是排在前面的不再一定是最相关的那条。")
    print("    实践建议：先用 similarity 跑通，只有当你发现召回结果高度雷同")
    print("    （典型症状是几条只差一个 overlap 窗口）时，才引入 MMR。")


# ===========================================================================
def main() -> None:
    """[脚手架] 解析命令行参数，按需运行指定的题目。"""
    # [脚手架] sys.argv[1:] 是命令行参数；不传就跑全部。
    wanted = {a.lower() for a in sys.argv[1:]} or {"1a", "1b", "2", "3"}

    print("正在加载嵌入模型...")
    emb = build_embeddings(with_prefix=True)
    chunks = build_chunks()
    print(f"语料准备完成：{len(chunks)} 块")

    if wanted & {"1a", "1b", "1"}:
        test_1_model_mismatch(chunks, run_1b=bool(wanted & {"1b", "1"}))
    if "2" in wanted:
        test_2_stable_ids(chunks, emb)
    if "3" in wanted:
        test_3_mmr_diversity(chunks, emb)

    banner("全部完成")


if __name__ == "__main__":
    main()
