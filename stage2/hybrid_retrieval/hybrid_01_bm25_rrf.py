"""
hybrid_01_bm25_rrf.py —— 手写混合检索：BM25 + 向量，两路结果用 RRF 融合

混合检索（hybrid search）一句话：
    向量检索比的是"意思像不像"，对"X200-Pro"和"X210-Pro"这种只差一位数字的字符串分不清；
    BM25 按字面逐词比对，专治这种精确匹配，但用户换个说法它就找不到。
    两路同时跑，再合成一个排序，各补各的短板。

融合为什么不能直接把两路分数相加：
    BM25 分数没有上界（可能是 12.7），余弦相似度在 -1 到 1 之间（可能是 0.83），量纲不同。
    RRF（Reciprocal Rank Fusion，倒数排名融合）只用名次不用分数：
        融合分(块) = Σ 1 / (c + 该块在这一路里的名次)，c 通常取 60
    出处：Cormack, Clarke, Buettcher. Reciprocal Rank Fusion outperforms Condorcet
         and individual Rank Learning Methods. SIGIR 2009.
    位置：Modular RAG（arXiv:2407.21059）把混合检索归在 Retrieval 模块，RRF 属于 Orchestration 的 fusion。

本文件做的事：
    1. 从 common/corpus.py 拿到同一批块（BM25 和向量库必须索引完全相同的块，否则没法融合）
    2. 用 jieba 给每块分词，建 BM25 索引
    3. 对 7 道题分别跑三路：纯向量 / 纯 BM25 / RRF 混合
    4. 打印每路的前 3 块、标准答案的名次、R@3，并和跑之前写下的预测对照
    5. 最后用 LangChain 官方的 EnsembleRetriever 跑一遍，验证手写 RRF 的排序和它一致

运行方式（在 stage2 目录下）：
    uv run python -m hybrid_retrieval.hybrid_01_bm25_rrf

函数索引
    tokenize()          —— 用 jieba 把中文切成词，BM25 的输入
    build_bm25()        —— 建 BM25 索引，并打印一条分词结果供肉眼检查
    search_bm25()       —— BM25 单路检索，返回 [(块, 分数), ...]
    rrf_fuse()          —— 【本课核心】手写 RRF，把多路排序合成一路
    search_hybrid()     —— 向量一路 + BM25 一路 → rrf_fuse
    verify_with_ensemble() —— 用官方 EnsembleRetriever 对照，确认手写版没写错
    show_top()          —— 打印某一路的前 K 条
    main()              —— 串起全部流程并输出汇总表

阅读地图
    必读：rrf_fuse()      —— 融合规则，本课全部的"新知识"都在这 20 行里
          tokenize()      —— 中文 BM25 的成败在这一步
          search_bm25()   —— BM25 打分和排序的真实过程
    扫读：build_bm25() / search_hybrid() —— 数据组装
          verify_with_ensemble()        —— 手写版与官方封装的对照
    跳过：show_top() / main() 里的打印格式 —— 终端输出
"""

# [了解] jieba —— 中文分词库。中文词之间没有空格，BM25 又是按"词"计数的，
#        所以必须先把"青鸾挂了"切成 ["青鸾", "挂", "了"] 再喂给它。
import jieba

# [核心] BM25Okapi —— rank_bm25 库里 BM25 算法的标准实现。
#        它吃的是"已经分好词的文档列表"（list[list[str]]），不是原始字符串。
#        LangChain 的 BM25Retriever 内部调用的也是它，这里直接用，
#        一是能拿到每块的真实分数，二是不依赖已于 2026-05 停止维护并归档的 langchain-community 包。
from rank_bm25 import BM25Okapi

# [脚手架] Document 只在类型标注里用到；Chroma 是向量库类型。
from langchain_core.documents import Document
from langchain_chroma import Chroma

# [核心] 数据准备全部来自 common/corpus.py，和 HyDE、Multi-Query 那两课用的是同一份语料、同一套切块参数。
#        只有这样，三课的名次表才能横向比较。
from common.corpus import (
    K,
    build_vectorstore,
    gold_rank,
    load_pdf_text,
    recall_at_k,
    search_by_question,
    split_into_chunks,
    split_into_sections,
)

# [核心] RRF 的平滑常数。论文和绝大多数实现都取 60。
#        它的作用：分母从 61 起步，第 1 名(1/61≈0.0164)和第 2 名(1/62≈0.0161)几乎一样高，
#        所以"某一路的冠军"不能单独霸榜，必须两路都靠前才能赢。c 取 0 会让冠军权重暴涨。
RRF_C = 60

# [核心] 本课自己的测试集。Q1~Q4 沿用前两课（口语提问，BM25 吃亏的情况），
#        Q5~Q7 是新加的，问题里带"稀有字面词"（玄武 / RRF / k1），BM25 占优的情况。
#        predict 是跑之前写下的预测，跑完对照，防止事后找理由。
QUESTIONS = [
    {"id": "Q1", "kind": "口语", "gold": "§05", "predict": "向量赢",
     "q": "我搜商品型号，比如 X200-Pro，结果老给我返回 X210-Pro 这种差一点点的，怎么办？"},
    {"id": "Q2", "kind": "口语", "gold": "§07", "predict": "向量赢",
     "q": "第一轮捞出来几十条，顺序不太靠谱，怎么把最相关的几条挑到前面？"},
    {"id": "Q3", "kind": "口语", "gold": "§09", "predict": "向量赢",
     "q": "去上海出差住酒店，一晚最多能报多少？"},
    {"id": "Q4", "kind": "专名", "gold": "§12", "predict": "BM25 赢",
     "q": "青鸾挂了应该找谁？"},
    {"id": "Q5", "kind": "专名", "gold": "§12", "predict": "BM25 赢",
     "q": "玄武的写权限要谁批？"},
    {"id": "Q6", "kind": "专名", "gold": "§06", "predict": "混合赢",
     "q": "RRF 里那个常数 60 有什么用？"},
    {"id": "Q7", "kind": "专名", "gold": "§05", "predict": "BM25 赢",
     "q": "k1 和 b 这两个参数分别管什么？"},
    # [核心] Q8~Q13 是 2026-09-24 补语料（§15~§24）之后新加的。
    #        加它们的原因：原来 7 道题里 6 道三路并列第 1，实验区分不出方法差异。
    #        这 6 道题按"谁该赢"分成三组，每组针对一种检索方式的固有弱点：
    #          Q8~Q10 稀有标识符 —— 向量把"差一位数字"的串压成几乎相同，必然分不清
    #          Q11~Q12 纯语义跳跃 —— 问题用词和文档用词零重合，BM25 必然对不上
    #          Q13    两者各半   —— 字面信号 + 语义信号各贡献一路，只有融合能同时用上
    {"id": "Q8", "kind": "标识", "gold": "§15", "predict": "BM25 赢",
     "q": "X200-Pro 的保修期是多久？"},
    {"id": "Q9", "kind": "标识", "gold": "§17", "predict": "BM25 赢",
     "q": "E5022 是什么原因造成的？"},
    {"id": "Q10", "kind": "标识", "gold": "§18", "predict": "BM25 赢",
     "q": "INC-20260917-0342 这种编号怎么解读？"},
    {"id": "Q11", "kind": "语义", "gold": "§20", "predict": "向量赢",
     "q": "出差打车能报多少？"},
    {"id": "Q12", "kind": "语义", "gold": "§21", "predict": "向量赢",
     "q": "临时来的外面的人，账号多久过期？"},
    {"id": "Q13", "kind": "混合", "gold": "§22", "predict": "混合赢",
     "q": "值夜班一次补多少钱？"},
]


def tokenize(text: str) -> list[str]:
    """[核心] 中文分词：把一段连着写的中文切成词列表，这是 BM25 的唯一输入形式。"""
    # 1. [核心] jieba.lcut 返回一个列表（l = list）。jieba.cut 返回的是生成器，还要再转一次。
    words = jieba.lcut(text)
    # 2. [核心] 去掉空白词和长度为 0 的碎片。分词会把空格、换行也切出来，
    #    它们在每一块里都出现，对区分文档毫无贡献，留着只会拖慢计算。
    # [自测 2 的开关] 把上面两行注释掉、改成 return text.split()，就能复现"中文不分词"的静默失效。
    return [w.strip() for w in words if w.strip()]

def build_bm25(chunks: list[Document]) -> BM25Okapi:
    """[了解] 给全部块建 BM25 索引。返回的对象可以对任意查询算出"每一块的得分"。"""
    # 1. [核心] 先把每一块分好词。corpus 是 list[list[str]]：外层是块，内层是这块的词。
    corpus = [tokenize(doc.page_content) for doc in chunks]

    # 2. [脚手架] 打印第一块的前 20 个词，肉眼确认分词没出问题。
    #    如果这里印出来的是一整句话（只有一个元素），说明分词没生效，BM25 必然失效且不报错。
    print(f"分词自检｜第 1 块切出 {len(corpus[0])} 个词，前 20 个：{'/'.join(corpus[0][:20])}")

    # 3. [核心] BM25Okapi 在构造时就完成了全部统计：每个词在多少块里出现过（决定稀有度）、
    #    每块多长（决定长度惩罚）。所以建索引只做一次，之后每次查询都很快。
    #    参数 k1=1.5、b=0.75 是库的默认值，和语料里 §05 说的经验范围一致，这里不改。
    return BM25Okapi(corpus)


def search_bm25(bm25: BM25Okapi, chunks: list[Document], question: str,
                k: int) -> list[tuple[Document, float]]:
    """[核心] BM25 单路检索：返回前 k 条 [(块, 分数), ...]，分数越大越相关。"""
    # 1. [核心] 查询也要用同一个分词函数切。两边分词方式不一致，词就对不上，分数全是 0。
    query_words = tokenize(question)

    # 2. [核心] get_scores 一次性算出"查询 vs 每一块"的得分，返回的数组长度等于块数。
    #    注意它返回的是全部块的分数，不做排序也不截断。
    scores = bm25.get_scores(query_words)

    # 3. [了解] 把块和分数配对，按分数从大到小排。reverse=True 就是降序。
    #    BM25 分数没有上界，也没有下界之外的意义：0 分表示查询词一个都没命中。
    pairs = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return pairs[:k]


def rrf_fuse(rankings: list[list[Document]], c: int = RRF_C) -> list[tuple[Document, float]]:
    """[核心] 手写 RRF：把多路排序合成一路。rankings 是"每一路的块列表"，已按各自相关度排好。"""
    # 1. [核心] 融合分累加器。键用块的正文字符串，因为同一块会被不同路各返回一次，
    #    必须认出"它们是同一块"才能把两路的分加到一起。
    scores: dict[str, float] = {}
    # 2. [脚手架] 正文 → 块对象的映射，最后要把字符串还原成 Document。
    lookup: dict[str, Document] = {}

    # 3. [核心] 逐路遍历。enumerate(..., start=1) 让名次从 1 开始，而不是 0——
    #    名次 0 会让分母变成 c，等于白送一份分。
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc.page_content
            # 4. [核心] RRF 的全部公式就是这一行：名次越靠前，1/(c+名次) 越大。
            #    用 .get(key, 0.0) 是因为这块可能是第一次出现，还没有累计分。
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
            lookup[key] = doc

    # 5. [核心] 按融合分降序排。只在一路里出现的块也有分，只是通常拼不过两路都靠前的块。
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(lookup[key], score) for key, score in ordered]


def search_hybrid(store: Chroma, bm25: BM25Okapi, chunks: list[Document],
                  question: str, k: int) -> list[tuple[Document, float]]:
    """[核心] 混合检索：两路各自检索，再交给 rrf_fuse 合并。"""
    # 1. [核心] 向量一路。search_by_question 返回 [(块, 距离), ...]，距离越小越像，
    #    但 RRF 只看名次，所以这里把分数丢掉，只保留顺序。
    dense_docs = [doc for doc, _ in search_by_question(store, question, k=k)]

    # 2. [核心] BM25 一路，同样只保留顺序。
    sparse_docs = [doc for doc, _ in search_bm25(bm25, chunks, question, k=k)]

    # 3. [核心] 两路交给 RRF。传进去的顺序不影响结果——RRF 对各路是对称的（权重相同时）。
    return rrf_fuse([dense_docs, sparse_docs])


def verify_with_ensemble(store: Chroma, chunks: list[Document], question: str,
                         k: int) -> list[str]:
    """[了解] 用 LangChain 官方的 EnsembleRetriever 跑同一道题，返回它的前 k 条所属小节。"""
    # 1. [了解] EnsembleRetriever 在 langchain_classic 包里，做的就是"加权 RRF"，
    #    权重相同时和本文件的 rrf_fuse 是同一个公式，常数 c 默认也是 60。
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever

    # 2. [核心] BM25Retriever 默认用 text.split() 分词——对中文等于不分词。
    #    必须显式传 preprocess_func=tokenize，否则整句变成一个"词"，检索静默失效。
    sparse = BM25Retriever.from_documents(chunks, preprocess_func=tokenize)
    sparse.k = k

    # 3. [了解] 向量库转成检索器接口，两边都变成"吃问题、吐文档列表"的同一种对象，才能交给 Ensemble。
    dense = store.as_retriever(search_kwargs={"k": k})

    # 4. [了解] weights 不传就是各占一半。invoke 返回融合后的文档列表。
    ensemble = EnsembleRetriever(retrievers=[dense, sparse])
    return [doc.metadata["section"] for doc in ensemble.invoke(question)[:k]]


def show_top(label: str, results: list[tuple[Document, float]], gold: str) -> None:
    """[脚手架] 打印某一路的前 K 条：名次、分数、所属小节、正文开头。"""
    print(f"  【{label}】")
    for rank, (doc, score) in enumerate(results[:K], start=1):
        mark = "✔" if doc.metadata["section"] == gold else " "
        preview = doc.page_content.replace("\n", "")[:36]
        print(f"    {mark} #{rank}  {score:8.4f}  {doc.metadata['section']}  {preview}…")


def main() -> None:
    # 1. [了解] 数据准备：抽 PDF → 拆小节 → 建向量库；再单独切一份块给 BM25。
    #    split_into_chunks 和 build_vectorstore 内部用的是同一套切分参数，
    #    所以这两批块内容完全一致，RRF 才能按正文把它们认成同一块。
    print("=== 抽取检查 ===")
    sections = split_into_sections(load_pdf_text())
    store, _embeddings, n_chunks = build_vectorstore(sections)
    chunks = split_into_chunks(sections)
    print(f"小节数：{len(sections)}　块数：{n_chunks}")

    # 2. [核心] 建 BM25 索引。只建一次，7 道题共用。
    bm25 = build_bm25(chunks)
    print()

    summary = []
    for item in QUESTIONS:
        q, gold = item["q"], item["gold"]
        print(f"=== {item['id']}（{item['kind']}）{q}")
        print(f"  标准答案：{gold}　预测：{item['predict']}")

        # 3. [核心] 三路都取"全部块"的完整排序（k=n_chunks），
        #    这样标准答案就算掉到第 15 名也数得出来；打印时只显示前 K 条。
        dense = search_by_question(store, q, k=n_chunks)
        sparse = search_bm25(bm25, chunks, q, k=n_chunks)
        hybrid = search_hybrid(store, bm25, chunks, q, k=n_chunks)

        show_top("纯向量", dense, gold)
        show_top("纯 BM25", sparse, gold)
        show_top("RRF 混合", hybrid, gold)

        summary.append((
            item,
            gold_rank(dense, gold), gold_rank(sparse, gold), gold_rank(hybrid, gold),
            recall_at_k(dense, gold, K), recall_at_k(sparse, gold, K), recall_at_k(hybrid, gold, K),
        ))
        print()

    # 4. [脚手架] 汇总表：名次越小越好，None 表示全部块里都没找到（只有 BM25 会出现，查询词一个都没命中）。
    print("=== 汇总：标准答案小节的名次（越小越好）===")
    print(f"{'题号':<5}{'类型':<5}{'向量':>5}{'BM25':>6}{'混合':>5}"
          f"{'R@3向量':>9}{'R@3BM25':>10}{'R@3混合':>9}   预测")
    for item, r_d, r_s, r_h, rc_d, rc_s, rc_h in summary:
        print(f"{item['id']:<6}{item['kind']:<6}{r_d!s:>5}{r_s!s:>6}{r_h!s:>5}"
              f"{rc_d:>9}{rc_s:>10}{rc_h:>9}   {item['predict']}")

    # 5. [了解] 对照验证：官方 EnsembleRetriever 的前 3 条，和手写 RRF 的前 3 条是否一致。
    #    一致说明手写版的公式和名次起点都没写错。
    print("\n=== 对照：手写 RRF vs 官方 EnsembleRetriever（前 3 条所属小节）===")
    for item in QUESTIONS:
        mine = [doc.metadata["section"]
                for doc, _ in search_hybrid(store, bm25, chunks, item["q"], k=K)[:K]]
        theirs = verify_with_ensemble(store, chunks, item["q"], k=K)
        flag = "一致" if mine == theirs else "不一致"
        print(f"  {item['id']}  手写 {mine}   官方 {theirs}   {flag}")


if __name__ == "__main__":
    main()
