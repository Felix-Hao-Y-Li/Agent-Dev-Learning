"""
rerank_01_cross_encoder.py —— 两阶段检索：混合检索召回 → cross-encoder 重排

重排（re-ranking）一句话：
    第一阶段用便宜的方法从全库捞出几十条候选，目标是"别漏"；
    第二阶段用贵但准的模型只对这几十条重新打分，目标是"排准"。
    贵的那个模型叫交叉编码器（cross-encoder），它把"问题+文档"拼成一段话一起读，
    而不是像检索模型那样把两边分别压成向量再比距离。

本文件要回答的两个问题：
    1. 重排能不能把混合检索排错的答案纠正过来？
    2. 如果正确答案根本没进候选，重排能不能救？（答案是不能，这是重排的硬边界）
    为此故意跑两次：召回 10 条（Q13 的答案进不来）和召回 15 条（进得来），对比同一道题。

运行方式（在 stage2 目录下）：
    uv run python -m reranking.rerank_01_cross_encoder

函数索引
    load_reranker()      —— 加载本地 cross-encoder 模型
    rerank()             —— 【本课核心】把候选逐条打分并重新排序
    rank_in()            —— 在一个候选列表里找出标准答案排第几
    recall_and_rerank()  —— 召回 → 重排，返回两个阶段的结果与耗时
    show_top()           —— 打印某一路的前 3 条
    main()               —— 13 道题跑两轮（召回 10 / 召回 15）并输出对比表

阅读地图
    必读：rerank()            —— 全部机制都在这 10 行：组 pair、打分、排序
          recall_and_rerank() —— 两阶段的衔接方式与耗时统计
    扫读：load_reranker()     —— 模型加载参数
    跳过：show_top() / main() 里的打印 —— 终端输出
"""

# [脚手架] 标准库：time 用来量重排耗时，它是这一课最重要的代价指标。
import time
import torch
# [核心] CrossEncoder —— sentence-transformers 对交叉编码器的封装。
#        它和我们做检索用的 SentenceTransformer 是两类模型：
#        后者把一段文字变成向量（可以离线算好），前者直接读"两段文字"输出一个相关度分数（必须实时算）。
from sentence_transformers import CrossEncoder

# [脚手架] Document 只用于类型标注。
from langchain_core.documents import Document

# [核心] 数据准备与评估沿用 common/corpus.py，召回沿用上一课写好的混合检索。
#        这样"召回质量"这个变量是固定的，本文件唯一的新变量就是"重排"这一步。
from common.corpus import (
    K,
    build_vectorstore,
    load_pdf_text,
    recall_at_k,
    split_into_chunks,
    split_into_sections,
)
from hybrid_retrieval.hybrid_01_bm25_rrf import QUESTIONS, build_bm25, search_hybrid

# [核心] 重排模型。0.3B 参数、约 1.1GB、中英双语、最大输入 512 个 token。
#        换成 bge-reranker-large 会怎样：精度略高、体积约 2.2GB、耗时约翻倍。
RERANKER_NAME = "BAAI/bge-reranker-base"

# [核心] 召回条数。两轮分别取 10 和 15，用来验证"重排的天花板是召回"这条边界：
#        Q13 的标准答案在混合排序里是第 12 名，取 10 进不来、取 15 才进得来。
RECALL_SIZES = [3, 44]


def load_reranker() -> CrossEncoder:
    """[了解] 加载本地 cross-encoder。第一次运行会从 HuggingFace 下载权重并缓存。"""
    # 1. [了解] max_length=512 是模型自身的上限。"问题 + 文档"拼起来超过这个长度会被截断，
    #    被截掉的部分完全不参与打分，而且不会有任何提示——块切太大时这是隐性的质量损失。
    return CrossEncoder(RERANKER_NAME, max_length=512)


def rerank(model: CrossEncoder, question: str,
           candidates: list[Document]) -> list[tuple[Document, float]]:
    """[核心] 重排：把每个候选和问题配成一对，逐对打分，再按分数重排。"""
    # 1. [核心] 组装输入。cross-encoder 吃的是"成对的文本"，不是单段文本。
    #    这是它和检索模型最根本的区别：问题和文档在同一次前向计算里相互可见。
    pairs = [(question, doc.page_content) for doc in candidates]

    # 2. [核心] predict 对每一对算一个分数。len(pairs) 有多少，模型就要跑多少次，
    #    所以候选数量直接决定这一步的耗时——这是两阶段设计的全部代价来源。
    #    分数含义：sentence-transformers 在 num_labels=1 时默认套了一层 Sigmoid，
    #    所以这里拿到的是 0~1 之间的数，越大越相关。直接用 transformers 调同一个模型
    #    拿到的是未经 Sigmoid 的 logit（可能是负数），两者排序一致但数值不同。
    scores = model.predict(pairs, activation_fn=torch.nn.Identity())

    # 3. [核心] 按分数降序。注意这个分数只在同一个问题内部可比：
    #    A 问题的 0.9 和 B 问题的 0.9 不代表同样的相关程度，所以不能拿它做全局阈值过滤。
    ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [(doc, float(score)) for doc, score in ranked]


def rank_in(results: list[tuple[Document, float]], gold: str) -> int | None:
    """[了解] 在一个结果列表里找出标准答案小节第一次出现的名次，没出现返回 None。"""
    # 1. [了解] 和 common/corpus.py 的 gold_rank 同义，单独写一份是因为这里传入的
    #    是"候选集内部"的排序，而不是全库排序，语义不同，分开更不容易读混。
    for rank, (doc, _) in enumerate(results, start=1):
        if doc.metadata["section"] == gold:
            return rank
    return None


def recall_and_rerank(store, bm25, chunks: list[Document], model: CrossEncoder,
                      question: str, n_recall: int):
    """[核心] 完整的两阶段：混合检索召回 n_recall 条 → cross-encoder 重排。"""
    # 1. [核心] 第一阶段。search_hybrid 内部是"向量一路 + BM25 一路 → RRF 融合"，
    #    取前 n_recall 条作为候选。这一步的目标是别漏，不追求顺序对。
    t0 = time.perf_counter()
    recalled = search_hybrid(store, bm25, chunks, question, k=n_recall)[:n_recall]
    t_recall = time.perf_counter() - t0

    # 2. [核心] 第二阶段。只有这 n_recall 条进入重排，全库其余的块连看都不看。
    t1 = time.perf_counter()
    reranked = rerank(model, question, [doc for doc, _ in recalled])
    t_rerank = time.perf_counter() - t1

    return recalled, reranked, t_recall, t_rerank


def show_top(label: str, results: list[tuple[Document, float]], gold: str) -> None:
    """[脚手架] 打印前 K 条：名次、分数、所属小节、正文开头。"""
    print(f"  【{label}】")
    for rank, (doc, score) in enumerate(results[:K], start=1):
        mark = "✔" if doc.metadata["section"] == gold else " "
        preview = doc.page_content.replace("\n", "")[:34]
        print(f"    {mark} #{rank}  {score:7.4f}  {doc.metadata['section']}  {preview}…")


def main() -> None:
    # 1. [了解] 数据准备与上一课完全一致：24 小节 → 44 块 → 向量库 + BM25 索引。
    print("=== 准备数据 ===")
    sections = split_into_sections(load_pdf_text())
    store, _embeddings, n_chunks = build_vectorstore(sections)
    chunks = split_into_chunks(sections)
    bm25 = build_bm25(chunks)
    print(f"小节数：{len(sections)}　块数：{n_chunks}\n")

    # 2. [了解] 加载重排模型。只加载一次，13 道题 × 2 轮共用。
    print("=== 加载重排模型 ===")
    t0 = time.perf_counter()
    model = load_reranker()
    print(f"{RERANKER_NAME} 加载完成，耗时 {time.perf_counter() - t0:.1f} 秒\n")

    for n_recall in RECALL_SIZES:
        print(f"{'=' * 72}")
        print(f"=== 召回 {n_recall} 条 → 重排 ===")
        print(f"{'=' * 72}")

        summary = []
        for item in QUESTIONS:
            q, gold = item["q"], item["gold"]
            recalled, reranked, t_rec, t_rr = recall_and_rerank(
                store, bm25, chunks, model, q, n_recall)

            # 3. [核心] 三个关键数字：
            #    before —— 标准答案在"召回结果"里排第几（重排之前）
            #    after  —— 标准答案在"重排结果"里排第几（重排之后）
            #    两者都为 None，说明答案压根没进候选，重排无能为力。
            before = rank_in(recalled, gold)
            after = rank_in(reranked, gold)
            summary.append((item, before, after,
                            recall_at_k(recalled, gold, K),
                            recall_at_k(reranked, gold, K),
                            t_rec, t_rr))

            # 4. [脚手架] 只对"重排改变了名次"的题打印明细，避免刷屏。
            if before != after:
                print(f"\n--- {item['id']}　{q}　标准答案 {gold}")
                show_top("重排前（混合召回）", recalled, gold)
                show_top("重排后（cross-encoder）", reranked, gold)

        # 5. [脚手架] 汇总表。"—" 表示标准答案没进候选集。
        print(f"\n--- 汇总（召回 {n_recall} 条）---")
        print(f"{'题号':<5}{'类型':<5}{'重排前':>7}{'重排后':>7}"
              f"{'R@3前':>7}{'R@3后':>7}{'召回ms':>8}{'重排ms':>8}")
        for item, before, after, rc_b, rc_a, t_rec, t_rr in summary:
            b = "—" if before is None else str(before)
            a = "—" if after is None else str(after)
            print(f"{item['id']:<6}{item['kind']:<6}{b:>6}{a:>7}"
                  f"{rc_b:>8}{rc_a:>8}{t_rec * 1000:>8.1f}{t_rr * 1000:>8.1f}")

        # 6. [核心] 平均耗时对比：这是两阶段设计的代价，面试常问的具体数字。
        avg_rec = sum(s[5] for s in summary) / len(summary) * 1000
        avg_rr = sum(s[6] for s in summary) / len(summary) * 1000
        print(f"\n  平均：召回 {avg_rec:.1f} ms　重排 {avg_rr:.1f} ms"
              f"　重排占比 {avg_rr / (avg_rec + avg_rr) * 100:.0f}%")


if __name__ == "__main__":
    main()
