"""
mq_02_selftests.py —— Multi-Query 三道自测题的对照实验，每种配置重复跑 TRIALS 轮

为什么要重复跑：
    mq_01 不改代码跑两次，Q1 从第 5 名变成第 9 名、Q2 从输变成赢。
    波动来自大模型改写这一步，单跑一次得出的结论不可信，所以每种配置都跑多轮，看分布。

四种配置（每次只改一个变量）：
    A  基线          原始提示词（含背景句）+ 最高分合并 + 含原问题
    T1 删背景句       提示词删掉"用于在企业内部文档和技术手册中做向量检索"，其余同 A
    T2 投票合并       合并规则改成"先比被几路搜进前 5，再比最高相似度"，其余同 A
    T3 去掉原问题      只用 3 个改写检索，其余同 A

配对比较（本文件最重要的设计）：
    T2、T3 和 A 用的是【同一轮生成的同一组改写】，只换合并方式或去掉一路。
    所以 A 与 T2/T3 的差别，只可能来自那一处改动，不会混进"这次改写运气好不好"。
    T1 改了提示词，改写必须重新生成，所以它和 A 的差别里仍然混着改写的随机性。

函数索引
    search_vote()  —— T2 的投票合并规则
    leak_count()   —— 数一数改写里抄进了几次"企业内部 / 技术手册"
    fmt()          —— 把多轮结果排成一格文字
    main()         —— 跑 TRIALS 轮 × 4 道题 × 4 种配置，打印汇总

阅读地图
    必读：search_vote()                 —— 和 mq_01 的 search_multi_query() 对照着读
          main() 里生成 q_old / q_new 那几行 —— 配对比较是怎么实现的
    跳过：fmt() 和打印                   —— 终端输出
"""

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document

from common.corpus import (
    K,
    PDF_PATH,
    QUESTIONS,
    build_vectorstore,
    gold_rank,
    load_pdf_text,
    recall_at_k,
    search_by_question,
    split_into_sections,
)
from query_transformation.mq_01_manual import MQ_PROMPT as PROMPT_NEW  # [了解] 你已删掉背景句的版本 → T1 用
from query_transformation.mq_01_manual import N_REWRITE, generate_queries, hits, search_multi_query

# [核心] 重复轮数。3 轮只够看出"稳不稳"，不够做统计检验；正式评估要几十上百道题（RAGAs 那一课）。
TRIALS = 3

# [核心] 投票时"算搜到"的深度：某一路把这块排进前 VOTE_DEPTH，就给它投一票。
VOTE_DEPTH = 5

# [了解] 原始提示词，一字不差地保留在这里作为对照组 A。
PROMPT_OLD = (
    "你是检索助手。请把下面的用户问题改写成 {n} 个不同角度的检索问句，"
    "用于在企业内部文档和技术手册中做向量检索。\n"
    "要求：\n"
    "1. 每个问句换一种说法或一个角度，不要只是换同义词；\n"
    "2. 问题里的专有名词、型号、代号必须原样保留；\n"
    "3. 每行输出一个问句，不要编号，不要任何解释。\n\n"
    "用户问题：{question}"
)

CONFIGS = ["A 基线", "T1 删背景句", "T2 投票合并", "T3 去掉原问题"]


def search_vote(store: Chroma, queries: list[str], k: int) -> list[tuple[Document, float]]:
    """[核心] 投票合并：先比"被几路排进前 VOTE_DEPTH"，票数相同再比最高相似度。"""
    # 1. [核心] votes 记每块得了几票，best 记每块在各路里的最小距离（和最高分合并一样）。
    votes: dict[str, int] = {}
    best: dict[str, tuple[Document, float]] = {}

    for query in queries:
        results = search_by_question(store, query, k=k)
        # 2. [核心] 这一路的前 VOTE_DEPTH 名，每块得 1 票。一路最多给一块投 1 票。
        for doc, _ in results[:VOTE_DEPTH]:
            votes[doc.id] = votes.get(doc.id, 0) + 1
        # 3. [了解] 同时照常记录最高相似度，作为票数相同时的第二排序依据。
        for doc, dist in results:
            if doc.id not in best or dist < best[doc.id][1]:
                best[doc.id] = (doc, dist)

    # 4. [核心] 排序键是 (-票数, 距离)：票数多的在前；票数一样，更像的在前。
    #    和最高分合并的区别：一路跑偏只能给干扰块投 1 票，压不过被 3、4 路共同搜到的正确块。
    return sorted(best.values(), key=lambda pair: (-votes.get(pair[0].id, 0), pair[1]))


def leak_count(queries: list[str]) -> int:
    """[了解] 数改写里有几条抄进了提示词背景词（不算第 0 路原问题）。"""
    return sum(1 for q in queries[1:] if "企业内部" in q or "技术手册" in q)


def fmt(runs: list[tuple[int | None, int]]) -> str:
    """[脚手架] 把多轮 [(名次, 命中数), ...] 排成 "名次 5/9/5 命中 0/0/1"。"""
    ranks = "/".join(str(r) if r else "-" for r, _ in runs)
    hit = "/".join(str(h) for _, h in runs)
    return f"名次 {ranks:<8} 命中 {hit}"


def main() -> None:
    # 1. [了解] 数据准备和前两课相同。
    sections = split_into_sections(load_pdf_text(PDF_PATH))
    store, _, n_chunks = build_vectorstore(sections)
    llm = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

    # 2. [了解] records[配置][题号] = [(名次, R@3 命中数), ...]，每轮追加一个。
    records = {c: {item["id"]: [] for item in QUESTIONS} for c in CONFIGS}
    leaks = {"old": 0, "new": 0}

    for trial in range(1, TRIALS + 1):
        for item in QUESTIONS:
            q, gold = item["q"], item["gold"]

            # 3. [核心] 每轮每题只生成两组改写：旧提示词一组、新提示词一组。
            #    A、T2、T3 共用 q_old——这就是配对比较。
            q_old = generate_queries(llm, q, N_REWRITE, PROMPT_OLD)
            q_new = generate_queries(llm, q, N_REWRITE, PROMPT_NEW)
            leaks["old"] += leak_count(q_old)
            leaks["new"] += leak_count(q_new)

            # 4. [核心] 四种配置各检索一次。T3 的 q_old[1:] 就是"去掉第 0 路原问题"。
            results = {
                "A 基线": search_multi_query(store, q_old, k=n_chunks),
                "T1 删背景句": search_multi_query(store, q_new, k=n_chunks),
                "T2 投票合并": search_vote(store, q_old, k=n_chunks),
                "T3 去掉原问题": search_multi_query(store, q_old[1:], k=n_chunks),
            }
            for name, res in results.items():
                records[name][item["id"]].append((gold_rank(res, gold), hits(recall_at_k(res, gold, K))))
        print(f"第 {trial}/{TRIALS} 轮完成")

    # 5. [脚手架] 原问题检索是确定的，算一次作为参照。
    print(f"\n=== 汇总：{TRIALS} 轮结果（名次越小越好，命中 = 前 {K} 条里属于标准答案小节的块数）===")
    for item in QUESTIONS:
        base = search_by_question(store, item["q"], k=n_chunks)
        print(f"\n{item['id']}（{item['kind']}，标准答案 {item['gold']}）"
              f"原问题检索：名次 {gold_rank(base, item['gold'])}  命中 {hits(recall_at_k(base, item['gold'], K))}")
        for name in CONFIGS:
            print(f"    {name:<10}{fmt(records[name][item['id']])}")

    total = TRIALS * len(QUESTIONS) * N_REWRITE
    print(f"\n改写里抄进'企业内部/技术手册'的条数：原始提示词 {leaks['old']}/{total}，删背景句后 {leaks['new']}/{total}")


if __name__ == "__main__":
    main()
