"""
mq_01_manual.py —— 手写 Multi-Query，并和"原问题直接检索"并排对比

Multi-Query（多查询改写）一句话：
    让大模型把一个问题改写成几个不同角度的问法，每个问法各搜一遍，再把结果合并。
    它扩大的是"角度"，不是"文体"——改写出来的仍然是问句（这一点和 HyDE 不同）。

本文件做的事：
    1. 复用 hyde_01_manual 里的数据准备：抽 PDF → 拆小节 → 切块 → 建 Chroma
    2. 对每道题：大模型改写出 3 个问法，加上原问题，共 4 路
    3. 4 路分别检索；同一块取它在各路里的最高相似度，统一排序
    4. 打印前 3 块和"它是被哪一路搜到的"，最后汇总名次、R@3，和你的预测对照

本轮定下的设计（用户决定）：
    合并排序：按最高相似度　　问法：3 个改写 + 原问题
    判定规则：先比名次，名次相同再比 R@3 命中块数（跑之前定好，不看结果改）

函数索引
    generate_queries()    —— 让大模型改写问题，解析成问法列表，并把原问题放在第 0 路
    search_multi_query()  —— 4 路分别检索，按"每块最高相似度"合并排序
    hits()                —— 从 recall_at_k 的 "2/3" 里取出命中块数
    judge()               —— 按上面的判定规则给出 赢/平/输
    show_top()            —— 打印前 K 条，标出来源路
    main()                —— 串起全部流程并输出汇总表

阅读地图
    必读：MQ_PROMPT / generate_queries() —— 改写提示词怎么写、输出怎么解析
          search_multi_query()           —— 合并规则，Multi-Query 真正的难点
    扫读：judge() / hits()               —— 判定规则的实现
    跳过：show_top() / main() 里的打印    —— 终端输出
"""

# [脚手架] 标准库：re 用来去掉改写结果前面的编号。
import re

# [了解] 数据准备、对照组检索、评估函数全部从 hyde_01_manual 导入，这个文件只写 Multi-Query 本身。
#        导入时 hyde_01_manual 的顶层代码会执行（读 .env、定义常量），但 main() 不会执行，
#        因为它被 if __name__ == "__main__" 挡住了——只有直接运行那个文件时才会跑。
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

# [核心] 改写几个问法。原问题另外加在第 0 路，所以一共检索 N_REWRITE + 1 次。
N_REWRITE = 3

# [核心] 你的预测（跑之前写下的，不许改）。
MQ_PREDICT = {"Q1": "赢", "Q2": "赢", "Q3": "输", "Q4": "输"}

# [核心] 改写提示词。三个要点：
#        ① 明确要"不同角度"，否则大模型会给出三个几乎一样的句子，4 路检索等于 1 路。
#        ② 要求保留专有名词。HyDE 那轮 Q4 已经证明，"青鸾"这种生僻词是检索的救命稻草，
#           改写时丢了它，这一路就废了。
#        ③ 规定输出格式：每行一个、不加编号。后面按换行拆分，格式越死，解析越不容易出错。
MQ_PROMPT = (
    "你是检索助手。请把下面的用户问题改写成 {n} 个不同角度的检索问句，"
    "要求：\n"
    "1. 每个问句换一种说法或一个角度，不要只是换同义词；\n"
    "2. 问题里的专有名词、型号、代号必须原样保留；\n"
    "3. 每行输出一个问句，不要编号，不要任何解释。\n\n"
    "用户问题：{question}"
)


def generate_queries(llm, question: str, n: int, prompt: str = MQ_PROMPT) -> list[str]:
    """[核心] 改写出 n 个问法，返回 [原问题, 改写1, 改写2, ...]。"""
    # 1. [核心] 一次调用拿到全部改写（不是调用 n 次）：同一次回答里，模型会主动让几个问法彼此错开。
    #    [了解] prompt 参数默认用本文件的 MQ_PROMPT；自测脚本 mq_02 会传入改过的提示词做对比。
    response = llm.invoke(prompt.format(n=n, question=question))

    # 2. [了解] 按换行拆开，去掉空行。
    lines = [line.strip() for line in response.text.splitlines() if line.strip()]

    # 3. [了解] 模型有时不听话，仍然会加 "1." "2、" "- " 这类前缀，用正则去掉。
    #    这就是"让模型输出纯文本再自己解析"的脆弱之处；结构化输出（Week 3）能根治它。
    rewrites = [re.sub(r"^(\d+[.、)）]|[-*•])\s*", "", line) for line in lines]

    # 4. [核心] 原问题放在第 0 路，改写接在后面；只取前 n 个，防止模型多写。
    return [question] + rewrites[:n]


def search_multi_query(store: Chroma, queries: list[str], k: int) -> list[tuple[Document, float]]:
    """[核心] 每个问法各检索一次，按"每块在各路中的最高相似度"合并成一份排序。"""
    # 1. [核心] best 记录每一块目前见过的最小距离（= 最高相似度），key 是块在 Chroma 里的 id。
    #    用 id 而不是正文判断"是不是同一块"：两块正文偶然相同也不会被误合并。
    best: dict[str, tuple[Document, float]] = {}

    for route, query in enumerate(queries):
        # 2. [核心] 每一路都复用对照组的检索函数，保证"单路检索"的方式和原问题完全一样。
        for doc, dist in search_by_question(store, query, k=k):
            # 3. [核心] 只有这一路给出的距离更小，才替换，并记下是第几路立的功。
            if doc.id not in best or dist < best[doc.id][1]:
                doc.metadata["via"] = route  # [脚手架] 只用于打印来源，不参与排序。
                best[doc.id] = (doc, dist)

    # 4. [核心] 所有块按距离从小到大排：谁在任意一路里最像，谁就排前面。
    #    换成旧类 MultiQueryRetriever 的做法会怎样：它只拼接去重、不重新排序，
    #    第 3 路搜到的最佳答案只能排在第 1、2 路全部结果的后面。
    return sorted(best.values(), key=lambda pair: pair[1])


def hits(recall_text: str) -> int:
    """[了解] recall_at_k 返回 "2/3" 这样的字符串，取出斜杠前的命中块数。"""
    return int(recall_text.split("/")[0])


def judge(r_base: int | None, r_mq: int | None, h_base: int, h_mq: int) -> str:
    """[了解] 判定规则：先比名次（越小越好），名次相同再比 R@3 命中块数（越多越好）。"""
    # 1. [了解] 没出现在结果里的记作 999 名，保证能比较大小。
    r_base, r_mq = r_base or 999, r_mq or 999
    # 2. [了解] 名次不同，名次决定胜负。
    if r_mq != r_base:
        return "赢" if r_mq < r_base else "输"
    # 3. [了解] 名次相同，看前 K 条里多命中了几块。
    if h_mq != h_base:
        return "赢" if h_mq > h_base else "输"
    return "平"


def show_top(label: str, results: list[tuple[Document, float]], gold: str) -> None:
    """[脚手架] 打印前 K 条：名次、相似度、所属小节、来源路、正文开头。"""
    print(f"  【{label}】")
    for rank, (doc, dist) in enumerate(results[:K], start=1):
        mark = "✔" if doc.metadata["section"] == gold else " "
        via = doc.metadata.get("via")
        source = "" if via is None else ("  ←原问题" if via == 0 else f"  ←改写{via}")
        preview = doc.page_content.replace("\n", "")[:36]
        print(f"    {mark} #{rank}  相似度 {1 - dist:.3f}  {doc.metadata['section']}  {preview}…{source}")


def main() -> None:
    # 1. [了解] 数据准备与 HyDE 那一课完全相同，保证两课的对照组结果一致。
    print("=== 抽取检查 ===")
    sections = split_into_sections(load_pdf_text(PDF_PATH))
    store, _, n_chunks = build_vectorstore(sections)
    print(f"小节数：{len(sections)}　块数：{n_chunks}\n")

    # 2. [了解] temperature=0：本轮不研究随机性，让改写尽量稳定、可复现。
    llm = init_chat_model("deepseek:deepseek-v4-flash", temperature=0)

    summary = []
    for item in QUESTIONS:
        q, gold = item["q"], item["gold"]
        print(f"=== {item['id']}（{item['kind']}）{q}")
        print(f"  标准答案：{gold}　你的预测：MQ {MQ_PREDICT[item['id']]}")

        # 3. [核心] 对照组：原问题单路检索。实验组：原问题 + 3 个改写，共 4 路。
        #    k=n_chunks 取全部块的排序，这样标准答案掉到前 3 以外也知道排第几。
        base = search_by_question(store, q, k=n_chunks)
        queries = generate_queries(llm, q, N_REWRITE)
        mq = search_multi_query(store, queries, k=n_chunks)

        # 4. [核心] 把改写出的问法都打印出来：结果好坏，要回到"它改写成了什么"去解释。
        for route, query in enumerate(queries[1:], start=1):
            print(f"  改写{route}：{query}")
        show_top("原问题检索", base, gold)
        show_top("Multi-Query 检索", mq, gold)

        rc_base, rc_mq = recall_at_k(base, gold, K), recall_at_k(mq, gold, K)
        r_base, r_mq = gold_rank(base, gold), gold_rank(mq, gold)
        summary.append((item, r_base, r_mq, rc_base, rc_mq,
                        judge(r_base, r_mq, hits(rc_base), hits(rc_mq))))
        print()

    # 5. [脚手架] 汇总表。
    print("=== 汇总（判定：先比名次，名次相同再比 R@3）===")
    print(f"{'题号':<4}{'类型':<4}{'原问题':>6}{'MQ':>5}{'R@3原':>7}{'R@3MQ':>7}   实际   你的预测")
    for item, r_base, r_mq, rc_base, rc_mq, verdict in summary:
        print(f"{item['id']:<6}{item['kind']:<5}{r_base!s:>6}{r_mq!s:>5}{rc_base:>8}{rc_mq:>7}"
              f"    {verdict:<5}{MQ_PREDICT[item['id']]}")


if __name__ == "__main__":
    main()
