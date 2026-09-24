"""
hyde_01_manual.py —— 手写 HyDE，并和"原问题直接检索"并排对比

HyDE（Hypothetical Document Embeddings，假设文档嵌入）一句话：
    不拿"问题"去搜，而是先让大模型编一段"假答案"，拿假答案去搜。
    问题是短问句，文档是说明文，两者长得不像；假答案本身就是说明文，和文档长得像。
    论文：Gao, Ma, Lin, Callan. Precise Zero-Shot Dense Retrieval without Relevance Labels. arXiv:2212.10496

本文件做的事：
    1. 数据准备（抽 PDF → 拆小节 → 切块 → 建 Chroma）全部调用 common/corpus.py，本文件不再重复
    2. 对 4 道测试题，分别用"原问题"和"HyDE 假答案"检索，打印前 3 块和标准答案排第几

运行方式（在 stage2 目录下）：
    uv run python -m query_transformation.hyde_01_manual

函数索引
    generate_hypothetical_doc()  —— HyDE 的第一步：让大模型编一份假答案
    generate_hypothetical_docs() —— 一次编 N 份假答案（论文公式 8）
    search_by_hyde()             —— 实验组：单份假答案的向量检索
    search_by_hyde_avg()         —— 实验组：N 份假答案 + 原问题，向量取平均后检索
    show_top()                   —— 打印前 k 条结果
    main()                       —— 串起全部流程并输出汇总表

阅读地图
    必读：generate_hypothetical_doc() —— 提示词为什么这样写，HyDE 的全部"魔法"都在这
          search_by_hyde()            —— 假答案用什么方式向量化、怎么检索
    扫读：search_by_hyde_avg()        —— 多份假答案取平均
    跳过：show_top() / main() 里的打印 —— 终端输出
    另见：common/corpus.py 的 search_by_question() —— 对照组，和 search_by_hyde() 逐行对比着读
"""

# [了解] 各个包的角色：
#        init_chat_model —— LangChain v1 统一的"按名字创建聊天模型"入口
#        Chroma          —— 向量库，这里只放在内存里，程序结束就消失
#        Document        —— LangChain 的文档对象：page_content（正文）+ metadata（元数据字典）
#        HuggingFaceEmbeddings —— 把本地 sentence-transformers 模型包装成 LangChain 的向量化接口
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# [核心] 数据准备和评估已经搬到 stage2/common/corpus.py，各课共用同一套语料和口径。
#        本文件从这里只留 HyDE 本身的代码。运行方式随之改成（在 stage2 目录下）：
#            uv run python -m query_transformation.hyde_01_manual
#        直接 uv run query_transformation/hyde_01_manual.py 会找不到 common 包。
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

# [了解] 一个问题生成几份假答案，对应论文公式 8 里的 d̂1…d̂N。
N_HYPO = 4

# [核心] HyDE 的提示词。三个要点：
#        ① 让模型写"文档里的一段话"，而不是"回答用户"——我们要的是和语料同一种文体的文字。
#        ② 长度贴近块大小（约 150 字）。太长会混进无关内容，向量被稀释；太短又退化成问句。
#        ③ 绝不能写"不知道就说不知道"。模型一旦回答"我不清楚"，这句话的向量毫无检索价值。
#           HyDE 要的就是"敢编"——编错的细节交给后面的向量相似度去过滤。
HYDE_PROMPT = (
    "请写一段企业内部文档或技术手册中的段落，用来回答下面的问题。\n"
    "要求：约 150 字，书面语，直接陈述内容，不要客套话，不要提到“问题”本身。\n\n"
    "问题：{question}\n"
    "段落："
)


def generate_hypothetical_doc(llm, question: str) -> str:
    """[核心] HyDE 第一步：让大模型针对问题编一段"假答案"。"""
    # 1. [核心] 把问题填进提示词，调用大模型。只调用一次，得到一段文字。
    response = llm.invoke(HYDE_PROMPT.format(question=question))
    # 2. [了解] response 是 AIMessage，.text 取出纯文本内容。
    return response.text.strip()

def generate_hypothetical_docs(llm, question: str, n: int) -> list[str]:
    """[核心] 对同一个问题生成 n 份假答案，对应论文公式 8 里的 d̂1…d̂N。"""
    # 1. [核心] batch 把同一个提示词并发发送 n 次。
    #    配合 temperature=0.7，n 份答案各不相同——要的就是这种"不同"，平均才有意义。
    prompt = HYDE_PROMPT.format(question=question)
    responses = llm.batch([prompt] * n)
    # 2. [了解] 每个元素都是 AIMessage，取出纯文本。
    return [r.text.strip() for r in responses]


def search_by_hyde(store: Chroma, embeddings: HuggingFaceEmbeddings,
                   hypo_doc: str, k: int) -> list[tuple[Document, float]]:
    """[核心] 实验组：用假答案的向量检索。"""
    # 1. [核心] 把假答案当"文档"来向量化（embed_documents），而不是当"查询"（embed_query）。
    #    论文里假答案走的就是文档那一侧的编码。对 bge 而言两者此处相同，
    #    但有些模型会给查询加前缀，两边走错会导致向量不在同一个"文体区域"。
    vector = embeddings.embed_documents([hypo_doc])[0]
    # 2. [核心] 直接拿向量去检索，跳过"文字 → 向量"那一步，因为向量已经算好了。
    return store.similarity_search_by_vector_with_relevance_scores(vector, k=k)

def search_by_hyde_avg(store: Chroma, embeddings: HuggingFaceEmbeddings,
                       question: str, hypo_docs: list[str], k: int) -> list[tuple[Document, float]]:
    """[核心] 论文公式 8：n 份假答案的向量 + 原问题的向量，共 n+1 个，取平均后检索。"""
    # 1. [核心] n 份假答案当文档向量化，再把原问题的向量也加进去。
    #    原问题是"唯一确定不会编错"的信息，放进平均里能把方向往回拉。
    vectors = embeddings.embed_documents(hypo_docs) + [embeddings.embed_query(question)]
    # 2. [核心] axis=0 表示按列平均：结果向量的每一维，都是 n+1 个向量在这一维上的平均值。
    avg = np.mean(vectors, axis=0)
    # 3. [了解] 几个长度为 1 的向量取平均后，长度会小于 1（方向越分散越短）。
    #    重新除以长度，让它回到长度 1，和库里的向量保持同一尺度。
    avg = avg / np.linalg.norm(avg)
    # 4. [脚手架] Chroma 要的是 Python 列表，不是 numpy 数组，用 tolist() 转换。
    return store.similarity_search_by_vector_with_relevance_scores(avg.tolist(), k=k)


def show_top(label: str, results: list[tuple[Document, float]], gold: str) -> None:
    """[脚手架] 打印前 K 条：名次、余弦相似度、所属小节、正文开头。"""
    print(f"  【{label}】")
    for rank, (doc, dist) in enumerate(results[:K], start=1):
        mark = "✔" if doc.metadata["section"] == gold else " "
        preview = doc.page_content.replace("\n", "")[:40]
        print(f"    {mark} #{rank}  相似度 {1 - dist:.3f}  {doc.metadata['section']}  {preview}…")


def main() -> None:
    # 1. [了解] 数据准备：抽文字 → 拆小节 → 建向量库。
    print("=== 抽取检查 ===")
    text = load_pdf_text(PDF_PATH)
    sections = split_into_sections(text)
    store, embeddings, n_chunks = build_vectorstore(sections)
    print(f"小节数：{len(sections)}　块数：{n_chunks}\n")

    # 2. [了解] temperature=0 让假答案尽量稳定，重复运行结果可复现。
    llm_sampling = init_chat_model("deepseek:deepseek-v4-flash", temperature=0.7)

    summary = []
    for item in QUESTIONS:
        q, gold = item["q"], item["gold"]
        print(f"=== {item['id']}（{item['kind']}）{q}")
        print(f"  标准答案：{gold}　预测：{item['predict']}")

        # 3. [核心] 两组检索都取"全部块"的排序（k=n_chunks），这样即使标准答案不在前 3，
        #    也能知道它掉到了第几名；打印时只显示前 K 条。
        base = search_by_question(store, q, k=n_chunks)
        hypos = generate_hypothetical_docs(llm_sampling, q, N_HYPO)
        hypo = hypos[0]  # [脚手架] 下面的打印只展示第 1 份假答案
        hyde = search_by_hyde_avg(store, embeddings, q, hypos, k=n_chunks)

        # 4. [核心] 一定要把假答案打印出来：它编了什么、编错了什么，是解释结果的唯一依据。
        print(f"  假答案：{hypo}")
        show_top("原问题检索", base, gold)
        show_top("HyDE 检索", hyde, gold)
        summary.append((item, gold_rank(base, gold), gold_rank(hyde, gold),
                        recall_at_k(base, gold, K), recall_at_k(hyde, gold, K)))
        print()

    # 5. [脚手架] 汇总表：名次越小越好。
    print("=== 汇总：标准答案小节的名次（越小越好）===")
    print(f"{'题号':<4}{'类型':<4}{'原问题':>6}{'HyDE':>6}{'R@3原':>7}{'R@3HyDE':>9}   实际结果    预测")
    for item, r_base, r_hyde, rc_base, rc_hyde in summary:
        if r_base == r_hyde:
            verdict = "平手"
        else:
            verdict = "HyDE 赢" if (r_hyde or 999) < (r_base or 999) else "HyDE 输"
        print(f"{item['id']:<6}{item['kind']:<5}{r_base!s:>6}{r_hyde!s:>6}{rc_base:>8}{rc_hyde:>9}   {verdict:<10}{item['predict']}")


if __name__ == "__main__":
    main()
