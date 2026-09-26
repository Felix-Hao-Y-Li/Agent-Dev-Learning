"""
evaluation/eval_02_before_after.py —— 用 RAGAs 在公开数据集上做 A/B 对比：基准检索 vs 混合检索 + 重排

A/B 对比只改一个变量：
    A 基准：原问题直接做向量检索，取前 3 篇
    B 优化：向量 + BM25 两路召回 → RRF 融合取前 20 篇 → cross-encoder 重排，取前 3 篇
    测试题、语料库、生成模型、生成 prompt、裁判模型，两边完全相同。

数据：
    RGB 中文校正版 data/rgb/zh_refine.json（arXiv 2309.01431，github.com/chen700564/RGB，
    许可 CC BY-NC-SA 4.0，仅限非商业用途）。300 题里用固定种子抽 30 题，
    这 30 题的全部正例、负例文档汇成一个语料库，一篇文档就是一块，不再切分。
    数据集不随仓库提交（许可限制），在 stage2 目录下自行下载：
        curl -L -o data/rgb/zh_refine.json https://raw.githubusercontent.com/chen700564/RGB/master/data/zh_refine.json

三把尺子并排看：
    1) 检索规则指标：前 3 篇里有几篇是正例（数据集自带标签，确定、免费）
    2) 生成规则指标：RGB 官方 checkanswer —— 回答里是否包含标准答案字符串（确定、免费）
    3) RAGAs 四个指标：DeepSeek 当裁判，同一批回答打两遍分，量出"裁判自己的波动"

裁判为什么用 DeepSeek（生成也是 DeepSeek）：
    自我偏好（裁判偏爱自家文风）会同时加在 A、B 两边，对"差值"影响很小；
    它主要影响分数的绝对值。所以本课只解读 B−A，不解读"忠实度到底是 0.8 还是 0.9"。

怎么运行（在 stage2 目录
    uv run python -m evaluation.eval_02_before_after
    预计耗时：生成 60 次 + 裁判约 1000 次调用，并发 8 路，约 10~20 分钟。

阅读地图
    必读：answer_to_reference() / check_answer() —— RGB 答案格式的两条规则
          retrieve_baseline() / retrieve_optimized() —— A、B 唯一的不同
          judge_sample()                             —— 一条样本怎么被四个指标打分
          print_report() 里"差距 vs 波动"那一段      —— 怎么判断 B 是不是真的更好
    扫读：build_corpus() / build_index() / main()
    跳过：score_with_retry() / nan_mean()            —— 重试和算平均的工具
"""

# [脚手架] 标准库：asyncio 并发调用裁判；json 读数据集、存结果；math 判断 NaN；random 固定种子抽题。
import asyncio
import json
import math
import os
import random
from pathlib import Path

# [了解] 关掉 ragas 的使用统计上报，必须在 import ragas 之前设置。
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# [了解] 各个包的角色：
#        Chroma / langchain HuggingFaceEmbeddings —— 建向量库（和 stage2 前几课同一套）
#        ragas HuggingFaceEmbeddings —— 给 Answer Relevancy 用的 embedding（ragas 自己的包装，不是同一个类）
#        llm_factory + AsyncOpenAI    —— 把 DeepSeek 包装成裁判（DeepSeek 提供 OpenAI 兼容接口）
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

# [核心] 复用前几课写好的检索函数和本课补上的生成函数，保证 A、B 用的就是你学过的那两套系统。
from common.corpus import search_by_question
from common.generate import answer_with_contexts, build_generator
from hybrid_retrieval.hybrid_01_bm25_rrf import build_bm25, search_hybrid
from reranking.rerank_01_cross_encoder import load_reranker, rerank

load_dotenv(".env")

RGB_PATH = Path("data/rgb/zh_refine.json")
N_QUESTIONS = 30      # [了解] 抽多少题。题越多结论越稳，裁判调用也越多。
SEED = 42             # [核心] 固定种子：每次运行抽到的都是同一批题，测试集就被"冻结"了。
K = 3                 # [核心] 两套系统都只把前 3 篇交给生成模型。
N_RECALL = 20         # [了解] B 的第一阶段召回多少篇进重排。候选太少重排没得挑，太多重排变慢。
N_RUNS = 3            # [核心] 同一批回答让裁判打几遍分，用来量出裁判自己的波动。自测 1 从 2 改成 3。
CONCURRENCY = 8       # [脚手架] 同时进行的裁判任务数。
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RESULT_PATH = Path("evaluation/results/eval_02_raw.json")
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_rgb_subset() -> list[dict]:
    """[了解] 读 RGB 全部 300 题，用固定种子抽出 N_QUESTIONS 题。"""
    # 1. [了解] 文件是 JSON Lines 格式：每一行是一道题的 JSON。
    with RGB_PATH.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    # 2. [核心] random.Random(SEED) 是一个独立的随机数生成器，种子固定，抽样结果就固定。
    picked = random.Random(SEED).sample(items, N_QUESTIONS)
    return sorted(picked, key=lambda x: x["id"])


def answer_to_reference(answer: list) -> str:
    """[核心] RGB 答案列表 → RAGAs 需要的 reference 字符串。只拼接，不加任何字。"""
    # 1. [核心] 外层每一项都必须答到，用顿号连起来：['东盟','中国'] → "东盟、中国"
    # 2. [核心] 内层是几种等价说法，写成"甲（或 乙）"：[['1313851','131万']] → "1313851（或 131万）"
    parts = []
    for item in answer:
        if isinstance(item, list):
            parts.append(item[0] + "".join(f"（或 {alt}）" for alt in item[1:]))
        else:
            parts.append(item)
    return "、".join(parts)


def check_answer(prediction: str, answer: list) -> bool:
    """[核心] RGB 官方 checkanswer 的规则（evalue.py），外层每一项都命中才算答对。"""
    # 1. [核心] 忽略大小写，看标准答案字符串是否出现在回答里。
    prediction = prediction.lower()
    for item in answer:
        # 2. [核心] 内层列表：几种说法命中任意一种即可。
        alternatives = item if isinstance(item, list) else [item]
        if not any(alt.lower() in prediction for alt in alternatives):
            return False
    return True


def build_corpus(items: list[dict]) -> list[Document]:
    """[了解] 30 题的全部正例、负例文档 → 一个语料库。一篇文档就是一块。"""
    # 1. [了解] 同一段文字可能在不同题里重复出现，用正文当键去重。
    #    pos_for 记录"这篇是哪些题的正例"，写成 ",12,45," 这种字符串，
    #    因为 Chroma 的 metadata 只能存字符串、数字这类简单值，不能存列表。
    by_text: dict[str, str] = {}
    for item in items:
        for text in item["positive"]:
            by_text[text] = by_text.get(text, ",") + f"{item['id']},"
        for text in item["negative"]:
            by_text.setdefault(text, ",")
    return [Document(page_content=t, metadata={"pos_for": p}) for t, p in by_text.items()]


def build_index(docs: list[Document]) -> tuple[Chroma, object]:
    """[了解] 建向量库（A、B 都用）和 BM25 索引（只有 B 用）。"""
    # 1. [了解] 和 common/corpus.py 同一套配置：bge-small-zh、归一化、余弦距离。
    embeddings = LCHuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
    store = Chroma(collection_name="rgb_eval", embedding_function=embeddings,
                   collection_configuration={"hnsw": {"space": "cosine"}})
    store.add_documents(docs)
    # 2. [了解] BM25 用 jieba 分词建索引，函数来自混合检索那一课。
    return store, build_bm25(docs)


def retrieve_baseline(store: Chroma, question: str) -> list[Document]:
    """[核心] A 基准：原问题直接做向量检索，取前 K 篇。"""
    return [doc for doc, _ in search_by_question(store, question, k=K)]


def retrieve_optimized(store, bm25, docs, reranker, question: str) -> list[Document]:
    """[核心] B 优化：混合召回 N_RECALL 篇 → cross-encoder 重排 → 取前 K 篇。"""
    # 1. [核心] 第一阶段求"不漏"：向量、BM25 各取 N_RECALL 篇，RRF 融合后取前 N_RECALL 篇。
    candidates = [doc for doc, _ in search_hybrid(store, bm25, docs, question, k=N_RECALL)[:N_RECALL]]
    # 2. [核心] 第二阶段求"排准"：问题和每篇候选成对打分，按分数重排，只留前 K 篇。
    return [doc for doc, _ in rerank(reranker, question, candidates)[:K]]


def count_positives(docs: list[Document], qid: int) -> int:
    """[了解] 检索规则指标：前 K 篇里有几篇是这道题的正例。"""
    return sum(f",{qid}," in d.metadata["pos_for"] for d in docs)


def build_judge_metrics() -> dict:
    """[核心] 裁判 + 四个指标。"""
    # 1. [核心] DeepSeek 的 OpenAI 兼容地址。max_tokens=4096：deepseek-v4-flash 默认开启思考模式，
    #    会先"思考"再输出，ragas 默认的 1024 不够用，会报"输出因长度上限被截断"（本课实测踩到的坑）。
    #    同样因为思考模式：ragas 默认发送的 temperature=0.01 被 DeepSeek 静默忽略，
    #    top_p=0.1 被当作 0.95 处理（api-docs.deepseek.com/guides/thinking_mode）。
    #    所以裁判的判决并没有被"调成几乎确定"，表里波动为 0 不是温度的功劳。
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    llm = llm_factory("deepseek-v4-flash", client=client, max_tokens=4096)
    # 2. [核心] 只有 Answer Relevancy 需要 embedding，用和检索同一个中文模型。
    embeddings = RagasHuggingFaceEmbeddings(model=EMBED_MODEL, normalize_embeddings=True)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_with_retry(metric, fields: dict, tries: int = 3) -> float:
    """[脚手架] 调一次 ascore；网络抖动或限流时等一会儿重试，最后仍失败就记为 NaN 并打印原因。"""
    for attempt in range(1, tries + 1):
        try:
            return (await metric.ascore(**fields)).value
        except Exception as e:
            if attempt == tries:
                print(f"    [打分失败，记为 NaN] {type(e).__name__}: {str(e)[:120]}")
                return float("nan")
            await asyncio.sleep(10 * attempt)


async def judge_sample(metrics: dict, sample: dict) -> dict[str, float]:
    """[核心] 一条样本 → 四个指标的分数。每个指标只拿它需要的字段。"""
    s = sample
    return {
        # 1. [核心] 忠实度：回答 vs 检索块，不需要标准答案。
        "faithfulness": await score_with_retry(metrics["faithfulness"], {
            "user_input": s["user_input"], "response": s["response"], "retrieved_contexts": s["retrieved_contexts"]}),
        # 2. [核心] 回答相关性：回答 vs 问题。回答是"文档信息不足……"这类拒答时，会被判为含糊其辞，直接 0 分。
        "answer_relevancy": await score_with_retry(metrics["answer_relevancy"], {
            "user_input": s["user_input"], "response": s["response"]}),
        # 3. [核心] 上下文精确率：有用的块是否排在前面（参照标准答案判断"有用"）。
        "context_precision": await score_with_retry(metrics["context_precision"], {
            "user_input": s["user_input"], "reference": s["reference"], "retrieved_contexts": s["retrieved_contexts"]}),
        # 4. [核心] 上下文召回率：标准答案的信息，检索块里有没有。
        "context_recall": await score_with_retry(metrics["context_recall"], {
            "user_input": s["user_input"], "retrieved_contexts": s["retrieved_contexts"], "reference": s["reference"]}),
    }


async def judge_all(metrics: dict, samples: list[dict]) -> list[dict]:
    """[了解] 并发给一批样本打分。Semaphore 限制同时进行的任务数，避免把接口打爆。"""
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(sample):
        async with sem:
            return await judge_sample(metrics, sample)

    return await asyncio.gather(*(one(s) for s in samples))


def nan_mean(values: list[float]) -> float:
    """[脚手架] 忽略 NaN 求平均（NaN 表示这一条没打出分，比如回答拆不出任何断言）。"""
    ok = [v for v in values if not math.isnan(v)]
    return sum(ok) / len(ok) if ok else float("nan")


def print_report(samples: dict, scores: dict) -> None:
    """[核心] 三把尺子的汇总，以及"差距 vs 波动"的判断。"""
    n = N_QUESTIONS
    # 1. [核心] 两把确定性的尺子：不调用裁判，跑多少遍都一样。
    print(f"\n{'=' * 72}\n规则指标（确定性）")
    for name in ("A", "B"):
        pos = sum(s["n_pos"] for s in samples[name]) / n
        acc = sum(s["correct"] for s in samples[name])
        rej = sum("信息不足" in s["response"] for s in samples[name])
        print(f"  {name}：前 {K} 篇平均正例 {pos:.2f} 篇 | RGB 官方准确率 {acc}/{n} | 拒答 {rej} 题")

    # 2. [核心] RAGAs：每个指标、每个系统、每一遍的平均分都列出来。
    #    波动 = 同一系统各遍平均分里"最高减最低"（极差），表示"什么都没改，分数自己会晃多少"。
    #    自测 1 改动：原来写死了 a[0]-a[1]，只能处理 2 遍；改成极差后，N_RUNS 取几都能用。
    print(f"\nRAGAs（裁判 DeepSeek，同一批回答打 {N_RUNS} 遍）")
    for m in METRICS:
        a = [nan_mean([r[m] for r in run]) for run in scores["A"]]
        b = [nan_mean([r[m] for r in run]) for run in scores["B"]]
        a_noise, b_noise = max(a) - min(a), max(b) - min(b)
        noise = max(a_noise, b_noise)
        diff = sum(b) / len(b) - sum(a) / len(a)
        # 3. [核心] 只有差距明显大于波动，才能说 B 和 A 真的不同。这里用"大于 2 倍波动"作粗略门槛。
        #    注意：波动是从少数几遍里估出来的，遍数少时可能碰巧为 0，这个门槛就会失效（见本课第 4 节）。
        verdict = "差距明显大于波动" if abs(diff) > 2 * noise else "在波动范围内，不能下结论"
        print(f"  {m:<18} A 各遍 {' '.join(f'{x:.3f}' for x in a)}（波动 {a_noise:.3f}）"
              f" | B 各遍 {' '.join(f'{x:.3f}' for x in b)}（波动 {b_noise:.3f}）"
              f" | B−A {diff:+.3f} → {verdict}")

    # 4. [核心] 自测 3：统计 NaN（没打出分的条目）。nan_mean 会静默跳过它们，
    #    所以两边的"有效题数"可能不同，平均分的分母也就不同，必须摆出来。
    print("\n没打出分的条目数（NaN，按 系统/指标/各遍）")
    for name in ("A", "B"):
        counts = {m: [sum(math.isnan(r[m]) for r in run) for run in scores[name]] for m in METRICS}
        print(f"  {name}：" + "  ".join(f"{m} {counts[m]}" for m in METRICS))


async def main() -> None:
    # 1. [了解] 准备数据：抽题 → 建语料库 → 建索引 → 加载重排模型和生成模型。
    items = load_rgb_subset()
    docs = build_corpus(items)
    print(f"抽题 {len(items)} 道（种子 {SEED}），语料库 {len(docs)} 篇文档")
    store, bm25 = build_index(docs)
    reranker, generator = load_reranker(), build_generator()

    # 2. [核心] 对每道题：A、B 各检索一次、各生成一次回答。生成只做一次，后面裁判打两遍分用的是同一批回答，
    #    这样两遍之间的差异只可能来自裁判。
    samples = {"A": [], "B": []}
    for i, item in enumerate(items, start=1):
        q = item["query"]
        for name, retrieved in (("A", retrieve_baseline(store, q)),
                                ("B", retrieve_optimized(store, bm25, docs, reranker, q))):
            contexts = [d.page_content for d in retrieved]
            response = answer_with_contexts(generator, q, contexts)
            samples[name].append({
                "id": item["id"], "user_input": q, "retrieved_contexts": contexts, "response": response,
                "reference": answer_to_reference(item["answer"]),
                "n_pos": count_positives(retrieved, item["id"]), "correct": check_answer(response, item["answer"]),
            })
        print(f"  生成 {i}/{len(items)}", end="\r")

    # 3. [核心] 裁判打分：每个系统打 N_RUNS 遍。
    metrics = build_judge_metrics()
    scores = {"A": [], "B": []}
    for run in range(1, N_RUNS + 1):
        for name in ("A", "B"):
            print(f"\n裁判打分：系统 {name} 第 {run} 遍 ……")
            scores[name].append(await judge_all(metrics, samples[name]))

    # 4. [脚手架] 存下全部原始结果，方便事后逐题翻看。
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({"samples": samples, "scores": scores}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print_report(samples, scores)
    print(f"\n原始结果已保存：{RESULT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
