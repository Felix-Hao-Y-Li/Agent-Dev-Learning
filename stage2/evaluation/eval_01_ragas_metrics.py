"""
evaluation/eval_01_ragas_metrics.py —— 用 RAGAs 官方例子，逐个看懂四个核心指标怎么打分

这一课只做一件事：
    把 RAGAs 官方文档里每个指标页的例子原样搬过来，让 Qwen 当裁判重新打分，
    再和官方给出的分数并排对照。例子一律照抄官方，不自己编。

两组对照（只换"语言 + embedding 模型"，其余全部相同）：
    英文组：官方原文              + BAAI/bge-small-en-v1.5
    中文组：官方原文的忠实中文译本 + BAAI/bge-small-zh-v1.5
    embedding（把文字变成向量，用来比较两段话语义有多近）只有 Answer Relevancy 用得到，
    另外三个指标完全靠裁判模型判断。

怎么运行（在 stage2 目录下）：
    uv run python -m evaluation.eval_01_ragas_metrics

依据：
    docs.ragas.io 各指标页（ragas 0.4.3，查证于 2026-09-25），每条例子旁边标了出处页面。
    百炼 OpenAI 兼容模式：alibabacloud.com/help/zh/model-studio/compatibility-of-openai-with-dashscope

环境备注：
    ragas 0.4.3 在 import 时会去找 langchain_community.chat_models.vertexai，
    而 langchain-community 0.4.2 删掉了这个模块，所以 pyproject 里把 community 锁在 <0.4.2
    （ragas 官方 issue #2745）。等 ragas 发了修复版再解锁。

阅读地图
    必读：EXAMPLES                 —— 每条官方例子在测哪个指标、官方给了多少分
          build_metrics()          —— 四个指标各需要什么"零件"（裁判 / embedding）
          show_faithfulness_steps() —— 忠实度"先拆断言、再逐条核对"的中间过程
    扫读：build_judge() / score_with_retry() / run_group()
    跳过：print_table()            —— 只是排版
"""

# [脚手架] 标准库：asyncio 跑异步打分（ragas 的 ascore 是异步函数），也用它的 sleep 做重试等待；os 读环境变量。
import asyncio
import os

# [了解] 关掉 ragas 的使用统计上报（它默认会把"用了哪个模型、调了几次"发回官方服务器）。
#        必须在 import ragas 之前设置，否则已经晚了。
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# [了解] 各个包的角色：
#        dotenv       —— 从 stage2/.env 读 DASHSCOPE_API_KEY
#        AsyncOpenAI  —— OpenAI 官方 SDK 的异步客户端。百炼提供了"OpenAI 兼容模式"，
#                        所以换个 base_url 就能用同一套 SDK 调 Qwen
#        llm_factory  —— ragas 现行的"把一个客户端包装成裁判"的入口
#        HuggingFaceEmbeddings —— ragas 自己的本地 embedding 包装（不是 LangChain 那个同名类）
#        metrics.collections —— ragas 0.4 现行的指标类；旧的 ragas.metrics + single_turn_ascore 已标记废弃
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextUtilization,
    Faithfulness,
)

# [脚手架] 在 stage2 目录下运行，.env 就在当前目录。
load_dotenv(".env")

# [了解] 裁判模型：阿里云百炼的 qwen3.8-flash，走"OpenAI 兼容模式"，所以用 OpenAI 的 SDK 就能调。
#        被评估的系统（后面 eval_02 里）用 DeepSeek 生成回答，裁判换成另一家，避免"自己给自己打分"。
#        为什么不用 Gemini：免费额度每个模型每天只有 20 次请求，这一课就要约 42 次（2026-09-25 实测）。
#        base_url 用北京区的经典域名；官方现在推荐带 WorkspaceId 的新域名，旧域名仍可用。
JUDGE_MODEL = "qwen3.8-flash"
JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODELS = {"en": "BAAI/bge-small-en-v1.5", "zh": "BAAI/bge-small-zh-v1.5"}


# [核心] 官方例子。每一条都写明：测哪个指标、官方给了多少分、出处、英文原文、中文译本。
#        official 是官方页面上印出来的分数；页面只给了"高/低"而没给数字的，就写文字。
#        en 里的字符串逐字照抄官方（包括官方原文里的拼写，比如 "it's"）。
#        zh 是忠实翻译：事实、日期、句子结构都不动，只换语言。
_DOC = "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/"
EXAMPLES = [
    # ---- Faithfulness（忠实度）：回答里的断言，有几成能从检索块里推出来 ----
    {"id": "F1", "metric": "faithfulness", "official": "高（1.0）",
     "note": "官方高分例：日期与上下文一致", "src": _DOC + "faithfulness/",
     "en": {"user_input": "Where and when was Einstein born?",
            "response": "Einstein was born in Germany on 14th March 1879.",
            "retrieved_contexts": ["Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time"]},
     "zh": {"user_input": "爱因斯坦在哪里、何时出生？",
            "response": "爱因斯坦于1879年3月14日出生于德国。",
            "retrieved_contexts": ["阿尔伯特·爱因斯坦（生于1879年3月14日）是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大、最具影响力的科学家之一"]}},
    {"id": "F2", "metric": "faithfulness", "official": "0.5",
     "note": "官方低分例：日期写错，2 条断言只有 1 条被支持", "src": _DOC + "faithfulness/",
     "en": {"user_input": "Where and when was Einstein born?",
            "response": "Einstein was born in Germany on 20th March 1879.",
            "retrieved_contexts": ["Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time"]},
     "zh": {"user_input": "爱因斯坦在哪里、何时出生？",
            "response": "爱因斯坦于1879年3月20日出生于德国。",
            "retrieved_contexts": ["阿尔伯特·爱因斯坦（生于1879年3月14日）是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大、最具影响力的科学家之一"]}},
    {"id": "F3", "metric": "faithfulness", "official": "1.0",
     "note": "官方代码示例：上下文没出现'超级碗'这个词", "src": _DOC + "faithfulness/",
     "en": {"user_input": "When was the first super bowl?",
            "response": "The first superbowl was held on Jan 15, 1967",
            "retrieved_contexts": ["The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."]},
     "zh": {"user_input": "第一届超级碗是什么时候举行的？",
            "response": "第一届超级碗于1967年1月15日举行",
            "retrieved_contexts": ["首届 AFL–NFL 世界冠军赛是一场美式橄榄球比赛，于1967年1月15日在洛杉矶的洛杉矶纪念体育场举行。"]}},

    # ---- Context Precision（上下文精确率）：有用的块是不是排在前面 ----
    {"id": "P1", "metric": "context_precision", "official": "0.9999999999",
     "note": "官方例：有用的块排第 1", "src": _DOC + "context_precision/",
     "en": {"user_input": "Where is the Eiffel Tower located?",
            "reference": "The Eiffel Tower is located in Paris.",
            "retrieved_contexts": ["The Eiffel Tower is located in Paris.", "The Brandenburg Gate is located in Berlin."]},
     "zh": {"user_input": "埃菲尔铁塔位于哪里？",
            "reference": "埃菲尔铁塔位于巴黎。",
            "retrieved_contexts": ["埃菲尔铁塔位于巴黎。", "勃兰登堡门位于柏林。"]}},
    {"id": "P2", "metric": "context_utilization", "official": "0.49999999995",
     "note": "官方例：无关块被排到第 1（官方这里换用了不需要标准答案的变体）", "src": _DOC + "context_precision/",
     "en": {"user_input": "Where is the Eiffel Tower located?",
            "response": "The Eiffel Tower is located in Paris.",
            "retrieved_contexts": ["The Brandenburg Gate is located in Berlin.", "The Eiffel Tower is located in Paris."]},
     "zh": {"user_input": "埃菲尔铁塔位于哪里？",
            "response": "埃菲尔铁塔位于巴黎。",
            "retrieved_contexts": ["勃兰登堡门位于柏林。", "埃菲尔铁塔位于巴黎。"]}},

    # ---- Context Recall（上下文召回率）：标准答案里的断言，有几成能在检索块里找到 ----
    {"id": "R1", "metric": "context_recall", "official": "1.0",
     "note": "官方例：检索块只说了巴黎是首都", "src": _DOC + "context_recall/",
     "en": {"user_input": "Where is the Eiffel Tower located?",
            "retrieved_contexts": ["Paris is the capital of France."],
            "reference": "The Eiffel Tower is located in Paris."},
     "zh": {"user_input": "埃菲尔铁塔位于哪里？",
            "retrieved_contexts": ["巴黎是法国的首都。"],
            "reference": "埃菲尔铁塔位于巴黎。"}},

    # ---- Answer Relevancy（回答相关性）：从回答反推问题，和原问题有多像 ----
    {"id": "A1", "metric": "answer_relevancy", "official": "低",
     "note": "官方低分例：只答了一半（没说首都）", "src": _DOC + "answer_relevance/",
     "en": {"user_input": "Where is France and what is it's capital?",
            "response": "France is in western Europe."},
     "zh": {"user_input": "法国在哪里？它的首都是哪里？",
            "response": "法国位于西欧。"}},
    {"id": "A2", "metric": "answer_relevancy", "official": "高",
     "note": "官方高分例：两问都答了", "src": _DOC + "answer_relevance/",
     "en": {"user_input": "Where is France and what is it's capital?",
            "response": "France is in western Europe and Paris is its capital."},
     "zh": {"user_input": "法国在哪里？它的首都是哪里？",
            "response": "法国位于西欧，巴黎是它的首都。"}},
]


def build_judge():
    """[了解] 造一个 Qwen 裁判。返回 ragas 能直接用的 LLM 对象。"""
    # 1. [了解] 用 OpenAI 的 SDK，但把地址指向百炼的兼容端点。
    #    DASHSCOPE_API_KEY 是阿里云官方文档统一使用的环境变量名，写在 stage2/.env 里。
    client = AsyncOpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=JUDGE_BASE_URL)
    # 2. [核心] llm_factory 把客户端包装成"会按固定格式（JSON）回答"的裁判。
    #    ragas 内部要求裁判输出结构化结果（比如每条断言的 verdict=0/1），靠的就是这层包装。
    #    ragas 默认 temperature=0.01，让同一问题的判决尽量稳定。
    return llm_factory(JUDGE_MODEL, client=client)


def build_metrics(llm, embed_model: str) -> dict:
    """[核心] 造出四个指标（外加 P2 用的变体）。看清每个指标需要哪些零件。"""
    # 1. [核心] 只有 Answer Relevancy 需要 embedding：它要比较"反推出的问题"和"原问题"的向量距离。
    #    normalize_embeddings=True 让向量长度为 1，余弦相似度 = 内积（bge 官方要求这样用）。
    embeddings = HuggingFaceEmbeddings(model=embed_model, normalize_embeddings=True)
    # 2. [核心] 其余指标只要裁判。strictness=3 表示从回答反推 3 个问题再取平均（官方默认值）。
    return {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_utilization": ContextUtilization(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3),
    }


async def score_with_retry(metric, sample: dict, tries: int = 5) -> float:
    """[脚手架] 调一次 ascore；遇到 503（服务繁忙）或 429（请求太快）就等一会儿重试。"""
    for attempt in range(1, tries + 1):
        try:
            # 1. [核心] 所有指标的调用方式都一样：把样本的字段当关键字参数传进去。
            #    ascore 返回 MetricResult，分数在 .value 里。
            result = await metric.ascore(**sample)
            return result.value
        except Exception as e:  # [脚手架] ragas 把 HTTP 错误包在 InstructorRetryException 里抛出
            if not any(code in str(e) for code in ("503", "429")) or attempt == tries:
                raise
            # 2. [脚手架] 退避等待：第 1 次等 10 秒，第 2 次等 20 秒……
            print(f"    （裁判服务繁忙，{10 * attempt} 秒后重试）")
            await asyncio.sleep(10 * attempt)


async def show_faithfulness_steps(metric: Faithfulness, sample: dict) -> None:
    """[核心] 把忠实度的两步拆开跑，打印裁判拆出的断言和逐条判决。"""
    # 1. [了解] 这两个方法名以下划线开头，是 ragas 的内部步骤，版本升级可能改名；
    #    这里调用它们只为了"看见"分数是怎么来的，正式打分仍然用 ascore。
    statements = await metric._create_statements(sample["user_input"], sample["response"])
    # 2. [核心] 第二步把所有检索块拼成一段，让裁判逐条判断：这条断言能不能从上下文推出来。
    verdicts = await metric._create_verdicts(statements, "\n".join(sample["retrieved_contexts"]))
    for v in verdicts.statements:
        mark = "✔ 支持" if v.verdict else "✘ 不支持"
        print(f"    [{mark}] {v.statement}")
        print(f"             理由：{v.reason}")


async def run_group(lang: str) -> dict[str, float]:
    """[了解] 跑一组（英文或中文）：所有官方例子逐条打分，返回 {例子id: 分数}。"""
    # 1. [了解] 两组共用同一个裁判模型，只换 embedding 模型和例子的语言。
    llm = build_judge()
    metrics = build_metrics(llm, EMBED_MODELS[lang])
    scores = {}
    # 2. [了解] 逐条顺序跑，不并发——免费试用额度下，并发更容易触发限流。
    for ex in EXAMPLES:
        scores[ex["id"]] = await score_with_retry(metrics[ex["metric"]], ex[lang])
        print(f"  {ex['id']} {ex['metric']:<20} {scores[ex['id']]:.3f}")
    # 3. [核心] 对两个忠实度例子，额外打印中间过程：F2 是官方拆解过的，F3 是分数最可能不一致的。
    for ex_id in ("F2", "F3"):
        ex = next(e for e in EXAMPLES if e["id"] == ex_id)
        print(f"\n  【{ex_id} 拆解】回答：{ex[lang]['response']}")
        await show_faithfulness_steps(metrics["faithfulness"], ex[lang])
    return scores


def print_table(en: dict, zh: dict) -> None:
    """[脚手架] 官方分数 / 英文组 / 中文组 三列并排。"""
    print(f"\n{'=' * 78}\n{'例子':<5}{'指标':<21}{'官方':<14}{'英文组':>8}{'中文组':>8}   说明")
    for ex in EXAMPLES:
        print(f"{ex['id']:<6}{ex['metric']:<21}{ex['official']:<15}"
              f"{en[ex['id']]:>8.3f}{zh[ex['id']]:>9.3f}   {ex['note']}")


async def main() -> None:
    print(f"裁判：{JUDGE_MODEL}\n\n=== 英文组（官方原文 + {EMBED_MODELS['en']}）===")
    en = await run_group("en")
    print(f"\n=== 中文组（忠实译本 + {EMBED_MODELS['zh']}）===")
    zh = await run_group("zh")
    print_table(en, zh)


if __name__ == "__main__":
    asyncio.run(main())
