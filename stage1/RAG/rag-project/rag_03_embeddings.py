"""
RAG Part 2 之一：Embedding（嵌入）原理观察

这个脚本不建索引、不做检索，只干一件事：
    把句子变成一串数字，然后用眼睛看这串数字到底"懂不懂语义"。

运行方式：
    uv run rag_03_embeddings.py

第一次运行会从 Hugging Face 下载模型（约 95 MB），之后走本地缓存不再下载。
如果下载卡住（国内网络），在命令前加环境变量走镜像站：
    HF_ENDPOINT=https://hf-mirror.com uv run rag_03_embeddings.py
"""

# ---------------------------------------------------------------------------
# 名词先解释清楚，再往下看代码
# ---------------------------------------------------------------------------
# **Embedding（嵌入 / 向量化）**：把一段文字变成一串固定长度的小数，比如 512 个小数。
#   这串小数叫「向量（vector）」。关键性质是：**意思相近的句子，向量在空间里也靠得近**。
#   所以"怎么退货"和"退货流程是什么"这两句一个字都不完全一样，向量却几乎重合。
#
# **维度（dimension）**：向量里有多少个数。上面例子里就是 512 维。
#   维度越高通常越准，但存储和计算也越贵。
#
# **Sentence Transformers**：一个专门做"句子级向量化"的开源 Python 库
#   （官网 https://sbert.net）。它把 Hugging Face 上成千上万个嵌入模型
#   包装成统一的 encode() 接口。LangChain 通过 langchain-huggingface 这个包
#   再包一层，变成 HuggingFaceEmbeddings 类。
#
# **Hugging Face**：全球最大的开源模型托管平台，可以理解成"模型界的 GitHub"。
#   代码里写的 "BAAI/bge-small-zh-v1.5" 就是平台上的模型名字，
#   格式是「组织名/模型名」，程序会自动下载并缓存到本地。

import numpy as np                      # 数值计算库，用来算向量之间的相似度。官网：https://numpy.org/
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# 模型选择
# ---------------------------------------------------------------------------
# BAAI/bge-small-zh-v1.5 —— 北京智源研究院（BAAI）开源的中文嵌入模型。
#   bge   = BAAI General Embedding
#   small = 小号版本（还有 base、large），约 95 MB，CPU 上跑得动
#   zh    = 中文
#   v1.5  = 版本号
# 维度 512，最长输入 512 token（超过会被静默截断——这就是上一课强调
# "chunk_size 要按 token 算"的原因）。
MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# BGE 中文模型在训练时对「查询」和「文档」用了不同的写法：
# 查询要在前面加一句固定的指令，文档则**不加任何前缀**。
# 这个前缀叫「查询指令（query instruction）」，模型卡上写死了这句话，一个字都不能改。
# 加错或者两边都加，检索质量会明显下降，而且不会报错——属于最难排查的那类问题。
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def build_embeddings() -> HuggingFaceEmbeddings:
    """创建嵌入模型对象。"""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        # model_kwargs 传给底层的 SentenceTransformer 构造函数。
        # device="cpu" 是显式声明用 CPU 跑；有独立显卡时改成 "cuda" 会快很多。
        model_kwargs={"device": "cuda"},
        # encode_kwargs 传给每次编码时的 encode() 调用，作用于**文档**。
        # normalize_embeddings=True 表示把向量「归一化」——
        # 「归一化（normalize）」的意思是把向量缩放到长度正好等于 1，方向不变。
        # 好处：长度统一之后，两个向量的「点积」在数值上就等于「余弦相似度」，
        # 省掉一次除法，也让不同长度的文本可以公平比较。
        # 不归一化会怎样？长文本的向量往往更长，光看点积会误判成"更相关"。
        encode_kwargs={"normalize_embeddings": True},
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    # **余弦相似度（cosine similarity）**：衡量两个向量「方向」有多接近的指标。
    # 取值范围 -1 到 1：1 表示方向完全一致（语义最像），0 表示垂直（无关），
    # 负数表示方向相反。它只看方向、不看长度，所以不受文本长短影响。
    #
    # 公式是「点积 ÷ (模长 a × 模长 b)」。
    # np.dot 算点积（对应位置相乘再求和），np.linalg.norm 算模长（也就是向量的长度）。
    # 因为上面已经归一化过，分母其实都是 1，这里仍然写完整是为了让公式本身看得清楚。
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def char_overlap(a: str, b: str) -> float:
    """计算两句话的字面重合率，用来和语义相似度做对照。"""
    # set(a) 把字符串拆成「不重复字符的集合」。
    # & 求交集（两边都有的字），| 求并集（两边合起来的字）。
    # 交集大小 ÷ 并集大小 就是 Jaccard 相似度，是最朴素的"字面像不像"的度量。
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# 第 1 步：看看一句话变成向量之后长什么样
# ===========================================================================
def show_one_vector(emb: HuggingFaceEmbeddings) -> None:
    banner("第 1 步：一句话 -> 一串数字")

    text = "检索增强生成通过外部知识库缓解大模型的幻觉问题。"

    # embed_query 把**一句查询**变成向量，返回一个 Python 的 float 列表。
    # 对应的还有 embed_documents(list[str])，一次处理多段文本，内部会批量计算，比逐条快得多。
    vec = emb.embed_query(text)

    print(f"原文：{text}")
    print(f"向量类型：{type(vec).__name__}，长度（也就是维度）：{len(vec)}")
    # 只打印前 8 个数字，512 个全打出来没法看。
    print(f"前 8 维：{[round(v, 4) for v in vec[:8]]}")

    arr = np.array(vec)
    # 模长 = 向量的长度。因为设了 normalize_embeddings=True，这里应该是 1.0。
    print(f"向量模长：{np.linalg.norm(arr):.6f}  <- 归一化之后应该等于 1")
    print(f"数值范围：{arr.min():.4f} ~ {arr.max():.4f}")
    print("\n这 512 个数字没有任何一个是人能解释的——不存在'第 3 维代表褒贬'这种事。")
    print("它们唯一的意义体现在'和别的向量比'的时候，这就是下一步要做的。")


# ===========================================================================
# 第 2 步：语义相似度 vs 字面重合率
# ===========================================================================
def compare_pairs(emb: HuggingFaceEmbeddings) -> None:
    banner("第 2 步：向量真的懂语义吗？—— 和字面匹配做对照")

    # 四组句子，每组都是精心挑的：
    pairs = [
        # 同义改写：一个字都不重复，但意思完全一样
        ("怎么申请退货", "退货流程是什么样的", "同义改写"),
        # 相关但不同：退货和退款是两件事，应该"比较像"但不该"几乎一样"
        ("怎么申请退货", "退款多久能到账", "相关话题"),
        # 完全无关
        ("怎么申请退货", "今天北京的天气不错", "完全无关"),
        # 字面高度重合、语义完全不同：这是关键词检索最容易翻车的地方
        ("苹果发布了新手机", "苹果是一种常见的水果", "字面像/语义不同"),
    ]

    print(f"{'语义相似度':>10}  {'字面重合率':>10}   说明")
    print("-" * 74)
    for a, b, label in pairs:
        # 这里两句都用 embed_query，是为了公平比较（两边用同一种编码方式）。
        va = np.array(emb.embed_query(a))
        vb = np.array(emb.embed_query(b))
        print(f"{cosine(va, vb):>10.4f}  {char_overlap(a, b):>10.4f}   {label}")
        print(f"{'':>22}   A: {a}")
        print(f"{'':>22}   B: {b}")
        print("-" * 74)

    print("\n看两列数字的分歧：")
    print("  · 「同义改写」那组字面重合率很低，语义相似度却很高 —— 关键词检索会漏掉，向量检索能召回。")
    print("  · 「字面像/语义不同」那组正好相反 —— 关键词检索会误召回，向量检索能分开。")
    print("这两种分歧就是 RAG 用向量检索而不是用 SQL 的 LIKE 的全部理由。")


# ===========================================================================
# 第 3 步：查询指令前缀加不加，差别有多大
# ===========================================================================
def show_query_instruction(emb: HuggingFaceEmbeddings) -> None:
    banner("第 3 步：BGE 的查询指令前缀（用错了不会报错，只会变差）")

    query = "怎么申请退货"
    # 三段候选文档，第一段才是正确答案
    docs = [
        "退货说明：商品签收后 7 天内可无理由退货，请在订单详情页点击申请退货按钮提交。",
        "配送说明：下单后 48 小时内发货，偏远地区可能延长至 5 个工作日。",
        "发票说明：电子发票在确认收货后 3 个工作日内开具，可在个人中心下载。",
    ]

    # embed_documents 接收一个字符串列表，一次编码多段文本，返回向量列表。
    # 注意：文档端**不加**任何前缀，这是 BGE 模型卡明确要求的。
    doc_vecs = [np.array(v) for v in emb.embed_documents(docs)]

    gaps = {}  # 存两种写法各自的「区分度」，最后拿来对比
    for use_instruction in (False, True):
        # 加前缀就是简单的字符串拼接，没有任何魔法。
        q = (BGE_QUERY_INSTRUCTION + query) if use_instruction else query
        qv = np.array(emb.embed_query(q))
        scores = [cosine(qv, dv) for dv in doc_vecs]

        label = "加了查询指令前缀" if use_instruction else "没加前缀"
        print(f"\n【{label}】查询：{q}")
        for i, s in enumerate(scores):
            # 用 ← 标出得分最高的那一条
            mark = "  <- 得分最高" if s == max(scores) else ""
            print(f"    文档{i} 相似度 {s:.4f}{mark}  {docs[i][:24]}...")
        # 「区分度」= 正确答案的分 - 第二名的分。这个差值越大，检索越稳，
        # 越不容易因为一点噪声就把错误答案排到前面。
        best, second = sorted(scores, reverse=True)[:2]
        gaps[label] = best - second
        print(f"    区分度（第一名 - 第二名）：{best - second:.4f}")

    # 不预设结论，让数据自己说话。
    winner = max(gaps, key=gaps.get)
    print(f"\n本次结果：区分度更大的是【{winner}】"
          f"（{gaps['没加前缀']:.4f} vs {gaps['加了查询指令前缀']:.4f}）")
    print("""
怎么解读这个结果：
  · 两种写法都把正确答案排到了第一，所以「加不加前缀」在这个玩具例子上没有决定性影响。
  · 模型卡推荐加前缀的场景是「短查询 → 长文档」的检索，而且是在几万条文档的规模上
    统计出来的平均收益；三条文档的样本量根本测不出来这种差异，出现反向结果很正常。
  · 真正要记住的是硬规则：**前缀只加在查询上，文档端绝对不能加**。两边都加或者只给
    文档加，等于查询和文档用了两套不同的编码方式，那才是会明显掉点的错误。
  · 至于前缀在你自己的数据上到底有没有用，唯一的办法是拿你自己的问题和文档跑一遍评估——
    这也是官方文档反复强调"照搬排行榜不如自己测一遍"的原因。""")


# ===========================================================================
# 第 4 步：不归一化会怎样
# ===========================================================================
def show_normalization(emb: HuggingFaceEmbeddings) -> None:
    banner("第 4 步：normalize_embeddings 到底改变了什么（有个坑）")

    # 先看一眼模型内部长什么样。
    # SentenceTransformer 对象本质上是一条「流水线（pipeline）」：文本依次流过几个模块，
    # 每个模块做一步转换。named_children() 是 PyTorch 提供的方法，用来列出这些子模块。
    # HuggingFaceEmbeddings 把真正的 SentenceTransformer 对象存在 ._client 属性里
    # （带下划线前缀在 Python 里约定表示「内部属性」，官方不保证跨版本稳定，
    #   所以生产代码别依赖它，这里只是为了教学观察）。
    print("模型内部的模块流水线：")
    for name, module in emb._client.named_children():
        print(f"    [{name}] {type(module).__name__}")

    # 关键点：这个模型的最后一个模块就叫 Normalize —— 归一化被写死在模型自己的配置里了。
    # 所以对 BGE 系列来说，encode_kwargs={"normalize_embeddings": False} 是**无效**的，
    # 输出照样是单位长度。下面这两行验证这一点。
    raw = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},  # 显式关掉归一化
    )
    v_raw = np.array(raw.embed_query("退货"))
    print(f"\n把 normalize_embeddings 设成 False 之后，向量模长仍然是 "
          f"{np.linalg.norm(v_raw):.6f}")
    print("原因就是上面那个 Normalize 模块——它在模型内部，参数关不掉它。")
    print("这个坑值得记住：换成没有 Normalize 模块的模型时，同样的代码行为就变了，")
    print("所以正确做法是**永远显式传 normalize_embeddings=True**，不要依赖模型的默认配置。")

    # ---- 用一对人造向量把「归一化到底解决什么问题」演示清楚 -------------------
    # 这里不再用真实模型输出（它已经被归一化了），改用手工造的向量，
    # 这样才能看到"没归一化"时会发生什么。
    a = np.array([3.0, 4.0])          # 模长 = 5
    b = np.array([6.0, 8.0])          # 模长 = 10，方向和 a 完全相同
    c = np.array([-4.0, 3.0])         # 模长 = 5，方向和 a 垂直

    print("\n人造例子（二维，方便心算）：")
    print(f"    a = {a}  模长 {np.linalg.norm(a):.1f}")
    print(f"    b = {b}  模长 {np.linalg.norm(b):.1f}  方向和 a 完全一样，只是长了一倍")
    print(f"    c = {c}  模长 {np.linalg.norm(c):.1f}  方向和 a 垂直，也就是完全无关")
    print(f"\n    没归一化时的点积：a·b = {np.dot(a, b):.1f}，a·c = {np.dot(a, c):.1f}")
    print(f"    余弦相似度      ：a,b = {cosine(a, b):.4f}，a,c = {cosine(a, c):.4f}")
    print("\n    a 和 b 明明是同一个方向（语义完全一致），点积却是 50；")
    print("    如果换一对更长的向量，点积能轻松超过 100 —— 这个数字没有上限，无法互相比较。")
    print("    余弦相似度把模长除掉了，所以 a,b 是干干净净的 1.0（完全相同），a,c 是 0（无关）。")

    print("\n工程意义（下一个脚本的伏笔）：")
    print("    FAISS 的内积索引 IndexFlatIP 只会算点积，它不会替你除模长。")
    print("    只有先把向量归一化，IndexFlatIP 算出来的点积才等价于余弦相似度。")


def main() -> None:
    print("正在加载模型（首次运行需要下载约 95 MB，请耐心等待）...")
    emb = build_embeddings()
    print(f"模型加载完成：{MODEL_NAME}")

    show_one_vector(emb)
    compare_pairs(emb)
    show_query_instruction(emb)
    show_normalization(emb)

    banner("全部完成")


if __name__ == "__main__":
    main()
