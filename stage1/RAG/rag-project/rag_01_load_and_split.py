"""
RAG Part 1：文档加载（Load）与文本切分（Split）

对应 RAG 索引阶段的前两步：
    原始文件(PDF/Markdown/...) --加载--> Document 列表 --切分--> 更小的 Document 块 --> (下一步才是 embedding/向量库)

运行方式：
    uv run rag_01_load_and_split.py

依赖（已写进 pyproject.toml）：
    langchain-text-splitters   —— LangChain 官方的切分器包，独立于 langchain 主包单独发布
    pypdf                      —— 纯 Python 的 PDF 解析库，LangChain v1 官方教程现在直接用它读 PDF
    tiktoken                   —— OpenAI 开源的分词器，用来「按 token 数」而不是「按字符数」量长度
"""

# ---------------------------------------------------------------------------
# 标准库导入
# ---------------------------------------------------------------------------

# pathlib 是 Python 标准库里处理「文件路径」的模块。
# 它把路径包装成对象，可以用 / 运算符拼接子路径，比手写字符串拼接更安全（自动处理 Windows 的反斜杠）。
# 官方文档：https://docs.python.org/3/library/pathlib.html
from pathlib import Path

# statistics 是标准库里的基础统计模块，这里只用它的 median()（中位数）。
# 官方文档：https://docs.python.org/3/library/statistics.html
import statistics

# urllib.request 是标准库里的 HTTP 客户端。这里只用它下载一份示例 PDF，
# 避免把 2MB 的二进制文件提交进仓库。
# 官方文档：https://docs.python.org/3/library/urllib.request.html
import urllib.request

# ---------------------------------------------------------------------------
# 第三方库导入
# ---------------------------------------------------------------------------

# pypdf 是解析 PDF 的库：把一个 PDF 文件读成「页对象」的列表，每页可以调用 extract_text() 抽出纯文本。
# 官方文档：https://pypdf.readthedocs.io/
import pypdf

# Document 是 LangChain 里「一段文本 + 它的元数据」的统一容器，是加载和切分环节唯一的数据结构。
# 它只有三个属性：page_content(str，正文)、metadata(dict，任意附加信息)、id(可选的字符串标识)。
# 注意它来自 langchain_core（底层内核包），不是 langchain 主包。
from langchain_core.documents import Document

# 三个切分器，全部来自 langchain_text_splitters 这个独立发布的包：
#   RecursiveCharacterTextSplitter —— 官方推荐的通用切分器，按「段落→句子→词」逐级回退地切
#   CharacterTextSplitter          —— 最朴素的切分器，只认一个固定分隔符，这里用来做对照实验
#   MarkdownHeaderTextSplitter     —— 按 Markdown 标题层级切，并把标题写进 metadata
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# 贯穿全文的路径常量
# ---------------------------------------------------------------------------

# __file__ 是当前脚本文件的路径；.resolve() 把它变成绝对路径；.parent 取所在目录。
# 这样写的好处：不管你从哪个目录敲 uv run，脚本都能找到自己旁边的 data/ 文件夹。
# 如果直接写相对路径 "data/sample_zh.md"，就依赖你当前的工作目录，换个地方运行会找不到文件。
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # / 是 pathlib 重载过的运算符，等价于 os.path.join
MD_PATH = DATA_DIR / "sample_zh.md"
PDF_PATH = DATA_DIR / "nke-10k-2023.pdf"

# LangChain 官方教程用的示例 PDF：Nike 2023 年的 10-K 财报（107 页）。
PDF_URL = (
    "https://raw.githubusercontent.com/langchain-ai/langchain/v0.3/"
    "docs/docs/example_data/nke-10k-2023.pdf"
)


def banner(title: str) -> None:
    """打印一个分节标题，纯粹为了让终端输出好读，与 RAG 本身无关。"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def preview(text: str, n: int = 80) -> str:
    """把一段文本压成单行、截断到 n 个字符，方便在终端里预览块的内容。"""
    # 把换行替换成可见的两个字符 \n，否则一个块打印出来会占好几行，看不清块的边界在哪。
    # 这里没有用 ⏎ 之类的符号，是因为 Windows 终端默认编码是 GBK，打这类符号会直接抛 UnicodeEncodeError。
    one_line = text.replace("\n", "\\n")
    return one_line[:n] + ("..." if len(one_line) > n else "")


# ===========================================================================
# 第 1 步：加载 Markdown —— 「加载器」本质上只是「读文件 + 造 Document」
# ===========================================================================
def load_markdown() -> Document:
    """把一个 Markdown 文件读成一个 Document 对象。"""
    banner("第 1 步：加载 Markdown 文件")

    # read_text 以「文本模式」读文件。文本模式默认开启 universal newlines：
    # Windows 的 \r\n 会被自动翻译成 \n。这一点很关键——切分器内部是按 "\n" 找换行的，
    # 如果你改用二进制模式读再手动 decode，每行末尾会残留一个 \r，切出来的块里全是脏字符。
    # encoding 必须显式写 utf-8：Windows 上 Python 的默认编码可能是 GBK，读中文文件会直接抛 UnicodeDecodeError。
    raw_text = MD_PATH.read_text(encoding="utf-8")

    # 手动构造 Document。这就是「文档加载器（Document Loader）」做的全部事情：
    # 把某种格式的字节流变成 page_content 字符串，再挂上一份说明来源的 metadata。
    # metadata 里放什么完全由你决定，但 source（来源）是事实标准，后面展示引用出处要靠它。
    doc = Document(
        page_content=raw_text,
        metadata={"source": MD_PATH.name, "format": "markdown"},
    )

    print(f"文件：{MD_PATH.name}")
    print(f"字符数：{len(doc.page_content)}")
    print(f"metadata：{doc.metadata}")
    print(f"开头预览：{preview(doc.page_content, 60)}")
    return doc


# ===========================================================================
# 第 2 步：结构化切分 —— 按 Markdown 标题切，把章节标题存进 metadata
# ===========================================================================
def split_by_markdown_header(doc: Document) -> list[Document]:
    """用 MarkdownHeaderTextSplitter 按标题层级切分。"""
    banner("第 2 步：按 Markdown 标题结构切分")

    # headers_to_split_on 是「(标题前缀, 存进 metadata 时用的键名)」的列表。
    # 这里声明：遇到 "# " 记成 h1，"## " 记成 h2，"### " 记成 h3。
    # 前缀里不写空格，切分器内部会自己处理 "# 标题" 中间的那个空格。
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        # strip_headers=False 表示「标题行本身也保留在块的正文里」。
        # 默认值是 True（把标题从正文里剥掉，只留在 metadata 里）。
        # 做 RAG 时通常建议设成 False：块被单独召回后送进大模型时，正文里带着标题，
        # 模型才知道这段话在讲哪一节；否则模型只看到一段没有上下文的孤立文字。
        strip_headers=False,
    )

    # 注意这个方法叫 split_text，输入是「字符串」而不是 Document，但返回值是 list[Document]。
    # 它是唯一一个「吃字符串、吐 Document」的切分器，因为它必须自己创建 metadata（把标题塞进去）。
    chunks = splitter.split_text(doc.page_content)

    print(f"切出 {len(chunks)} 块（每块 = 一个标题层级下的内容）\n")
    for i, chunk in enumerate(chunks):
        print(f"[{i}] len={len(chunk.page_content):4d} metadata={chunk.metadata}")
        print(f"     {preview(chunk.page_content, 70)}")

    # 上一步 Document 的 metadata（source/format）不会自动继承过来，需要手动合并回去，
    # 否则后面就不知道这些块是从哪个文件来的了。
    for chunk in chunks:
        # dict 的 | 运算符（Python 3.9+）返回合并后的新字典，右边的键会覆盖左边的同名键。
        chunk.metadata = doc.metadata | chunk.metadata

    return chunks


# ===========================================================================
# 第 3 步：递归字符切分 —— 把过长的块二次切到可控大小
# ===========================================================================
def split_recursively(chunks: list[Document]) -> list[Document]:
    """对第 2 步的结果做二次切分，保证每块不超过 chunk_size。"""
    banner("第 3 步：RecursiveCharacterTextSplitter 二次切分（中文分隔符）")

    # separators 是「优先级从高到低」的分隔符列表。切分器的工作方式是：
    #   1. 先用第 0 个分隔符切；如果某个片段仍然超过 chunk_size，
    #   2. 就对这个片段改用第 1 个分隔符再切；如此递归，直到片段够小或者分隔符用完。
    # 这就是「Recursive（递归）」的含义：总是优先在语义边界（段落）上切，
    # 实在切不动了才退化到句子、再退化到单字。
    #
    # 默认的 separators 是 ["\n\n", "\n", " ", ""]，最后那个 "" 表示逐字符硬切。
    # 这套默认值是为英文设计的：英文用空格分词，所以 " " 这一级很有效。
    # 中文没有空格，一整段中文在默认配置下会直接从 "\n" 掉到 ""（逐字硬切），
    # 于是块的边界经常落在句子中间。下面这份列表补上了中文标点，让它能在句号、问号处断开。
    chinese_separators = [
        "\n\n",   # 空行 = 段落边界，语义最完整，优先在这里切
        "\n",     # 单个换行 = 行边界
        "。",     # 中文句号
        "！",
        "？",
        "；",     # 分号，句内的强停顿
        "，",     # 逗号，退而求其次
        " ",      # 空格，用于中英混排里的英文部分
        "",       # 兜底：逐字符硬切，保证一定能压到 chunk_size 以内
    ]

    splitter = RecursiveCharacterTextSplitter(
        # separators=chinese_separators,
        # chunk_size 的单位取决于 length_function，默认是 len，也就是「字符数」。
        # 这里故意设成 120（比第 2 步大多数标题块还小），是为了让二次切分真的发生、看得见效果；
        # 真实项目里中文常用 300~500 字符，英文常用 800~1000 字符。
        # 如果这里设成 200，本 demo 的标题块全都已经小于 200 了，你会看到 8 块进、8 块出，什么都没发生。
        chunk_size=120,
        # chunk_overlap：相邻两块之间重复的长度。作用是防止一句话正好被切开后，
        # 两边的块都读不懂。经验值是 chunk_size 的 10%~20%。
        # 设成 0 会怎样？边界处的语义就断了。设得过大会怎样？重复内容膨胀存储、也增加检索噪声。
        chunk_overlap=24,
        # add_start_index=True 会往 metadata 里加一个 start_index 字段，
        # 记录这一块在「传给切分器的那个 Document」里的字符偏移量。
        # 注意它不是相对于原始文件的偏移：这里传进来的是第 2 步切出的标题块，
        # 所以每个标题块内的第一个小块 start_index 都是 0。
        # 价值在于：召回后能定位回上一级文本做高亮，也能判断两个块是否相邻（做上下文扩展检索要用）。
        add_start_index=True,
        # keep_separator=True 是 RecursiveCharacterTextSplitter 的默认值，表示分隔符本身要保留。
        # 对中文尤其重要：如果丢掉分隔符，句号就没了，块读起来会变成没有标点的一长串。
        keep_separator=True,
    )

    # split_documents 吃 Document 列表、吐 Document 列表，并且会自动把原 Document 的
    # metadata 复制到每个切出来的小块上（这正是第 2 步末尾手动合并 metadata 的原因）。
    # 对照：split_text     吃字符串、吐字符串列表，metadata 全丢；
    #      create_documents 吃字符串列表、吐 Document 列表，metadata 要另外传参。
    small_chunks = splitter.split_documents(chunks)

    # ---- 自测 2：质量闸门（过滤 + 统计）----------------------------------
    # 「闸门（gate）」是工程里的说法，意思是「数据往下游走之前必须先通过的一道检查」，
    # 不合格的就在这里拦下来，不让它污染后面的向量库。
    #
    # 过滤必须新建一个列表，不能在 for 循环里对同一个列表做 remove。
    # 原因：Python 的 for 是靠「下标」往前走的。你删掉下标 3 的元素后，原来下标 4 的元素
    # 会顶上来变成下标 3，而循环下一轮直接看下标 4 —— 顶上来的那个元素就被跳过了。
    # 效果就是「连续的短块只会被删掉一半」。
    #
    # 下面这行叫「列表推导式（list comprehension）」，读法是：
    # 「对 small_chunks 里的每个 c，如果满足 if 条件，就把 c 收进新列表」。
    # 它生成的是一个全新的列表，原列表一个字节都没动，所以不存在上面那个跳元素的问题。
    MIN_CHUNK_CHARS = 20  # 阈值：短于这个长度的块直接丢弃
    kept = [c for c in small_chunks if len(c.page_content) >= MIN_CHUNK_CHARS]
    dropped = len(small_chunks) - len(kept)

    # 统计块长度的分布，用来判断参数设得合不合适。
    # statistics 是 Python 标准库里的基础统计模块，median() 就是求中位数。
    # 官方文档：https://docs.python.org/3/library/statistics.html
    # 为什么看中位数而不只看平均数？平均数会被一两个超长块拉高，
    # 中位数代表「一半的块比它长、一半比它短」，更能反映大多数块的真实大小。
    lengths = [len(c.page_content) for c in kept]

    print(f"二次切分后：{len(chunks)} 块 -> {len(small_chunks)} 块")
    print(f"质量闸门：丢弃 {dropped} 个短于 {MIN_CHUNK_CHARS} 字符的块，剩 {len(kept)} 块")
    print(
        f"块长度分布：最小 {min(lengths)} / 中位数 {statistics.median(lengths):.0f} "
        f"/ 最大 {max(lengths)} 字符\n"
    )

    small_chunks = kept  # 后面统一用过滤后的结果

    for i, chunk in enumerate(small_chunks[:6]):
        # 取最细一级的标题作为「这一块属于哪一节」的标识
        section = (
            chunk.metadata.get("h3")
            or chunk.metadata.get("h2")
            or chunk.metadata.get("h1")
        )
        print(
            f"[{i}] len={len(chunk.page_content):4d} "
            f"start={chunk.metadata['start_index']:4d} 章节={section}"
        )
        print(f"     {preview(chunk.page_content, 70)}")
    if len(small_chunks) > 6:
        print(f"... 其余 {len(small_chunks) - 6} 块略")

    return small_chunks


# ===========================================================================
# 第 4 步：对照实验 —— 递归切分 vs. 朴素字符切分
# ===========================================================================
def compare_splitters(text: str) -> None:
    """同样的文本、同样的 chunk_size，看看两种切分器的差别。"""
    banner("第 4 步：对照实验 —— 为什么推荐 Recursive 而不是 Character")

    # 先把示例文本压成「一整段没有空行的长文本」。
    # 这不是为了刁难谁：从 PDF 抽出来的文字、从数据库导出的字段，绝大多数就是这个样子——
    # 没有空行，甚至没有换行，只有一串连续的句子。
    # splitlines() 按行拆开；条件里过滤掉标题行和空行；再用 "" 拼回去 = 一整段。
    one_paragraph = "".join(
        line for line in text.splitlines() if line.strip() and not line.startswith("#")
    )
    print(f"构造出一段无空行的长文本，共 {len(one_paragraph)} 字符\n")

    # CharacterTextSplitter 只认一个固定的 separator（默认 "\n\n"）。
    # 它先按这个分隔符切，然后把相邻片段合并到接近 chunk_size 为止 —— 但如果某个片段
    # 本身就超过 chunk_size，它没有任何退路，只能整块吐出来（并打印一条 "created a chunk of size ..." 的警告）。
    naive = CharacterTextSplitter(separator="\n\n", chunk_size=200, chunk_overlap=0)
    naive_chunks = naive.split_text(one_paragraph)

    # 同样参数的递归切分器：切不动时会自动降级到下一级分隔符，所以不会超长。
    recursive = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "，", ""],
        chunk_size=200,
        chunk_overlap=0,
    )
    recursive_chunks = recursive.split_text(one_paragraph)

    # max(map(len, xs)) = 把 len 作用到每个元素上再取最大值，也就是「最长的块有多少字符」
    print(
        f"CharacterTextSplitter          : {len(naive_chunks):2d} 块，"
        f"最长 {max(map(len, naive_chunks))} 字符  <- chunk_size=200 完全没生效"
    )
    print(
        f"RecursiveCharacterTextSplitter : {len(recursive_chunks):2d} 块，"
        f"最长 {max(map(len, recursive_chunks))} 字符  <- 老老实实压在 200 以内"
    )
    print("\n结论：CharacterTextSplitter 找不到它那个唯一的分隔符时就彻底失效，chunk_size 形同虚设；")
    print("      Recursive 会逐级降级到句号、逗号乃至单字，在保证硬上限的同时尽量在语义边界切开。")


# ===========================================================================
# 第 5 步：按 token 计长 —— 因为限制你的是模型的 token 数，不是字符数
# ===========================================================================
def split_by_token(text: str) -> None:
    """用 tiktoken 把「长度」的单位从字符换成 token。"""
    banner("第 5 步：按 token 而不是按字符控制块大小")

    # tiktoken 是 OpenAI 开源的 BPE 分词器（把文本切成模型真正看到的最小单位 token）。
    # 嵌入模型和大模型的输入上限都是按 token 算的，而中文的「字符数 → token 数」
    # 换算比例和英文完全不同：英文大约 4 个字符 = 1 token，中文大约 1 个汉字 = 1~2 token。
    # 所以如果你按字符设 chunk_size=1000，中文块的真实 token 数可能是英文块的好几倍，
    # 直接撞上嵌入模型的最大输入长度被静默截断 —— 这是 RAG 里最隐蔽的一类数据丢失。
    #
    # from_tiktoken_encoder 是一个类方法（classmethod），返回一个已经把 length_function
    # 换成「tiktoken 编码后的 token 数」的切分器实例。
    # encoding_name="cl100k_base" 是 GPT-3.5/GPT-4 与 text-embedding-3-* 系列用的编码表。
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        separators=["\n\n", "\n", "。", "，", ""],
        chunk_size=100,      # 单位现在是 token，不是字符
        chunk_overlap=20,
    )
    token_chunks = token_splitter.split_text(text)

    # 拿同样的数字 100 做字符切分，对比一下真实 token 数差多少。
    char_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "，", ""], chunk_size=100, chunk_overlap=20
    )
    char_chunks = char_splitter.split_text(text)

    import tiktoken  # 局部导入：只有这一段用得到，放这里能让「它服务于哪段逻辑」一目了然

    enc = tiktoken.get_encoding("cl100k_base")  # 拿到 cl100k_base 这张编码表

    print(
        f"chunk_size=100（token 计长）: {len(token_chunks):2d} 块，"
        f"最长 {max(len(enc.encode(c)) for c in token_chunks):3d} token"
        f"  <- 保证不超 100 token"
    )
    print(
        f"chunk_size=100（字符计长） : {len(char_chunks):2d} 块，"
        f"最长 {max(len(enc.encode(c)) for c in char_chunks):3d} token"
        f"  <- 实际超了 100 token"
    )

    # 更关键的一点：同样按字符算，中文和英文的「字符 → token」换算比例天差地别。
    english = (
        "Retrieval-augmented generation combines a retriever and a generator. "
        "The retriever fetches relevant passages from an external corpus, and "
        "the generator conditions on those passages to produce a grounded answer."
    )
    zh_ratio = len(enc.encode(text)) / len(text)
    en_ratio = len(enc.encode(english)) / len(english)
    print(f"\n中文样本：{len(text)} 字符 -> {len(enc.encode(text))} token，"
          f"平均每字符 {zh_ratio:.2f} token")
    print(f"英文样本：{len(english)} 字符 -> {len(enc.encode(english))} token，"
          f"平均每字符 {en_ratio:.2f} token")
    print(f"比例相差 {zh_ratio / en_ratio:.1f} 倍")
    print("\n结论：一个写死的 chunk_size=1000（字符）在英文语料里约 250 token，在中文语料里可能超过 1000 token。")
    print("      嵌入模型的输入上限是按 token 算的，超了会被静默截断而不是报错——所以 chunk_size 要按 token 设。")


# ===========================================================================
# 第 6 步：加载 PDF —— 每页一个 Document，再递归切分
# ===========================================================================
def load_and_split_pdf() -> list[Document]:
    """用 pypdf 读 PDF，按页造 Document，然后递归切分。"""
    banner("第 6 步：加载并切分 PDF")

    # 首次运行时下载示例 PDF 并缓存到 data/ 下；已存在就跳过，避免重复下载。
    if not PDF_PATH.exists():
        print(f"首次运行，正在下载示例 PDF 到 {PDF_PATH} ...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)  # exist_ok=True：目录已存在时不报错
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
        print("下载完成。")

    # PdfReader 打开 PDF 并解析出页对象列表。
    reader = pypdf.PdfReader(PDF_PATH)

    # 逐页抽取文本，每页造一个 Document。
    # extract_text() 在整页是扫描图片、没有文字层时会返回空字符串（个别情况可能返回 None），
    # 所以用 `or ""` 兜底，防止后面 len() 抛 TypeError。
    # metadata 里存页码是刚需：用户看到答案时要能点回原文第几页。
    pdf_docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": PDF_PATH.name, "page": i},
        )
        for i, page in enumerate(reader.pages)  # enumerate 同时给出下标 i 和元素 page
    ]

    print(f"PDF 共 {len(pdf_docs)} 页 -> {len(pdf_docs)} 个 Document")
    print(f"第 0 页字符数：{len(pdf_docs[0].page_content)}")
    print(f"第 0 页预览：{preview(pdf_docs[0].page_content, 100)}")

    # 一页往往有两三千字符，作为检索单位太粗：一次召回会带回大量无关内容。
    # 所以还要按 chunk_size 二次切分。这里用默认分隔符（示例 PDF 是英文财报）。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 英文场景的常用起点
        chunk_overlap=200,    # 20% 重叠
        add_start_index=True,
    )
    splits = splitter.split_documents(pdf_docs)

    print(f"\n切分后：{len(splits)} 块")
    print(f"示例块 metadata：{splits[10].metadata}")
    print(f"示例块内容：{preview(splits[10].page_content, 100)}")

    # 统计有多少页是「抽不出文字」的空页 —— 真实 PDF 里扫描件、纯图表页很常见，
    # 这类页 pypdf 无能为力，需要换成带 OCR 的解析工具（Docling / MinerU / Unstructured）。
    empty_pages = sum(1 for d in pdf_docs if not d.page_content.strip())
    print(f"\n抽不出文字的空页数：{empty_pages} / {len(pdf_docs)}")

    return splits


def main() -> None:
    md_doc = load_markdown()
    header_chunks = split_by_markdown_header(md_doc)
    split_recursively(header_chunks)
    compare_splitters(md_doc.page_content)
    split_by_token(md_doc.page_content)
    load_and_split_pdf()

    banner("全部完成")


# 只有直接运行本文件时才执行 main()；被别的模块 import 时不执行。
if __name__ == "__main__":
    main()
