"""切分层 —— 把 data/ 目录下的文档变成一个个可以入库的「块」。

**块（chunk）**：一段几十到几百字的文本。为什么不整篇文档入库？因为检索的目标是
「找出跟问题相关的那几段」，整篇文档里 95% 的内容和当前问题无关，一起塞给模型
既贵又会干扰它。切分的质量直接决定检索的上限——切坏了，后面再好的模型也救不回来。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    load_documents(data_dir)
        吃一个目录路径 -> 吐 list[Document]，一个文件一条（PDF 是一页一条）。
        被 app/ingest.py 调用。
    _load_markdown(path)
        吃一个 .md 文件路径 -> 吐 [Document]，整篇一条。被 load_documents 调用。
    _load_pdf(path)
        吃一个 .pdf 文件路径 -> 吐 list[Document]，一页一条。被 load_documents 调用。
    split_documents(docs)
        吃 list[Document]（整篇的）-> 吐 list[Document]（切碎的小块）。
        被 app/ingest.py 调用。
    stable_id(chunk)
        吃一个块 -> 吐一个字符串 id，如 "after-sales-policy.md#售后服务政策/一、七天无理由退货/退货运费#0"。
        被 app/ingest.py 调用。

阅读地图
    必读：split_documents（两级切分为什么要两级）、stable_id（什么叫「稳定」）
    扫读：load_documents 的分派逻辑
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

from app.core.logging import get_logger

logger = get_logger(__name__)

# [了解] 切分参数。中文文档一个汉字约等于 1~2 个 token（token 是模型看到的最小文字单位，
#        API 计费和上下文长度都按它算），400 字符大约对应半屏到一屏的内容。
CHUNK_SIZE = 400
# [核心] 相邻两块之间重叠的字符数。为什么要重叠？防止一句完整的话被切断在两块交界处，
#        导致哪一块都答不全。代价是同样的内容被存了两份，库会稍微变大。
CHUNK_OVERLAP = 80
# [了解] 短于这个长度的块直接丢弃。这类块通常是孤零零的一行标题，
#        没有实质信息，却会占掉检索结果的名额。
MIN_CHUNK_CHARS = 30


# ===========================================================================
# 一、加载：文件 -> Document
# ===========================================================================
# **Document** 是 LangChain 里表示「一段文本 + 它的附属信息」的对象，只有两个要点：
#   doc.page_content  str，正文
#   doc.metadata      dict，你想记住的任何信息，比如它来自哪个文件、第几页
# 后面所有环节（切分、入库、检索）传来传去的都是它。


def _load_markdown(path: Path) -> list[Document]:
    """读一个 Markdown 文件，整篇作为一条 Document 返回。"""
    # 步骤 1：把文件内容读成字符串。encoding="utf-8" 必须显式写，
    #         否则 Windows 上默认按 GBK 解码，中文会乱码或直接抛 UnicodeDecodeError。
    text = path.read_text(encoding="utf-8")

    # 步骤 2：包成 Document，并在 metadata 里记下它来自哪个文件。
    #         source 这个字段名是 LangChain 生态的惯例，后面答案要用它标注出处。
    return [Document(page_content=text, metadata={"source": path.name})]


def _load_pdf(path: Path) -> list[Document]:
    """读一个 PDF，每一页作为一条 Document 返回。"""
    # 步骤 1：打开 PDF。PdfReader 来自 pypdf 库，它把 PDF 解析成一个页对象列表。
    #         文档：https://pypdf.readthedocs.io/
    reader = PdfReader(str(path))

    docs: list[Document] = []
    # 步骤 2：逐页抽取文字。enumerate(..., start=1) 让页码从 1 开始数，符合人的习惯。
    for page_no, page in enumerate(reader.pages, start=1):
        # [核心] extract_text() 对**扫描版 PDF（本质是图片）会返回空字符串，且不报错**。
        #        这是 RAG 里最典型的静默失败：整本书入库成功，一个字都没进去。
        #        所以这里 or "" 兜底，并在下一步把空页记进日志。
        text = page.extract_text() or ""
        if not text.strip():
            logger.warning("PDF 页面抽不出文字", extra={"file": path.name, "page": page_no})
            continue
        # 步骤 3：每页一条 Document，metadata 里多记一个页码，答案就能精确到页。
        docs.append(Document(page_content=text, metadata={"source": path.name, "page": page_no}))
    return docs


def load_documents(data_dir: Path) -> list[Document]:
    """扫描目录，把里面的 .md 和 .pdf 全部读成 Document 列表。"""
    docs: list[Document] = []

    # 步骤 1：列出目录下所有文件。sorted() 保证每次运行的顺序一致——
    #         顺序不稳定会让后面的调试变得很痛苦（同样的输入，两次结果排列不同）。
    for path in sorted(data_dir.iterdir()):
        # 步骤 2：跳过子目录和隐藏文件（.gitkeep 之类）。
        if not path.is_file() or path.name.startswith("."):
            continue

        # 步骤 3：按扩展名分派给对应的加载函数。.suffix 是 pathlib 提供的属性，
        #         值形如 ".md"；.lower() 是为了兼容 ".MD" 这种写法。
        suffix = path.suffix.lower()
        if suffix == ".md":
            docs.extend(_load_markdown(path))
        elif suffix == ".pdf":
            docs.extend(_load_pdf(path))
        else:
            # 步骤 4：不认识的格式记一条日志。不要静默忽略——
            #         用户把 .docx 丢进 data/ 却发现问不出内容时，这条日志是唯一线索。
            logger.warning("跳过不支持的文件类型", extra={"file": path.name})

    logger.info("文档加载完成", extra={"files": len(docs)})
    return docs


# ===========================================================================
# 二、切分：整篇 Document -> 一堆小块 Document
# ===========================================================================
def split_documents(docs: list[Document]) -> list[Document]:
    """两级切分：先按 Markdown 标题切，再按长度切。"""
    # ---- 练习 1：实现两级切分 ---------------------------------------------
    # [核心] 为什么要两级：
    #   第一级按标题切，好处是每一块天然对应文档里的一个小节，
    #       并且能把「它属于哪一章哪一节」记进 metadata，答案就能标出章节出处。
    #       坏处是某些小节可能很长（几千字），超出模型能舒服处理的长度。
    #   第二级按长度切，把过长的小节再切成 400 字左右的块，并保留章节 metadata。
    #
    # [核心] 要求写四个步骤：
    #   步骤 1  用 MarkdownHeaderTextSplitter 按 #/##/### 三级标题切。
    #           它的构造参数：headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")]，
    #           strip_headers=False（保留标题文字在正文里，让块自带上下文）。
    #           注意它的方法是 split_text(字符串)，不是 split_documents。
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers = False,
    )
    header_chunks: list[Document] = []
    for doc in docs:
        pieces = header_splitter.split_text(doc.page_content)
    #   步骤 2  把上游的 metadata（source）合并回每一块。
    #           **这一步不能省**：MarkdownHeaderTextSplitter 是唯一「吃字符串、吐 Document」
    #           的切分器，它自己新造 metadata，上游的 source 不会自动带过来。
    #           合并写法：块.metadata = 原文档.metadata | 块.metadata
        for piece in pieces:
            piece.metadata = doc.metadata | piece.metadata
            header_chunks.append(piece)
    #   步骤 3  用 RecursiveCharacterTextSplitter 按长度再切一次。
    #           构造参数：separators=["\n\n","\n","。","！","？","；","，"," ",""]，
    #           chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True。
    #           它的方法是 split_documents(列表)，会自动继承 metadata。
    char_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        add_start_index = True,
    )
    chunks = char_splitter.split_documents(header_chunks)
    #   步骤 4  丢掉长度小于 MIN_CHUNK_CHARS 的块，并打一条日志记录最终块数。
    chunks = [chunk for chunk in chunks if len(chunk.page_content) >= MIN_CHUNK_CHARS]
    logger.info("切分完成", extra = {"chunks": len(chunks)})
    return chunks
    # -----------------------------------------------------------------------


# ===========================================================================
# 三、稳定 id
# ===========================================================================
def stable_id(chunk: Document) -> str:
    """给一个块生成永远不变的 id。"""
    # ---- 练习 2：拼出稳定 id ----------------------------------------------
    # [核心] 「稳定」的意思是：同一段文本，无论今天跑还是下个月跑，都得到完全相同的 id。
    #        有了它，重复入库就变成「覆盖同一条记录」而不是「插入一条新记录」，
    #        知识库才能增量更新——文档改了就重跑一次，不会堆出一堆重复内容。
    #
    #        所以不能用：随机 uuid（每次都不同）、时间戳（每次都不同）、
    #        列表下标（切分参数一改就整体错位）。
    #
    # [核心] 要求：拼成 "文件名#章节路径#起始偏移" 三段，中间用 # 连接。
    #        章节路径 = h1/h2/h3 三个 metadata 用 "/" 连起来。
    #        为什么必须带章节路径：start_index 是**相对于传给切分器的那一小节**的偏移，
    #        每个小节的第一块 start_index 都是 0，只用 "文件名#0" 会撞车。
    #        取 metadata 用 .get(键, "") 而不是 [键]，因为不是每块都有 h3（有的小节只到二级标题）。
    meta = chunk.metadata
    section = "/".join(meta.get(k, "") for k in ["h1", "h2", "h3"])
    return f"{meta['source']}#{section}#{meta.get('start_index', 0)}"
    # -----------------------------------------------------------------------
