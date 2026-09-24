"""
common/corpus.py —— 三条检索实验线共用的"数据准备 + 评估"底座

为什么单独抽出来：
    抽 PDF、拆小节、切块、建向量库、算名次，这几步和"用哪种检索技术"完全无关。
    HyDE、Multi-Query、混合检索、重排都要先做这几步，所以放在这里各课共用。
    各课自己的文件里只留该技术本身的代码。

怎么运行（重要）：
    这些文件分散在 stage2 的几个子文件夹里，互相 import。
    Python 只会把"启动脚本所在的目录"加进搜索路径，所以
        uv run query_transformation/hyde_01_manual.py     ← 会报 ModuleNotFoundError: common
    必须在 stage2 目录下用 -m 以"模块"的方式启动，此时当前目录才会进搜索路径：
        uv run python -m query_transformation.hyde_01_manual
    这也是文件夹用下划线而不是连字符的原因：模块名里不允许出现连字符。

函数索引
    load_pdf_text()       —— 用 pypdf 抽出全文，并打印抽取检查
    split_into_sections() —— 按 §NN 标题把全文拆成小节 Document
    split_into_chunks()   —— 小节再切成块，每块继承所属小节的 metadata
    build_vectorstore()   —— 切块 + 向量化 + 存入 Chroma
    search_by_question()  —— 基准检索：原问题直接做向量检索
    gold_rank()           —— 标准答案小节第一次出现在第几名
    recall_at_k()         —— 前 k 条命中标准答案小节的块数 / 该小节总块数

阅读地图
    必读：split_into_chunks() / build_vectorstore() —— 后面每一课的数据都由这里产出
    扫读：load_pdf_text() / split_into_sections()   —— 一次写好、不再改动的数据清洗
          gold_rank() / recall_at_k()               —— 评估口径，看懂输出表格要靠它
    跳过：文件顶部的路径常量                         —— 只是拼路径
"""

# [脚手架] 标准库：re 用来匹配小节标题，Path 用来拼路径。
import re
from pathlib import Path

# [了解] 各个包的角色：
#        dotenv          —— 从 .env 读 DEEPSEEK_API_KEY（本文件不调用大模型，但各课普遍要用）
#        pypdf           —— 解析 PDF，抽文字
#        Document        —— LangChain 的文档对象：page_content（正文）+ metadata（元数据字典）
#        RecursiveCharacterTextSplitter —— 递归字符切分器，优先在段落、句子边界断开
#        HuggingFaceEmbeddings —— 把本地 sentence-transformers 模型包装成 LangChain 的向量化接口
#        Chroma          —— 向量库，这里只放在内存里，程序结束就消失
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# [脚手架] 本文件在 stage2/common/ 下，所以 parent.parent 才是项目根目录 stage2/。
#          语料和 .env 都放在 stage2/ 根目录，各课共用同一份。
STAGE2_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = STAGE2_DIR / "data" / "rag_handbook.pdf"
load_dotenv(STAGE2_DIR / ".env")

# [了解] 检索时每次取前 K 条。语料约 20 块，K=3 大约只看全部内容的 15%，差异才看得出来。
K = 3

# [核心] 基准测试集：问题故意用口语，和文档的书面术语错开。
#        gold 写的是小节编号而不是"第几块"——换了 chunk_size，标准答案依然有效。
#        predict 是"跑之前写下的预测"，跑完拿来对照，防止事后找理由。
#        注意 predict 的内容是对 HyDE 那一课写的；别的课要自己写预测，
#        不要直接沿用这里的判断（混合检索那一课会另起一份题目表）。
QUESTIONS = [
    {"id": "Q1", "kind": "公开", "gold": "§05", "predict": "HyDE 赢",
     "q": "我搜商品型号，比如 X200-Pro，结果老给我返回 X210-Pro 这种差一点点的，怎么办？"},
    {"id": "Q2", "kind": "公开", "gold": "§07", "predict": "平手",
     "q": "第一轮捞出来几十条，顺序不太靠谱，怎么把最相关的几条挑到前面？"},
    {"id": "Q3", "kind": "私有", "gold": "§09", "predict": "平手（假答案会编错金额）",
     "q": "去上海出差住酒店，一晚最多能报多少？"},
    {"id": "Q4", "kind": "私有", "gold": "§12", "predict": "HyDE 输",
     "q": "青鸾挂了应该找谁？"},
]


def load_pdf_text(path: Path = PDF_PATH) -> str:
    """[了解] 用 pypdf 抽出全文，并打印抽取检查。"""
    # 1. [了解] PdfReader 解析 PDF 结构；extract_text() 靠嵌入字体里的"字形编号→汉字"对照表还原文字。
    reader = PdfReader(path)
    pages = [page.extract_text() for page in reader.pages]
    text = "\n".join(pages)

    # 2. [核心] 抽取检查：在做任何检索实验之前，先确认文字没坏。
    #    否则检索效果差时，你分不清是"技术不行"还是"PDF 抽坏了"。
    #    � 是 Unicode 的"替换字符"（显示成 �），出现它说明有字形没能还原成汉字。
    print(f"页数：{len(pages)}　总字符数：{len(text)}")
    print(f"乱码字符(\\ufffd)数量：{text.count(chr(0xFFFD))}")
    print(f"找到的小节标题数：{len(re.findall(r'§\s*\d{2}', text))}（应为 24）")
    # 3. [脚手架] 用 repr 打印开头，\n 会原样显示出来，能看到 PDF 在哪里断了行。
    print("开头 200 字：", repr(text[:200]))
    return text


def split_into_sections(text: str) -> list[Document]:
    """[了解] 按 §NN 标题把全文拆成小节，每节一个 Document，metadata 记录小节标题。"""
    # 1. [了解] 去掉"第一部分/第二部分/第三部分"这三行大标题，它们不属于任何小节。
    #    2026-09-24 语料加了第三部分（§15~§24），这里的字符类必须同步加"三"，
    #    否则那一行大标题会被当成正文留在 §14 里面。
    text = re.sub(r"第[一二三]部分[^\n]*\n?", "", text)

    # 2. [了解] 用"前瞻"正则 (?=...) 在每个 § 之前切开：切点不吃掉 § 本身，标题留在各自小节里。
    parts = re.split(r"(?=§\s*\d{2})", text)

    sections = []
    for part in parts:
        # 3. [了解] 第一行就是小节标题；开头那段没有标题的封面文字直接丢掉。
        match = re.match(r"§\s*(\d{2})\s*([^\n]*)", part)
        if not match:
            continue
        section_id = f"§{match.group(1)}"
        sections.append(Document(
            page_content=part.strip(),
            metadata={"section": section_id, "title": match.group(2).strip()},
        ))
    return sections


def split_into_chunks(sections: list[Document]) -> list[Document]:
    """[核心] 小节 → 块。单独抽成一个函数，是因为混合检索那一课的 BM25 需要拿到同一批块。"""
    # 1. [核心] 在小节内部切块：split_documents 会把小节的 metadata 复制给切出来的每一块，
    #    所以每块都知道自己属于哪一节。先按结构拆、再按长度切，块不会跨越两个小节。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "，", ""],  # [了解] 中文要显式加上中文标点，默认只认英文句号和空格。
    )
    return splitter.split_documents(sections)


def build_vectorstore(sections: list[Document]) -> tuple[Chroma, HuggingFaceEmbeddings, int]:
    """[了解] 小节 → 块 → 向量 → Chroma。返回向量库、向量化模型和块数。"""
    # 1. [了解] 切块交给上面的 split_into_chunks，两边用的一定是同一套参数。
    chunks = split_into_chunks(sections)

    # 2. [核心] normalize_embeddings=True 让每个向量长度为 1，这时余弦相似度 = 内积。
    #    bge 系列官方要求归一化后再比较。
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3. [了解] collection_configuration 把距离度量设为余弦。Chroma 返回的是"距离"（越小越像），
    #    余弦距离 = 1 - 余弦相似度。不设的话默认是 L2 欧氏距离，数值不直观。
    store = Chroma(
        collection_name="rag_demo",
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    store.add_documents(chunks)
    return store, embeddings, len(chunks)


def search_by_question(store: Chroma, question: str, k: int) -> list[tuple[Document, float]]:
    """[核心] 基准检索：原问题直接做向量检索，后面每一课都拿它当对照组。"""
    # 1. [核心] similarity_search_with_score 内部先用 embed_query 把问题变成向量，再找最近的 k 块。
    #    返回 [(Document, 距离), ...]，距离越小越相似。
    return store.similarity_search_with_score(question, k=k)


def gold_rank(results: list[tuple[Document, float]], gold: str) -> int | None:
    """[了解] 返回标准答案小节第一次出现的名次（从 1 开始），没出现返回 None。"""
    # 1. [了解] 逐条看结果的 metadata["section"]，找到第一块属于标准答案小节的。
    for rank, (doc, _) in enumerate(results, start=1):
        if doc.metadata["section"] == gold:
            return rank
    return None


def recall_at_k(results: list[tuple[Document, float]], gold: str, k: int) -> str:
    """[核心] 召回率：前 k 条里命中了标准答案小节的几块 / 该小节一共几块。"""
    # 1. [核心] 分母：标准答案小节一共被切成了几块。results 是全部块的排序，直接数全体即可。
    total = sum(1 for doc, _ in results if doc.metadata["section"] == gold)
    # 2. [核心] 分子：前 k 条里有几块属于标准答案小节。
    hit = sum(1 for doc, _ in results[:k] if doc.metadata["section"] == gold)
    # 3. [脚手架] 返回 "2/3" 这样的字符串，方便直接打印。
    return f"{hit}/{total}"
