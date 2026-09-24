"""
自测 3：用 pypdf 体检一份真实的中文 PDF，判断它到底能不能用纯文本抽取搞定。

这个脚本不做任何"修复"，它只回答一个问题：
    这份 PDF 用 pypdf 读出来的质量如何？够不够格直接进 RAG 流水线？

运行方式：
    uv run rag_02_pdf_probe.py                 # 体检 data/ 目录下所有 PDF
    uv run rag_02_pdf_probe.py 某个文件.pdf     # 只体检指定的一份
"""

import re                      # 正则表达式模块，用来做字符统计。文档：https://docs.python.org/3/library/re.html
import statistics              # 标准库统计模块，这里用 median()（中位数）
import sys                     # 用来读命令行参数 sys.argv
from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 中文语料专用的分隔符列表：在默认的 ["\n\n", "\n", " ", ""] 基础上补进中文标点，
# 否则一段没有空格的中文会直接掉到 "" 那一级逐字硬切。
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 这是一个"正则表达式"（regular expression，简称 regex）：一种描述字符串模式的小语言。
# 一-鿿 是 Unicode 里中日韩统一表意文字（也就是绝大多数汉字）的编码区间，
# [ ] 表示"这个区间里的任意一个字符"。所以这个模式的含义就是"任意一个汉字"。
# re.compile 把这个模式预编译成对象，反复使用时比每次现写现编译快。
CJK_PATTERN = re.compile(r"[一-鿿]")


def probe(pdf_path: Path) -> None:
    """对一份 PDF 做体检并打印报告。"""
    print("\n" + "=" * 74)
    print(f"体检文件：{pdf_path.name}")
    print(f"文件大小：{pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 74)

    # PdfReader 打开 PDF，reader.pages 是"页对象"的列表。
    reader = pypdf.PdfReader(pdf_path)

    # ---- 第一项：逐页抽文字，每页造一个 Document ----------------------------
    # extract_text() 只能读出 PDF 里"文字图层"的内容。
    # 「文字图层」的意思是：这份 PDF 里真的存了字符编码（比如"李"这个字），
    # 而不是只存了一张写着"李"的图片。扫描件、拍照转的 PDF 就只有图片没有文字图层，
    # 这时 extract_text() 会返回空字符串——它不会报错，这才是麻烦的地方。
    docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": pdf_path.name, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]

    total_chars = sum(len(d.page_content) for d in docs)
    # strip() 去掉首尾空白；空白页去掉空白后就是空字符串，在 if 判断里为假。
    empty_pages = [d.metadata["page"] for d in docs if not d.page_content.strip()]
    cjk_chars = sum(len(CJK_PATTERN.findall(d.page_content)) for d in docs)

    print(f"\n[1] 基本情况")
    print(f"    总页数        : {len(docs)}")
    print(f"    抽出总字符数  : {total_chars}")
    print(f"    其中汉字数    : {cjk_chars}"
          f"（占 {cjk_chars / total_chars * 100:.1f}%）" if total_chars else "")
    print(f"    平均每页字符数: {total_chars // len(docs) if docs else 0}")

    print(f"\n[2] 空页体检（判断是不是扫描件的关键指标）")
    print(f"    抽不出文字的页数: {len(empty_pages)} / {len(docs)}"
          f"（{len(empty_pages) / len(docs) * 100:.1f}%）")
    if empty_pages:
        # 只列前 20 个页码，避免刷屏。切片 [:20] 表示取列表前 20 个元素。
        print(f"    空页页码（前 20 个）: {empty_pages[:20]}")

    # 判定规则是经验值，不是官方标准：
    #   空页 > 30%  → 基本可以断定是扫描件，pypdf 无解，必须上 OCR
    #   空页 5%~30% → 混合型（正文是电子版、插图/表格页是图片），要看丢的是不是关键内容
    #   空页 < 5%   → 电子生成的 PDF，pypdf 够用
    ratio = len(empty_pages) / len(docs) if docs else 1.0
    if ratio > 0.30:
        verdict = "扫描件为主 —— pypdf 无能为力，必须换 MinerU / Docling 这类带 OCR 的工具"
    elif ratio > 0.05:
        verdict = "混合型 —— 正文能读，但部分页丢失，需要人工看看丢的是不是关键内容"
    else:
        verdict = "电子生成的 PDF —— pypdf 抽取基本可用"
    print(f"    判定: {verdict}")

    # ---- 第三项：肉眼检查阅读顺序 -------------------------------------------
    # 「阅读顺序（reading order）」指的是"人应该按什么顺序读这些文字"。
    # PDF 本身不保存这个信息，pypdf 是按文字在文件里出现的先后顺序输出的，
    # 这个顺序对单栏文档通常没问题，但双栏排版就可能把左右两栏的句子交错在一起。
    # 有没有交错，靠打印出来用眼睛看最快，没有可靠的自动检测办法。
    print(f"\n[3] 阅读顺序抽样（请肉眼判断句子有没有被打乱）")
    # 挑三页看：约 1/4 处、中间、约 3/4 处，避开封面和参考文献
    sample_idx = [len(docs) // 4, len(docs) // 2, len(docs) * 3 // 4]
    for idx in sample_idx:
        text = docs[idx].page_content.strip()
        if not text:
            print(f"\n    --- 第 {idx} 页：空页，无文字 ---")
            continue
        print(f"\n    --- 第 {idx} 页前 260 字符 ---")
        # 把换行替换成可见的 \n，再截断，方便一眼看出断行位置
        print("    " + text.replace("\n", "\\n")[:260])

    # ---- 第四项：切分后的块长度分布 -----------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        separators=CHINESE_SEPARATORS,
        chunk_size=400,        # 中文语料的常用起点（单位是字符）
        chunk_overlap=80,      # 20% 重叠
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)
    # 过滤掉太短的碎块（页眉页脚、孤立的页码经常会切成这种）
    kept = [c for c in splits if len(c.page_content) >= 30]
    lengths = [len(c.page_content) for c in kept]

    print(f"\n[4] 切分结果（chunk_size=400 字符，overlap=80）")
    print(f"    {len(docs)} 页 -> {len(splits)} 块 -> 过滤掉短块后剩 {len(kept)} 块")
    if lengths:
        print(f"    块长度：最小 {min(lengths)} / 中位数 "
              f"{statistics.median(lengths):.0f} / 最大 {max(lengths)} 字符")
        mid = kept[len(kept) // 2]
        print(f"\n    中间那一块的 metadata: {mid.metadata}")
        print(f"    中间那一块的内容: {mid.page_content[:200]}...")


def main() -> None:
    # sys.argv 是命令行参数列表，argv[0] 永远是脚本名本身，所以真正的参数从 [1:] 开始。
    args = sys.argv[1:]
    if args:
        targets = [DATA_DIR / a if not Path(a).is_absolute() else Path(a) for a in args]
    else:
        # glob("*.pdf") 列出目录里所有扩展名为 .pdf 的文件；sorted 保证顺序稳定。
        targets = sorted(DATA_DIR.glob("*.pdf"))

    if not targets:
        print(f"在 {DATA_DIR} 下没找到任何 PDF。")
        return

    for path in targets:
        if not path.exists():
            print(f"跳过（文件不存在）：{path}")
            continue
        probe(path)


if __name__ == "__main__":
    main()
