"""切分层的测试 —— 不需要任何模型，纯文本处理。

测试文件的命名规则是 pytest 的约定：文件名 test_*.py，函数名 test_*。
pytest 靠这个约定自动找到要跑的测试，不需要注册。

阅读地图
    必读：test_stable_id_is_stable（为什么这条是最重要的）
"""

from langchain_core.documents import Document

from app.rag.chunking import MIN_CHUNK_CHARS, split_documents, stable_id

# 一份小的 Markdown，结构和真实语料一样，但短到能一眼看完。
SAMPLE_MD = """# 售后服务政策

## 一、退货

### 退货运费
质量问题由平台承担运费。

### 时限
签收后 7 天内可申请。

## 二、发票

电子普票 24 小时内开具。
"""


def _make_doc() -> Document:
    return Document(page_content=SAMPLE_MD, metadata={"source": "sample.md"})


def test_split_keeps_source_metadata():
    """切完之后，每一块都要还记得自己来自哪个文件。"""
    # 步骤 1：切分。
    chunks = split_documents([_make_doc()])

    # 步骤 2：断言每块都带着 source。
    #         这一条专门防住一个真实的坑：
    #         MarkdownHeaderTextSplitter 会新造 metadata，
    #         如果忘了手动合并上游的 metadata，source 就会丢。
    #         丢了之后答案标不出出处，而且不会报错。
    assert chunks, "切分结果不该为空"
    for c in chunks:
        assert c.metadata.get("source") == "sample.md"


def test_split_records_section_path():
    """切完之后，块要知道自己属于哪一章哪一节。"""
    chunks = split_documents([_make_doc()])

    # 步骤 1：找出那块讲退货运费的。
    target = next(c for c in chunks if "平台承担运费" in c.page_content)

    # 步骤 2：三级标题应该都记下来了。
    assert target.metadata["h1"] == "售后服务政策"
    assert target.metadata["h2"] == "一、退货"
    assert target.metadata["h3"] == "退货运费"


def test_split_drops_tiny_chunks():
    """太短的碎块要被丢掉。"""
    chunks = split_documents([_make_doc()])
    for c in chunks:
        assert len(c.page_content) >= MIN_CHUNK_CHARS


def test_stable_id_is_stable():
    """同样的输入，两次切分要得到完全一样的 id。

    这是整个离线侧最重要的一条测试。
    id 一旦不稳定，重复入库就会变成不断追加重复数据，
    知识库里堆满副本，检索结果前几名全是同一段话。
    而且这个问题不会报错，只会表现为「答案质量莫名其妙变差」。
    """
    # 步骤 1：独立切两次。
    ids_first = [stable_id(c) for c in split_documents([_make_doc()])]
    ids_second = [stable_id(c) for c in split_documents([_make_doc()])]

    # 步骤 2：两次结果必须逐个相同。
    assert ids_first == ids_second


def test_stable_id_is_unique():
    """同一篇文档切出来的块，id 不能撞车。"""
    ids = [stable_id(c) for c in split_documents([_make_doc()])]

    # set 去重后长度不变，说明没有重复。
    assert len(ids) == len(set(ids)), f"出现重复 id：{ids}"


def test_stable_id_contains_section():
    """id 里要带章节路径，否则不同小节的第一块会撞车。

    因为 start_index 是相对于「传给切分器的那一小节」的偏移，
    每个小节的第一块 start_index 都是 0。
    """
    chunks = split_documents([_make_doc()])
    target = next(c for c in chunks if "平台承担运费" in c.page_content)

    got = stable_id(target)
    assert got.startswith("sample.md#")
    assert "退货运费" in got
