# RAG 学习脚本流程图

对应 `rag_01`~`rag_04` 四个脚本的真实执行流程。节点名就是代码里的函数名或 API 调用，
边上标的是那一步进出的数据形态，橙色警示节点标的是**会静默失败**的位置。
终端里当纯文本读，提交到 GitHub 会渲染成图。

---

## 图 1a — Markdown 结构化切分链（`rag_01_load_and_split.py`）

```mermaid
flowchart TD
    F["data/sample_zh.md<br/>2430 字节"]
    D0["Document<br/>page_content=全文<br/>metadata=source/format"]
    S1["split_by_markdown_header<br/>MarkdownHeaderTextSplitter<br/>strip_headers=False"]
    M["手动合并 metadata<br/>把 source 补回每个块"]
    S2["split_recursively<br/>RecursiveCharacterTextSplitter<br/>chunk_size=120 overlap=24"]
    G["质量闸门<br/>丢弃长度小于 20 字符的块"]
    OUT["11 块<br/>最小 26 / 中位数 94 / 最大 118 字符"]
    W1["注意：唯一吃 str 吐 Document 的切分器<br/>它自己造 metadata<br/>上游字段不会自动继承"]

    F -->|"read_text encoding=utf-8<br/>得到 str"| D0
    D0 -->|"split_text 吃 str<br/>吐 list of Document"| S1
    S1 -->|"8 块 · metadata 多出 h1/h2/h3"| M
    M -->|"list of Document"| S2
    S2 -->|"14 块 · metadata 多出 start_index"| G
    G -->|"list of Document"| OUT

    S1 -.-> W1
    W1 -.-> M

    style W1 fill:#fff3cd,stroke:#d39e00
```

**这张图最该记住的**：两级切分之间那个「手动合并 metadata」不是可有可无的胶水。
`MarkdownHeaderTextSplitter` 是唯一「吃字符串、吐 `Document`」的切分器，
它自己造 metadata，所以上游的 `source` 不会自动带过来；
后面的 `split_documents` 才会自动继承，前提是这一步已经补齐了。

---

## 图 1b — PDF 通道与体检判定（`rag_01` 第 6 步 / `rag_02_pdf_probe.py`）

```mermaid
flowchart TD
    P["PDF 文件"]
    R["pypdf.PdfReader"]
    E["逐页 page.extract_text 兜底为空串"]
    DOCS["N 个 Document<br/>metadata=source/page"]
    CHK{"抽不出文字的空页占比"}
    OCR["换 MinerU / Docling<br/>带 OCR 与版面理解"]
    MIX["混合型<br/>人工确认丢的是不是关键内容"]
    SPL["RecursiveCharacterTextSplitter<br/>chunk_size=400 overlap=80"]
    OUT2["块列表<br/>论文实测 47 页到 174 块"]
    W2["注意：扫描页返回空字符串而不报错<br/>数据在这里悄悄丢失"]

    P -->|"文件路径"| R
    R -->|"reader.pages 页对象列表"| E
    E -->|"list of Document 每页一条"| DOCS
    DOCS --> CHK
    CHK -->|"大于 30%"| OCR
    CHK -->|"5% 到 30%"| MIX
    CHK -->|"小于 5% · pypdf 够用"| SPL
    SPL -->|"list of Document"| OUT2

    E -.-> W2

    style W2 fill:#fff3cd,stroke:#d39e00
```

**这张图最该记住的**：`extract_text()` 在扫描页上返回空字符串**而不是抛异常**，
所以「空页占比」这个统计不是装饰——它是唯一能让你发现数据在这一步悄悄丢了的手段，
也是「要不要上重型解析工具」的决策依据。你的毕业论文实测 0/47，pypdf 够用。

---

## 图 2 — 嵌入的两条编码路径（`rag_03_embeddings.py`）

```mermaid
flowchart TD
    DT["文档文本 list of str"]
    DE["embed_documents<br/>走 encode_kwargs<br/>不加任何前缀"]
    DV["文档向量<br/>每条 512 维 · 模长 1"]

    QT["查询文本 str"]
    QP["拼前缀<br/>为这个句子生成表示以用于检索相关文章"]
    QE["embed_query<br/>走 query_encode_kwargs"]
    QV["查询向量<br/>512 维 · 模长 1"]

    MODEL["SentenceTransformer 流水线<br/>0 Transformer → 1 Pooling → 2 Normalize"]
    COS["cosine<br/>点积除以两个模长"]
    W3["注意：Normalize 是模型内部第 3 层<br/>normalize_embeddings=False 关不掉它"]

    DT --> DE --> DV
    QT --> QP --> QE --> QV

    DE -.->|"同一个模型对象"| MODEL
    QE -.->|"同一个模型对象"| MODEL
    MODEL -.-> W3

    DV -->|"ndarray"| COS
    QV -->|"ndarray"| COS
    COS -->|"相似度 -1 到 1"| SCORE["排序 / 取 top-k"]

    style W3 fill:#fff3cd,stroke:#d39e00
```

**这张图最该记住的**：查询和文档走的是**两条不同的编码路径**，差别只在查询端多拼了一句固定前缀。
前缀加到文档端、或者两端都加，等于查询和文档说了两套语言——不报错，只是默默掉点。
另外 `Normalize` 是模型内部的第 3 层，参数关不掉，所以向量模长恒为 1，点积直接等于余弦相似度。

---

## 图 3a — 原生 FAISS：索引里到底发生了什么（`rag_04` 第 1 步）

```mermaid
flowchart TD
    CH["build_chunks<br/>8 块 list of Document"]
    EMB["emb.embed_documents"]
    VEC["ndarray 形状 8 乘 512<br/>dtype 必须是 float32"]
    IL2["faiss.IndexFlatL2 512<br/>欧氏距离 · 越小越像"]
    IIP["faiss.IndexFlatIP 512<br/>内积 · 越大越像"]
    SRCH["index.search qv k<br/>qv 必须是二维数组"]
    RES["distances 加 indices<br/>只有下标 · 没有原文"]
    W4["注意：float64 直接报错<br/>下标到原文的映射要自己维护"]

    CH -->|"取 page_content"| EMB
    EMB -->|"list of list of float"| VEC
    VEC -->|"index.add"| IL2
    VEC -->|"index.add"| IIP
    IL2 --> SRCH
    IIP --> SRCH
    SRCH -->|"两个 ndarray 形状 1 乘 k"| RES
    RES -.-> W4

    style W4 fill:#fff3cd,stroke:#d39e00
```

**这张图最该记住的**：原生 `search()` **只还给你下标，它根本不知道原文是什么**，
「下标 → Document」的映射得你自己维护——LangChain 封装帮你干的主要就是这件事。
另外对归一化向量有 `L2距离² = 2 - 2×余弦相似度`，两种索引排序必然相同，只是分数读法不同。

---

## 图 3b — LangChain 封装：FAISS 与 Chroma（`rag_04` 第 2 至 4 步）

```mermaid
flowchart TD
    CH2["8 块 list of Document"]
    FS["FAISS.from_documents<br/>编码 建索引 建映射 一步完成"]
    CS["Chroma<br/>collection_name<br/>embedding_function<br/>persist_directory"]
    HIT["similarity_search_with_score<br/>返回 Document 与距离的二元组列表"]
    FLT["Chroma 独有<br/>filter 先按 metadata 筛再检索"]
    SAVE["serialize_to_bytes<br/>再用 Path.write_bytes 落盘"]
    AUTO["写入即落盘<br/>不需要调用任何 save"]
    RET["as_retriever<br/>similarity 或 mmr<br/>上层只认 invoke 传字符串"]
    W5["注意：返回的是距离 越小越相关<br/>按分数从高到低排会完全反"]

    CH2 --> FS
    CH2 --> CS
    FS --> HIT
    CS --> HIT
    CS --> FLT
    HIT -.-> W5
    FS -->|"save_local 在中文路径下崩溃"| SAVE
    CS --> AUTO
    HIT --> RET

    style W5 fill:#fff3cd,stroke:#d39e00
```

**这张图最该记住的**：两个库的 `similarity_search_with_score` 返回的都是**距离**（实测数值完全一致，
都是 0.8091 / 0.9718 / 1.0225），越小越相关；按「分数高的排前面」写排序会完全反过来，而且不报错。
`as_retriever` 是上层唯一该依赖的接口——底下换 FAISS、Chroma 还是 Milvus，业务代码一行都不用改。

mermaid
flowchart LR
    subgraph OFF["离线侧 · python -m app.ingest（人工触发）"]
        F["data/*.md *.pdf"] -->|"str"| SP["切分<br/>标题切 + 递归字符切"]
        SP -->|"list[Document]<br/>带章节 metadata"| ID["生成稳定 id<br/>source#章节#start_index"]
        ID -->|"documents + ids"| UP["Chroma upsert"]
    end
    subgraph ON["在线侧 · FastAPI 进程（只读）"]
        Q["POST /v1/ask"] -->|"str"| R["retriever.ainvoke<br/>k 条"]
        R -->|"list[Document]"| C["拼编号上下文<br/>手写"]
        C -->|"str"| P["ChatPromptTemplate<br/>严格拒答系统提示词"]
        P -->|"PromptValue"| L["model.ainvoke<br/>asyncio.timeout 包裹"]
        L -->|"AIMessage"| CIT["解析 [编号] → Citation"]
        CIT -->|"AskResponse"| OUT["答案 + 出处"]
    end
    UP -.->|"chroma_db/ 磁盘"| R
    style UP fill:#d4edda,stroke:#28a745
    style R fill:#d4edda,stroke:#28a745
