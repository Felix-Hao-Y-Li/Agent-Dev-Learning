"""离线入库命令 —— 把 data/ 里的文档灌进向量库。

运行方式（在项目根目录下）：
    uv run python -m app.ingest              正常入库
    uv run python -m app.ingest --reset      先清空集合，再入库

**为什么入库要单独做成一个命令，而不是服务启动时顺手做掉？**

Chroma 的本地库底层是一个 SQLite 数据库文件。
SQLite 不支持多个进程同时往里写。
而我们的服务将来要多开几个进程来扛并发。
如果每个进程启动时都写一遍库，它们会抢同一个文件，结果是数据损坏或启动失败。

所以这里划一条硬边界：只有这个命令写库，服务进程只读。
这条决策记在 README 的 ADR-003。

函数速查表（吃什么 -> 吐什么 -> 谁会调它）
    check_ids_unique(ids)
        吃一个 id 列表 -> 无返回值。有重复就抛异常，没重复就静静返回。
        被 main() 调用。
    upsert_in_batches(store, chunks, ids)
        吃「向量库 + 块列表 + id 列表」-> 吐实际写入的条数。
        被 main() 调用。
    main()
        整个命令的流程编排。被命令行调用。

阅读地图
    必读：check_ids_unique（为什么要提前自检）、upsert_in_batches（为什么要分批）
    扫读：main 的七个步骤
"""

import argparse

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.rag.chunking import load_documents, split_documents, stable_id
from app.rag.embeddings import build_embeddings
from app.rag.store import build_store, count_documents

logger = get_logger(__name__)

# [了解] 每批往库里写多少条。
#        21 块的小语料一批就写完了，这个参数看不出效果。
#        但语料上万条时，一次性把所有文字送进 GPU 编码会把显存撑爆。
#        分批是让内存占用变成常数而不是随语料线性增长。
BATCH_SIZE = 64


def check_ids_unique(ids: list[str]) -> None:
    """检查 id 有没有重复，有就立刻抛异常。"""
    # ---- 练习 1：实现唯一性自检 -------------------------------------------
    # [核心] 为什么要单独做这个检查，而不是等 Chroma 自己报错？
    #
    #        Chroma 确实会对重复 id 报错，但那个错误发生在「写库」那一刻。
    #        从「stable_id 拼错了」到「写库报错」之间隔着好几层函数调用。
    #        报错信息只会告诉你「有重复的 id」，不会告诉你 id 是怎么拼出来的。
    #
    #        在这里检查，问题就暴露在离病根最近的地方。
    #        这种做法叫「前置校验」，是让 bug 变便宜的常用手段。
    #
    # [核心] 要求写三个步骤：
    #   步骤 1  用 set(ids) 去重，和原列表比长度，算出重复了几个。
    #   步骤 2  没有重复就直接 return。
    #   步骤 3  有重复就抛 ValueError，异常消息里要带上重复的那几个 id。
    #           怎么找出重复的：遍历 ids，用一个 set 记录见过的，
    #           遇到已经见过的就收集起来。
    # 步骤1：去重后比长度，算出重复了几个
    dup_count = len(ids) - len(set(ids))
    # 步骤2：没有重复就直接返回
    if dup_count == 0:
        return
    # 步骤3：有重复就找出是哪几个，带进异常消息里
    seen: set[str] = set()
    dups: list[str] = []
    for i in ids:
        if i in seen:
            dups.append(i)
        else:
            seen.add(i)
    # -----------------------------------------------------------------------


def upsert_in_batches(store: Chroma, chunks: list[Document], ids: list[str]) -> int:
    """分批把块写进向量库，返回写入的总条数。"""
    # ---- 练习 2：实现分批写入 ---------------------------------------------
    # [核心] **upsert** 是 update + insert 拼出来的词。
    #        含义是：这个 id 库里没有就插入，已经有就覆盖。
    #
    #        langchain_chroma 的写入方法内部走的就是 upsert。
    #        所以只要 id 稳定，重复跑这个命令就是覆盖，不会堆出重复数据。
    #        这正是知识库增量更新的正确姿势：文档改了就重跑一次。
    #
    # [核心] 要求写三个步骤：
    #   步骤 1  用 range(0, len(chunks), BATCH_SIZE) 产生每批的起始下标。
    #           比如 100 条、每批 64，就会得到 0 和 64 两个起点。
    #   步骤 2  每批取出 chunks[start:start+BATCH_SIZE] 和对应的 ids 切片，
    #           调 store.add_documents(那批块, ids=那批 id)。
    #           **两个切片的范围必须完全一致**，错开一位会让 id 和内容对不上号。
    #   步骤 3  每批写完记一条 info 日志，带上这批的条数和累计进度。
    #           最后返回总条数。
    total = 0
    # 步骤1：算出每批的起始下标
    for start in range(0, len(chunks), BATCH_SIZE):
        # 步骤2：切出这一批的块和对应的id，写进库
        batch_chunks = chunks[start:start + BATCH_SIZE]
        batch_ids = ids[start:start + BATCH_SIZE]
        store.add_documents(batch_chunks, ids=batch_ids)

        # 步骤3：记一条日志，带上进度
        total += len(batch_chunks)
        logger.info(
            "批次写入完成",
            extra={"batch": len(batch_chunks), "done": total, "all": len(chunks)}
        )

    return total
    # -----------------------------------------------------------------------


def main() -> None:
    """[脚手架] 命令的流程编排，七个步骤。"""
    # 步骤 1：解析命令行参数。
    #         argparse 是标准库里解析命令行参数的模块。
    #         文档：https://docs.python.org/3/library/argparse.html
    #         action="store_true" 表示这是一个开关，写了就是 True，不写就是 False。
    parser = argparse.ArgumentParser(description="把 data/ 下的文档入库")
    parser.add_argument("--reset", action="store_true", help="入库前先清空集合")
    args = parser.parse_args()

    # 步骤 2：读配置，并把日志配置好。
    #         日志必须先配，否则后面所有 logger.info 都打不出来。
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("开始入库", extra={"data_dir": str(settings.data_dir), "reset": args.reset})

    # 步骤 3：加载文档，切成块。
    docs = load_documents(settings.data_dir)
    if not docs:
        logger.error("data/ 目录下没有可加载的文档，终止")
        return
    chunks = split_documents(docs)

    # 步骤 4：生成稳定 id，并立刻自检唯一性。
    ids = [stable_id(c) for c in chunks]
    check_ids_unique(ids)

    # 步骤 5：加载嵌入模型，连接向量库。
    #         注意顺序：模型要先加载好，才能传给向量库。
    embeddings = build_embeddings(settings)
    store = build_store(settings, embeddings)

    # 步骤 6：需要的话先清空集合。
    #         reset_collection() 走数据库层删除，不碰文件系统。
    #         为什么不用删目录的方式清空？因为 Windows 上文件可能还被进程占着，
    #         删除会失败，而失败又常常被静默忽略，表现成「清空没生效」。
    if args.reset:
        store.reset_collection()
        logger.info("集合已清空")

    # 步骤 7：写库，并对比写入前后的条数。
    before = count_documents(store)
    written = upsert_in_batches(store, chunks, ids)
    after = count_documents(store)

    logger.info(
        "入库完成",
        extra={"written": written, "total_before": before, "total_after": after},
    )
    print(f"\n入库完成：本次写入 {written} 条，库里从 {before} 条变成 {after} 条")
    if before > 0 and before == after:
        print("条数没变，说明这些 id 已经在库里，本次是覆盖更新（upsert 生效）")


if __name__ == "__main__":
    main()
