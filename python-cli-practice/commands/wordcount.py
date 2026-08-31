import logging
import re
from collections import Counter
from pathlib import Path

# 获取这个模块自己的logger
logger = logging.getLogger(__name__)
def wordcount(file: Path, top: int = 10):
    """Count the number of words in a file and return the top N most common words."""
    # 记录一条info日志
    logger.info(f"开始处理文件: {file}")
    # 1.读取整份文本内容为一个字符串
    text = file.read_text(encoding="utf-8")
    # 2.用正则表达式把文本拆成单词列表，全部转为小写
    words = re.findall(r"\w+", text.lower())
    # 自动计数器，传入一个可迭代对象，自动统计每个元素出现的次数，返回一个字典
    word_counts = Counter(words)
    logger.info(f"共统计到 {len(word_counts)} 个不同的词")
    # most_common(n) 返回一个列表，里面是出现次数最多的 n 个元素，按出现次数从大到小排序
    for word, count in word_counts.most_common(top):
        print(f"{word}: {count}")