"""配置层 —— 把环境变量读成一个带类型校验的对象。

为什么单独一层：生产环境里同一份代码要跑在开发机、测试机、线上，差异全部体现在配置上。
配置一旦硬编码进业务代码，换环境就得改代码、改代码就得重新测试、重新发版。
这条原则叫 12-Factor App 的「配置与代码分离」（Config in the environment）。

阅读地图
    必读：Settings 类的字段定义 —— 这就是本服务全部可调的旋钮
          get_settings()        —— 为什么要缓存，不缓存会怎样
    扫读：BASE_DIR 的算法
"""

from functools import lru_cache        # 标准库：给函数加缓存的装饰器。文档：https://docs.python.org/3/library/functools.html
from pathlib import Path

# pydantic-settings 是 pydantic 官方的配置扩展包：
# 声明一个类，它自动去环境变量和 .env 文件里找同名的值，并按类型注解做校验和转换。
# 文档：https://docs.pydantic.dev/latest/concepts/pydantic_settings/
from pydantic_settings import BaseSettings, SettingsConfigDict

# [脚手架] 本文件位于 <项目根>/app/core/config.py，parents[2] 就是项目根目录。
#          parents[0]=core, parents[1]=app, parents[2]=项目根。
#          用绝对路径而不是相对路径，是因为「从哪个目录启动服务」不该影响它去哪里找 .env。
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """服务的全部配置项。字段名小写，对应的环境变量名是它的大写形式。

    例如字段 `retrieval_top_k` 会去读环境变量 `RETRIEVAL_TOP_K`。
    """

    # ---- 练习 1：把这个类和 .env 文件接上 ----------------------------------
    # [核心] 没有这一行，Settings 只会读系统环境变量，不会去看 .env 文件。
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # ------------------------------------------------------------------------

    # ---- 应用元信息 --------------------------------------------------------
    app_name: str = "doc-qa-service"
    log_level: str = "INFO"

    # ---- 大模型（生成侧）---------------------------------------------------
    # [核心] 没有默认值的字段是**必填**的。启动时环境变量里找不到它，
    #        pydantic 会直接抛 ValidationError 让进程起不来——这是好事：
    #        宁可启动就炸，也不要跑到第一个用户提问时才发现没配 key。
    deepseek_api_key: str

    llm_model: str = "deepseek:deepseek-v4-flash"
    # [核心] temperature（温度）控制生成的随机性，0 表示尽量确定、每次答案一致。
    #        客服场景要的是稳定复现，不是创意，所以固定 0。
    llm_temperature: float = 0.0
    # [核心] 调外部 API 必须设超时，否则对方卡住就会把我们的连接一直占着，
    #        并发一上来连接池耗尽，整个服务跟着挂。这叫「级联故障」。
    llm_timeout_seconds: float = 30.0

    # ---- 嵌入模型（检索侧）------------------------------------------------
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # [了解] cuda 表示用 NVIDIA 显卡跑；没有显卡的机器要改成 "cpu"（慢 5~10 倍）。
    embedding_device: str = "cuda"
    # [核心] BGE 系列模型的硬要求：查询要加这句指令前缀，文档不加。
    #        两边不一致会导致检索质量悄悄下降且**不报任何错**。
    embedding_query_prefix: str = "为这个句子生成表示以用于检索相关文章："
    # [核心] 同时最多允许几个请求在编码。GPU 显存是有限资源，不限流会 OOM。
    embedding_concurrency: int = 2

    # ---- 向量库 -----------------------------------------------------------
    chroma_dir: Path = BASE_DIR / "chroma_db"
    chroma_collection: str = "ecom_support_kb"
    data_dir: Path = BASE_DIR / "data"

    # ---- 检索与问答 --------------------------------------------------------
    # [核心] 每次检索取回几个片段。太小可能漏掉答案，太大会塞进无关内容
    #        干扰模型、并且推高 token 成本。4 是中文短文档的常用起点。
    retrieval_top_k: int = 4
    # [了解] 请求体里问题的长度上限，防止有人塞一整本书进来把上下文撑爆。
    max_question_chars: int = 500


# ---- 练习 2：让配置只被解析一次 --------------------------------------------
# [核心] 每调用一次 Settings() 都会重新读文件、重新校验。
#        这个函数要保证：无论被调用多少次，都返回**同一个** Settings 实例。
@lru_cache
def get_settings() -> Settings:
    """返回全局唯一的配置对象。"""
    return Settings()
# ----------------------------------------------------------------------------
