import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

def setup_logging():
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(log_dir / "app.log", interval = 1, when = "midnight", backupCount=7, encoding="utf-8")
    # level=logging.INFO：只显示 INFO 级别及以上的日志（INFO、WARNING、ERROR、CRITICAL
    # 更低级别的 DEBUG 会被忽略
    # format=...：定义每条日志打印出来的格式
    #   %(asctime)s  -> 时间戳
    #   %(levelname)s -> 级别名称，比如 INFO
    #   %(name)s     -> 是哪个模块打的日志（对应 getLogger(__name__)）
    #   %(message)s  -> 你实际写的日志内容
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            file_handler  # 同时输出到文件
        ]
    )
