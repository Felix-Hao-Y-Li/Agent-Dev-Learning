# syntax=docker/dockerfile:1
#
# 文档问答服务的镜像。
#
# 三个设计取舍写在这里，面试时可以直接讲：
#   1. Linux 下装 CPU 版 torch，不装 CUDA 版。
#      CUDA 版加上 nvidia 运行库超过 3 GB，镜像会大到推不动，
#      而且要宿主机装 nvidia-container-toolkit 才能真的用上显卡。
#      代价是编码速度慢几倍，本项目语料小，可以接受。
#   2. 模型权重在构建时下载并烘进镜像，运行时不联网。
#      好处是启动快、不依赖 HuggingFace 的可用性、离线环境也能部署。
#      代价是镜像大了约 100 MB，换模型要重新构建。
#   3. 向量库不进镜像，运行时挂载卷。
#      因为它是数据不是代码，而且要在 ingest 和 api 两个容器之间共享。

FROM python:3.12-slim

# [核心] uv 官方提供了一个只含二进制的镜像，直接从里面把可执行文件拷过来。
#        比用 pip 装 uv 快，而且不会污染 Python 环境。
#        版本写死而不用 latest：构建结果要可复现，今天和三个月后必须装出同一套东西。
COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /uvx /bin/

# [了解] 三个环境变量各解决一个问题：
#        UV_LINK_MODE=copy      容器里缓存和目标不在同一文件系统，硬链接会失败并告警。
#        UV_COMPILE_BYTECODE=1  构建时就把 .py 编译成 .pyc，容器启动能快一点。
#        PYTHONUNBUFFERED=1     Python 默认会缓冲 stdout，日志要攒够一批才输出。
#                               容器里这会导致 docker logs 半天看不到东西，出事时抓瞎。
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------------------------------------------------------------------------
# 第 1 层：只装依赖
# ---------------------------------------------------------------------------
# [核心] 为什么先单独拷这两个文件，而不是一次性 COPY . ？
#        因为 Docker 是按层缓存的：某一层的输入没变，这一层就直接复用缓存。
#        依赖清单几天才改一次，代码一天改几十次。
#        分开之后，改代码只会让后面几层失效，装依赖这一层（最慢的一层）能一直命中缓存。
#        合成一层的话，每改一行代码都要重装几百兆依赖。
COPY pyproject.toml uv.lock ./

# [了解] --locked      严格按 uv.lock 装，版本对不上就报错。
#                      保证镜像里的版本和你本地测过的完全一致。
#        --no-dev      不装 pytest 这些开发依赖，生产镜像里不需要。
#        --no-install-project  只装第三方依赖，先不装项目本身（代码还没拷进来）。
RUN uv sync --locked --no-dev --no-install-project

# 后面的命令直接用虚拟环境里的 python，不用每次写 uv run。
ENV PATH="/app/.venv/bin:$PATH"

# ---------------------------------------------------------------------------
# 第 2 层：把嵌入模型烘进镜像
# ---------------------------------------------------------------------------
# [核心] HF_HOME 指定 HuggingFace 的缓存目录。
#        不指定的话默认在 ~/.cache，而我们后面会换成非 root 用户运行，
#        家目录会变，缓存就找不到了。显式指定一个固定路径最稳妥。
ENV HF_HOME=/opt/hf

# [核心] 构建时把 HF_HUB_OFFLINE 设成 0，允许联网下载；
#        运行时（见下面的 ENV）设回 1，禁止联网。
#        这样模型只在构建时下载一次，之后每次启动都直接读本地缓存。
RUN HF_HUB_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# ---------------------------------------------------------------------------
# 第 3 层：拷代码
# ---------------------------------------------------------------------------
COPY app ./app
COPY data ./data

# ---------------------------------------------------------------------------
# 运行时配置
# ---------------------------------------------------------------------------
# [核心] 这些值会被 Settings 读到（环境变量名是字段名的大写形式）。
#        CHROMA_DIR 指向 /data/chroma，那是运行时挂载卷的位置，不在镜像里。
ENV HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    EMBEDDING_DEVICE=cpu \
    CHROMA_DIR=/data/chroma \
    DATA_DIR=/app/data

# [核心] 建一个非 root 用户来跑服务。
#        容器默认用 root，一旦应用被攻破，攻击者在容器里就是 root，
#        再配合容器逃逸漏洞就可能拿到宿主机权限。
#        这是容器安全检查的第一条，面试也常问。
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/chroma \
    && chown -R appuser:appuser /app /data /opt/hf
USER appuser

EXPOSE 8000

# [了解] HEALTHCHECK 让 Docker 自己定期探活。
#        start-period=40s 是给启动留的宽限期：加载模型要十几秒，
#        这段时间内探测失败不算数，否则容器会被判定不健康而反复重启。
#        注意探的是 /healthz 不是 /readyz，理由见 app/api/routes/health.py。
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# [核心] --host 0.0.0.0 不能省。
#        默认只监听 127.0.0.1，那是「容器内部的本机」，
#        宿主机的端口映射会连不上，表现为端口映射了却死活访问不了。
#        这是容器化最常见的坑之一。
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
