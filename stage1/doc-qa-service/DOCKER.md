# 容器化部署说明

这份文档覆盖三件事：怎么构建、怎么运行、遇到问题怎么查。

## 零、名词先讲清楚

**镜像（image）**：一个打包好的只读文件系统快照，里面有操作系统、Python、依赖、你的代码。
可以理解成「装好了一切的硬盘镜像」。

**容器（container）**：镜像跑起来之后的进程。同一个镜像可以同时跑出很多个容器。
镜像是类，容器是实例。

**卷（volume）**：容器内部的文件系统是临时的，容器删掉就没了。
要持久保存的数据必须放在卷里，卷由 Docker 单独管理，独立于容器的生命周期。

**编排（orchestration）**：一个服务往往由多个容器组成，还有启动顺序、网络、卷的依赖关系。
`docker-compose.yml` 就是把这些关系写下来，一条命令拉起全部。

## 一、前置条件

1. 装好 Docker Desktop，并确认能用：

   ```powershell
   docker --version
   docker compose version
   ```

2. **确认锁文件是最新的**。`Dockerfile` 用的是 `uv sync --locked`，
   意思是严格按 `uv.lock` 安装，和 `pyproject.toml` 对不上就直接报错：

   ```powershell
   uv lock --check
   ```

   输出 `Resolved N packages` 就说明没问题。如果它报错，执行一次 `uv lock` 再构建。

3. 确认 `.env` 存在且有 `DEEPSEEK_API_KEY`。
   这个文件**不会**进镜像（`.dockerignore` 排除了它），是运行时通过 `env_file` 注入的。

## 二、构建

```powershell
docker compose build
```

第一次构建要做四件事，耗时主要花在前两件上：

| 步骤 | 内容 | 预计耗时 |
|---|---|---|
| 1 | 拉 `python:3.12-slim` 基础镜像 | 1 分钟内 |
| 2 | 装依赖（CPU 版 torch 约 200 MB） | 3～8 分钟，看网速 |
| 3 | 下载 bge 模型权重（约 100 MB） | 1 分钟内 |
| 4 | 拷代码 | 几秒 |

构建完看一下体积：

```powershell
docker images doc-qa-service
```

预期在 **1.5～2.5 GB** 之间。如果看到 8 GB 以上，说明装成了 CUDA 版 torch，
去检查 `pyproject.toml` 里 `[tool.uv.sources]` 那段的平台标记。

**之后再构建会快很多**：只要 `pyproject.toml` 和 `uv.lock` 没变，
装依赖那一层直接命中缓存，改代码只需要重跑最后一层。
这就是 Dockerfile 里「先拷依赖清单、再拷代码」的意义。

## 三、运行

```powershell
docker compose up -d
```

`-d` 是后台运行。这条命令会按顺序做两件事：

1. 启动 `ingest` 容器，把 `data/` 里的文档写进向量库，跑完自动退出。
2. 等 `ingest` 成功退出后，启动 `api` 容器。

看日志：

```powershell
docker compose logs -f api
```

## 四、验证

```powershell
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

`/readyz` 应返回 `{"status":"ready","documents":21}`。

提问用项目自带的脚本：

```powershell
.\scripts\ask.ps1 "退货的运费谁出？"
```

不要直接用 `Invoke-RestMethod`。Windows PowerShell 5.1 会把中文答案显示成乱码，
原因和解决办法写在 `scripts/ask.ps1` 的开头。

浏览器打开 <http://127.0.0.1:8000/docs> 可以看接口文档并直接试调用。

## 五、日常操作

| 目的 | 命令 |
|---|---|
| 改了 `data/` 里的文档，重新入库 | `docker compose run --rm ingest` |
| 改了代码，重建并重启 | `docker compose up -d --build` |
| 停止服务（保留数据） | `docker compose down` |
| 停止并清空向量库 | `docker compose down -v` |
| 进容器里看看 | `docker compose exec api bash` |

改了 `data/` 之后**不需要重新构建镜像**，因为语料目录是从宿主机挂进去的。
重跑一次 `ingest` 即可，稳定 id 保证是覆盖而不是追加。

## 六、常见问题

**端口映射了却访问不了**
检查 `CMD` 里有没有 `--host 0.0.0.0`。
默认只监听 `127.0.0.1`，那是「容器内部的本机」，宿主机连不进去。

**容器反复重启**
`docker compose logs api` 看最后的报错。
如果日志停在加载模型那一步，多半是 `HEALTHCHECK` 的 `start-period` 太短，
模型还没加载完就被判定不健康。本项目给了 40 秒。

**`uv sync --locked` 报版本对不上**
宿主机改过 `pyproject.toml` 但没重新 `uv lock`。执行一次 `uv lock` 再构建。

**`/readyz` 一直 503**
`ingest` 没跑成功，或者两个容器挂的不是同一个卷。
用 `docker compose run --rm ingest` 手动跑一次，看它的输出。

**镜像里没有 `.env`，密钥怎么进去的**
通过 `docker-compose.yml` 的 `env_file` 在**启动容器时**注入成环境变量。
密钥绝不能打进镜像——镜像会被推到仓库、被别人拉取，打进去等于公开。

## 七、这套方案的取舍

| 决定 | 好处 | 代价 |
|---|---|---|
| Linux 下用 CPU 版 torch | 镜像从 10 GB 降到 2 GB 上下，不依赖宿主机显卡 | 向量编码慢几倍 |
| 模型权重烘进镜像 | 启动快，不依赖 HuggingFace 可用性，离线环境可部署 | 镜像大约 100 MB，换模型要重建 |
| 向量库放命名卷 | 数据和代码分离，两个容器能共享 | 多一个卷要管理，换机器要重新入库 |
| ingest 和 api 拆成两个服务 | 写读分离落到部署层面，符合 ADR-003 | 部署多一个步骤 |

**如果要上 GPU**：基础镜像换成 `nvidia/cuda:12.x-runtime`，
torch 装回 CUDA 版，`EMBEDDING_DEVICE` 改成 `cuda`，
并在 compose 里声明 `deploy.resources.reservations.devices`。
宿主机需要装 nvidia-container-toolkit。镜像体积会到 8 GB 以上。
