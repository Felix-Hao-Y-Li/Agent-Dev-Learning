# pycli-toolbox

用于打磨 Python 基础语法的个人练习项目，与仓库主线（Agent 开发求职路线图）无关，不计入进度跟踪。

一个用 [Typer](https://typer.tiangolo.com/) 搭建的个人 CLI 工具箱，依赖管理用 [uv](https://docs.astral.sh/uv/)。每个子命令对应一个练习主题，覆盖了 Agent 开发路线图中会反复用到的 Python 语法面（类型注解、异步、数据模型、并发、标准库等）。

## 如何运行

```powershell
# 安装依赖（首次或依赖变更后）
uv sync

# 查看所有可用命令
uv run python main.py --help

# 查看某个命令的用法
uv run python main.py <命令> --help
```

## 已实现的命令

| 命令 | 说明 | 示例 |
|---|---|---|
| `hello` | 打个招呼，Typer 最基础的单命令示例 | `uv run python main.py hello --name Alice` |
| `todo add` | 新增一条待办事项 | `uv run python main.py todo add "买菜"` |
| `todo list` | 列出所有待办事项 | `uv run python main.py todo list` |
| `todo done` | 把一条待办标记为已完成 | `uv run python main.py todo done 1` |
| `todo delete` | 删除一条待办事项 | `uv run python main.py todo delete 1` |
| `weather` | 查询单个城市当前天气（异步请求外部 API） | `uv run python main.py weather 北京` |
| `weathers` | 并发查询多个城市当前天气（`asyncio.gather`） | `uv run python main.py weathers 北京 上海 广州` |
| `wordcount` | 统计文本文件中出现频率最高的词 | `uv run python main.py wordcount sample.txt --top 5` |

`todo` 是一个命令组（内部多个子命令），其余是单个顶层命令——两种注册模式的区别见下方"学到的知识点"。

## 项目结构

```
main.py                 # 入口，注册所有命令
logging_config.py        # 全局日志配置
commands/
  hello.py               # hello 命令
  todo.py                 # todo 命令组 + Pydantic 数据模型 + JSON 持久化
  weather.py              # weather / weathers 命令，异步请求 Open-Meteo API
  wordcount.py             # wordcount 命令，正则 + Counter 词频统计
tests/
  test_hello.py            # hello 命令的 pytest 测试
notes/
  python-must-know.md      # Agent 开发方向的 Python 核心知识点整理
data/
  todos.json               # todo 命令的持久化数据（已 gitignore）
```

## 学到的知识点

- **类型注解 & Typer 参数映射**：函数参数的类型注解会被 Typer 自动翻译成 CLI 参数——没有默认值的变成位置参数（如 `todo delete <id>`），有默认值的变成 `--选项`（如 `wordcount --top`）。
- **模块拆分的两种模式**：单个命令用 `app.command()(函数)` 直接挂到顶层（模式 A，如 `hello`/`weather`）；一组相关命令让模块自建 `typer.Typer()` 再用 `add_typer` 挂载成命令组（模式 B，如 `todo`）。踩过的坑：给单命令模块套模式 B 会产生多余的嵌套（`weather weather 北京`），要避免。
- **Pydantic**：`BaseModel` 做数据校验，`model_validate`/`model_dump` 在"外部数据（JSON）"和"内部对象"之间转换，配合 `pathlib` 做简单的文件持久化。
- **异步编程**：`async def`/`await` 定义和驱动协程，`httpx.AsyncClient` 发异步 HTTP 请求，`asyncio.run()` 是同步入口桥接异步逻辑的唯一方式（一个同步入口只能调用一次）；`asyncio.gather()` 并发执行多个协程，耗时约等于最慢的那一个，而不是逐个耗时相加。
- **标准库**：`re.findall` 按正则规则提取文本片段，`collections.Counter` 自动统计元素出现次数并用 `most_common(n)` 取高频项。
- **logging**：`logging.basicConfig()` 全局配置一次即可，各模块用 `logging.getLogger(__name__)` 获取自己的 logger 并自动继承全局配置；默认日志输出到 `stderr`，不落盘，如需持久化要额外加 `FileHandler`。
- **pytest**：约定优于配置——`test_` 开头的文件/函数会被自动发现，`assert` 断言失败即测试失败；`capsys` 这类内置 fixture 可以捕获被测函数的 `print` 输出用于断言。运行测试时如果被测代码不在测试文件同一目录树下，需要在 `pyproject.toml` 里配置 `[tool.pytest.ini_options] pythonpath = ["."]`，否则会因为找不到项目根目录而 `ModuleNotFoundError`。
