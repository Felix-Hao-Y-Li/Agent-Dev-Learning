# Python 核心知识点清单（Agent 开发方向）

> 目的：整理 Agent 开发学习路线中会反复用到的 Python 核心能力，配合官方文档 + demo，练到"看到场景就知道怎么写"的肌肉记忆程度。
>
> 标记说明：🔴 必须练到不查资料就能写出来（面试高频）；🟡 见过、能看懂、知道去哪查即可，不必死记细节。

---

## 1. 类型注解（Type Hints）

**是什么**：给变量、函数参数、返回值标注"这是什么类型"。Python 运行时其实不强制检查这些注解（不写也能跑），但它能让编辑器做智能提示、能让 FastAPI/Pydantic 这类框架"读懂"你的数据结构去自动做校验和转换。

**为什么重要**：Agent 开发里几乎所有框架（FastAPI 路由、Pydantic 模型、LangChain 工具定义）都依赖类型注解来自动生成 schema，写不对注解，框架的自动化能力就用不起来。

### 🔴 基础注解

```python
# 变量注解：变量名: 类型
age: int = 25
name: str = "Alice"

# 函数注解：参数名: 类型，-> 返回值类型
def greet(name: str, times: int = 1) -> str:
    # times: int = 1 表示"类型是 int，默认值是 1"
    return (f"Hello, {name}! " * times).strip()

# 容器类型：list[元素类型]、dict[键类型, 值类型]、tuple[各元素类型...]
scores: list[int] = [90, 85, 77]
config: dict[str, str] = {"env": "dev"}
point: tuple[float, float] = (39.9, 116.4)  # 例如经纬度
```

### 🔴 `X | None`（表示"可能没有值"）

```python
# Python 3.10+ 写法，等价于旧版的 Optional[str]
def find_user(user_id: int) -> str | None:
    users = {1: "Alice"}
    return users.get(user_id)  # 找不到时 dict.get 返回 None，符合注解

result = find_user(2)
if result is None:          # 用 is None 判断，不要用 == None（后面 OOP 章节会解释原因）
    print("用户不存在")
```

### 🟡 `TypedDict` / `Protocol`（了解即可）

```python
from typing import TypedDict

# TypedDict：给普通 dict 加上"这个 key 必须是什么类型"的约束，
# 常见于描述外部 API 返回的 JSON 结构（比如 Agent 工具的返回值）
class WeatherResult(TypedDict):
    temperature: float
    windspeed: float

def show(data: WeatherResult) -> None:
    print(data["temperature"])
```

**官方文档**：https://docs.python.org/3/library/typing.html

---

## 2. 面向对象（OOP）

**是什么**：把数据和操作数据的行为打包成一个"类"，用类创建出的每个"实例"都拥有自己的一份数据。Agent 开发里，一个"工具（Tool）"、一个"Agent"、一个"记忆模块"，通常都会被建模成一个类。

### 🔴 `class` / `__init__` / 继承

```python
class Tool:
    def __init__(self, name: str, description: str):
        # __init__ 是"构造函数"：创建实例时自动调用，用来初始化这个实例自己的数据
        # self 代表"当前这个实例本身"，self.name 就是把参数存到这个实例上
        self.name = name
        self.description = description

    def run(self, query: str) -> str:
        return f"[{self.name}] 处理: {query}"


class SearchTool(Tool):  # 继承 Tool，拥有 Tool 的所有能力，还能再加/覆盖
    def __init__(self, name: str, description: str, api_key: str):
        super().__init__(name, description)  # 调用父类的 __init__，把公共部分交给父类处理
        self.api_key = api_key

    def run(self, query: str) -> str:  # 覆盖父类方法，实现自己的逻辑
        return f"用 {self.api_key} 搜索: {query}"


tool = SearchTool("web_search", "联网搜索", api_key="xxx")
print(tool.run("Python 教程"))
```

### 🔴 `@dataclass`（省去手写 `__init__` 的数据容器）

```python
from dataclasses import dataclass

@dataclass
class Point:
    # 只声明字段，__init__/__repr__（打印格式）都由 dataclass 自动生成
    x: float
    y: float

p = Point(x=1.0, y=2.0)
print(p)  # 自动生成好看的打印格式：Point(x=1.0, y=2.0)
```

> 和 Pydantic 的 `BaseModel` 区别：`dataclass` 不做数据校验（传错类型也不报错），`BaseModel` 会在创建实例时严格校验并转换类型。数据来自外部（用户输入、API 返回）时优先用 Pydantic；纯内部临时数据容器可以用 `dataclass`。

### 🟡 常用魔术方法

```python
class Agent:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        # __repr__ 决定了 print(实例) 或调试时显示的内容
        return f"Agent(name={self.name!r})"

    def __call__(self, task: str) -> str:
        # 定义了 __call__ 之后，实例可以像函数一样被"调用"
        return f"{self.name} 正在处理: {task}"

a = Agent("助手")
print(a)          # 触发 __repr__ -> Agent(name='助手')
print(a("写代码")) # 触发 __call__ -> 助手 正在处理: 写代码
```

**官方文档**：https://docs.python.org/3/tutorial/classes.html

---

## 3. 异步编程（重中之重）

**是什么**：让程序在"等待"（比如等网络请求返回）的时候，不傻等，而是先去处理别的任务，等结果回来了再继续——本质是单线程内的"任务切换"，不是真正的多线程并行。Agent 开发里调用 LLM API、调用外部工具，几乎都是网络 IO，异步能大幅提升吞吐量。

### 🔴 `async def` / `await` / `asyncio.run()`

```python
import asyncio

async def fetch_data(name: str) -> str:
    # await asyncio.sleep(1) 模拟一次耗时 1 秒的网络请求
    # 在真正的异步库（httpx.AsyncClient 等）里，await 期间事件循环可以去处理别的协程
    await asyncio.sleep(1)
    return f"{name} 的数据"

async def main():
    result = await fetch_data("用户A")  # 只能在 async def 函数内部用 await
    print(result)

# asyncio.run() 是"同步世界"进入"异步世界"的唯一入口：
# 它新建一个事件循环、跑完 main()、然后关闭循环，一个同步入口里只能调用一次
asyncio.run(main())
```

### 🔴 `asyncio.gather()`（并发跑多个协程）

```python
import asyncio, time

async def fetch(name: str) -> str:
    await asyncio.sleep(1)
    return f"{name} done"

async def sequential():
    # 顺序 await：一个个等，总耗时 = 1+1+1 = 3 秒
    for name in ["A", "B", "C"]:
        await fetch(name)

async def concurrent():
    # gather：同时发起 3 个任务，一起等，总耗时约等于最慢的那一个 = 1 秒
    await asyncio.gather(fetch("A"), fetch("B"), fetch("C"))

asyncio.run(sequential())   # 约 3 秒
asyncio.run(concurrent())   # 约 1 秒 —— 这就是并发的意义
```

### 🔴 `async with` / `async for`

```python
import httpx

async def get_weather():
    # async with：异步版的 with，进入/退出时的资源准备与清理本身也是异步操作
    # （比如建立/关闭网络连接），所以要用 async with 而不是普通 with
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.open-meteo.com/v1/forecast", params={...})
        return resp.json()
```

### 🟡 `asyncio.Semaphore`（限制并发数）

```python
import asyncio

sem = asyncio.Semaphore(3)  # 最多允许 3 个协程同时执行被它包裹的代码块

async def limited_fetch(name: str):
    async with sem:  # 超过 3 个的请求会在这里排队，防止把外部 API 打爆
        await asyncio.sleep(1)
        return name
```

**官方文档**：https://docs.python.org/3/library/asyncio.html

---

## 4. 上下文管理器（`with` 语句）

**是什么**：确保"进入一段代码块前做准备、离开代码块后自动做清理"，即使中间发生异常也一定会清理——最常见场景是文件/网络连接/数据库连接的"打开-使用-关闭"。

### 🔴 基本用法

```python
# 不用 with：必须手动 close，还容易忘记，或者出异常时 close 根本不会执行
f = open("data.txt", "r")
content = f.read()
f.close()

# 用 with：无论中间是否报错，退出 with 代码块时都会自动调用 f.close()
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
# 出了这个缩进块，文件已经自动关闭
```

### 🟡 自己写一个上下文管理器

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label: str):
    start = time.time()
    yield  # yield 之前是"进入时"要做的事，yield 之后是"退出时"要做的事
    print(f"{label} 耗时: {time.time() - start:.2f}s")

with timer("查询任务"):
    time.sleep(1)
# 输出：查询任务 耗时: 1.00s
```

**官方文档**：https://docs.python.org/3/library/contextlib.html

---

## 5. 装饰器（Decorator）

**是什么**：一个"包装函数的函数"——本质是把你的函数传进去，返回一个新的（通常是加了点额外行为的）函数，再用新函数替换掉原来的名字。`@app.get(...)`、`@app.command()` 这些框架能力全部靠装饰器实现：框架用装饰器"记住"了你的函数，以便之后路由到它。

### 🔴 理解 `@decorator` 的本质

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("调用前")
        result = func(*args, **kwargs)  # 真正执行原函数
        print("调用后")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}")

# 上面的写法完全等价于：
# say_hello = my_decorator(say_hello)

say_hello("Alice")
# 输出：
# 调用前
# Hello, Alice
# 调用后
```

### 🟡 自己写一个计时装饰器

```python
import time
from functools import wraps

def timed(func):
    @wraps(func)  # 保留原函数的名字/文档字符串，不加这个的话 func.__name__ 会变成 "wrapper"
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时 {time.time() - start:.3f}s")
        return result
    return wrapper

@timed
def slow_task():
    time.sleep(0.5)

slow_task()  # 输出：slow_task 耗时 0.500s
```

**官方文档**：https://docs.python.org/3/glossary.html#term-decorator

---

## 6. 迭代器 / 生成器

**是什么**：生成器是"惰性"产生数据的方式——不是一次性把所有结果算好放进内存，而是"要一个给一个"。处理大文件、大量文档切分、LLM 流式输出时，能显著省内存。

### 🔴 生成器表达式 vs 列表推导式

```python
# 列表推导式：立刻把所有结果算出来，存进一个列表，占用对应内存
squares_list = [x * x for x in range(1000000)]

# 生成器表达式：不会立刻计算，只有在被遍历/被 next() 取值时才会一个个算
# 处理超大数据集合时（比如遍历几万篇文档），生成器几乎不占额外内存
squares_gen = (x * x for x in range(1000000))

for s in squares_gen:
    if s > 100:
        break  # 提前退出的话，后面的值根本不会被计算
```

### 🔴 `yield`（写一个生成器函数）

```python
def read_large_file(path: str):
    # 这个函数只要包含 yield，调用它就不会立刻执行，而是返回一个"生成器对象"
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip()  # 每次被 next() 请求时，运行到这里就"暂停"并交出一个值

for line in read_large_file("big.log"):
    # 逐行处理，不需要把整个文件读进内存
    print(line)
```

> LLM 的流式输出（一个 token 一个 token 蹦出来）在 Python 侧的实现原理正是生成器 / 异步生成器（`async def` + `yield`）。

**官方文档**：https://docs.python.org/3/tutorial/classes.html#generators

---

## 7. 错误处理

**是什么**：预判"哪里可能出错"，并且明确地捕获、处理或者向上抛出，而不是让程序直接崩溃。调用外部 API（网络超时、返回格式不对）、调用 Agent 工具（参数错误）是错误处理的高发地带。

### 🔴 `try / except / finally`

```python
def call_api(url: str):
    try:
        # 假设这里调用一个可能失败的网络请求
        response = risky_request(url)
    except TimeoutError:
        print("请求超时，稍后重试")
        return None
    except ValueError as e:
        # as e：把捕获到的异常对象存到变量 e，可以打印详细信息
        print(f"参数错误: {e}")
        return None
    finally:
        # 无论成功、失败、还是 return，finally 里的代码都会执行——常用来做资源清理
        print("请求结束")
```

### 🔴 自定义异常 + `raise ... from`

```python
class ToolExecutionError(Exception):
    """自定义异常：让错误信息带有业务含义，而不是笼统的 Exception"""
    pass

def run_tool(name: str):
    try:
        1 / 0  # 模拟内部真实抛出的异常
    except ZeroDivisionError as e:
        # raise X from e：抛出新异常的同时，保留原始异常的调用链，
        # 方便调试时既看到"表面原因"也看到"根本原因"
        raise ToolExecutionError(f"工具 {name} 执行失败") from e

run_tool("calculator")
```

**官方文档**：https://docs.python.org/3/tutorial/errors.html

---

## 8. 常用标准库

### 🔴 `pathlib.Path`（文件路径操作）

```python
from pathlib import Path

# / 运算符被 Path 重载了，用来拼接路径，跨平台自动处理分隔符（Windows 的 \ / Linux 的 /）
data_file = Path(__file__).parent / "data" / "todos.json"

if not data_file.exists():
    data_file.parent.mkdir(parents=True, exist_ok=True)  # 递归创建目录，已存在也不报错

data_file.write_text("hello", encoding="utf-8")
content = data_file.read_text(encoding="utf-8")
```

### 🔴 `json`（序列化/反序列化）

```python
import json

data = {"name": "Alice", "age": 25}
text = json.dumps(data, ensure_ascii=False, indent=2)  # dict -> JSON 字符串
# ensure_ascii=False：允许直接输出中文，而不是转义成 \uXXXX
# indent=2：格式化缩进，方便人类阅读

parsed = json.loads(text)  # JSON 字符串 -> dict
```

### 🔴 `collections.Counter` / `defaultdict`

```python
from collections import Counter, defaultdict

words = ["苹果", "香蕉", "苹果", "橙子", "苹果"]
counts = Counter(words)
print(counts)              # Counter({'苹果': 3, '香蕉': 1, '橙子': 1})
print(counts.most_common(2))  # [('苹果', 3), ('香蕉', 1)] —— 出现次数最多的前 2 个

# defaultdict：访问不存在的 key 时，自动用工厂函数创建默认值，不用先判断 key 在不在
groups = defaultdict(list)
groups["fruit"].append("苹果")  # 不需要先写 if "fruit" not in groups: groups["fruit"] = []
```

### 🟡 `re`（正则表达式）

```python
import re

text = "联系邮箱：test@example.com"
match = re.search(r"[\w.]+@[\w.]+", text)  # 匹配一个邮箱格式的子串
if match:
    print(match.group())  # test@example.com
```

**官方文档**：https://docs.python.org/3/library/collections.html

---

## 9. Pydantic（第三方库，但 Agent 开发里约等于必修）

**是什么**：基于类型注解自动做"数据校验 + 类型转换 + 序列化"的库。FastAPI 的请求/响应模型、LangChain/OpenAI 的工具参数 schema，底层基本都是 Pydantic。

### 🔴 `BaseModel` 基本用法

```python
from pydantic import BaseModel

class TodoItem(BaseModel):
    id: int
    text: str
    done: bool = False  # 有默认值的字段，创建实例时可以不传

# model_validate：接收一个 dict（比如从 JSON 解析出来的），
# 校验字段类型是否匹配，不匹配会尝试转换，实在转不了就抛出校验错误
item = TodoItem.model_validate({"id": "1", "text": "买菜"})
print(item.id, type(item.id))  # 1 <class 'int'> —— 字符串 "1" 被自动转成了 int

# model_dump：反过来，把模型实例转换成 dict，方便存成 JSON
print(item.model_dump())  # {'id': 1, 'text': '买菜', 'done': False}
```

### 🟡 `field_validator`（自定义校验规则）

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    age: int

    @field_validator("age")
    @classmethod
    def check_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError("年龄不能为负数")
        return v

User(age=25)   # 正常
User(age=-1)   # 抛出 pydantic 的校验错误
```

**官方文档**：https://docs.pydantic.dev/latest/

---

## 10. 包与项目结构

**是什么**：Python 怎么组织多个文件、怎么让一个目录被当成"可导入的模块集合"，以及项目依赖/虚拟环境是怎么管理的。

### 🔴 `import` / `__init__.py`

```python
# 目录结构：
# commands/
#   __init__.py      # 空文件即可，它的存在标志着 commands 是一个"包"，可以被 import
#   hello.py          # 里面定义了 def hello(): ...

# main.py 里：
from commands.hello import hello   # 绝对导入：从 commands 包里导入 hello 模块的 hello 函数
from commands import todo          # 导入整个 todo 模块，用 todo.app 访问其中的对象
```

### 🟡 `pyproject.toml` / 虚拟环境

```toml
# pyproject.toml：声明项目名称、依赖列表等元信息，uv/pip 靠它知道要装什么
[project]
name = "pycli-toolbox"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "typer>=0.27.1",
]
```

> 虚拟环境（`.venv/`）的作用：把当前项目的依赖隔离在一个独立目录里，不同项目之间互不影响，避免"这个项目要 requests 1.0，那个项目要 requests 2.0"的冲突。

**官方文档**：https://docs.python.org/3/tutorial/modules.html

---

## 学习建议

1. 每个 🔴 知识点，建议脱离这份笔记，自己重新敲一遍 demo，能默写出来才算过关。
2. 遇到官方文档里的术语看不懂，回到这份笔记里对应的"通俗解释"部分对照理解。
3. 结合 `python-cli-practice` 项目实践：`todo` 用到了第 1/2/8/9/10 点，`weather` 用到了第 1/3 点，后面的 `wordcount`（Week 阶段 4）会用到第 6/8 点。
