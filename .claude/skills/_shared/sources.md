# 权威源查找规则（learn / review / interview 共用）

> 这不是一个 skill，是被三个学习技能引用的参考文件。

## 一、查找顺序（必须自上而下）

1. **官方文档 MCP**（已接入 4 个，优先用它们，最快最准）

   | MCP | 覆盖范围 | 主要工具 |
   |---|---|---|
   | `docs-langchain` | LangChain / LangGraph / LangSmith 的概念、教程、产品文档 | `search_docs_by_lang_chain`（概念检索）、`query_docs_filesystem_docs_by_lang_chain`（用 `rg`/`head`/`cat` 读原文，路径形如 `/oss/python/langchain/agents.mdx`） |
   | `reference-langchain` | LangChain 全部包的 API reference：类、方法、参数、签名 | `search_api`、`get_symbol` |
   | `claude-platform-docs` | Claude API / Anthropic SDK：Messages API、tool use、prompt caching、Agent SDK、模型与定价 | 文档检索 |
   | `claude-code-docs` | Claude Code：skills、hooks、MCP、settings、slash command | 文档检索 |

   查 LangChain 用法时**两个 langchain MCP 一起用**：`docs-langchain` 给"为什么和怎么做"，`reference-langchain` 给准确签名。

2. **该库官方文档站**（WebFetch）。上述 MCP 未覆盖的库（FastAPI、Redis、Milvus、Docker、Prometheus/Grafana、RAGAs 等）走这一档。部分站点提供 `llms.txt` 索引可先取；FastAPI 站点没有 `llms.txt`（2026-09-04 实测 404），直接抓页面。
3. **官方 GitHub release notes / CHANGELOG** — 判断某个 API 是哪个版本引入或废弃时用这个，比文档更准确。
4. **开源学习资料**（AgentGuide 等）。允许用，但输出里**必须标注"非官方，可能滞后"**，且不能作为 API 写法的唯一依据。

## 二、动手前的版本自检（不可跳过）

写任何代码之前，先读当前学习目录的 `pyproject.toml`（必要时 `uv.lock`）确认实际装的版本，不要按记忆假设。

已知的当前环境（`stage1/LangChain-Learning/learning-project/pyproject.toml`，核对于 2026-09-04）：

```
langchain[google-genai]>=1.3.15
langchain-anthropic>=1.6.0
langchain-deepseek>=1.1.0
python-dotenv>=1.2.3
```

尚未安装：`langchain-community`、任何向量库（FAISS / Chroma）、embedding、text splitter、`pypdf`。做 RAG 相关内容时需要先 `uv add`。

**可用模型 id**：`deepseek:deepseek-v4-flash`（注意是连字符；旧的 `deepseek:deepseek-chat` 在 DeepSeek v4 上线后已失效）、`google_genai:gemini-3.7-flash`、`anthropic:claude-sonnet-4-6`。API key 在同目录 `.env`，用 `load_dotenv()` 读取。

## 二点五、环境配置：Claude 指导，用户手动执行（不可代劳）

**凡是改变环境的操作，一律由用户自己敲命令，Claude 只负责给出指导。**用户要通过亲手配置来熟悉工具链，Claude 替他跑掉就等于剥夺了这次练习。

属于"环境配置"、**必须交给用户执行**的操作：

- 建项目、装/删依赖：`uv init`、`uv add`、`uv remove`、`uv sync`、`pip install`
- 安装系统级工具：Docker、Milvus、Redis、Ollama、数据库等
- 下载模型权重（含首次运行脚本时触发的自动下载）
- 写 `.env`、设置环境变量、申请或填写 API key
- 注册 MCP server、改全局配置

**仍然由 Claude 自己做**的操作：写代码文件、读文件、查文档、以及在依赖**已经装好**的前提下 `uv run <脚本>`。

### 指导的标准格式

不要只丢一句"你需要装 X"。每次环境配置的指导必须包含这四项：

1. **要装什么、为什么** —— 这个包/工具在本次学习里承担什么角色，不装会卡在哪一步。
2. **精确到可以直接复制的命令** —— 写明在哪个目录下执行。多条命令分行列出，不要合并成一长串。
3. **预期结果与代价** —— 大概装多久、下载多大、会不会拉起 GPU/CPU 版本的差异；会触发模型下载的要提前说明体积和缓存位置。
4. **怎么验证成功** —— 给一条一次性的校验命令（例如打印版本号），让用户自己确认装好了。

给完指导就**停下来等用户回话**，不要自己往下跑。用户确认装好之后再继续写代码、跑 demo。

如果用户明确说"这次你直接装吧"，那就照办——这是用户对单次操作的授权，不改变默认规则。

## 三、已知的过时说法（引用前必须重查一次）

以下每条都核对于 **2026-09-04**。时间一长就可能再次变化，输出前用上面的 MCP 重新确认，不要直接抄这张表。

| 已过时 | 现行 | 依据 |
|---|---|---|
| LCEL / LangChain Expression Language | LangChain v1 不再以链式表达式为中心；组合逻辑交给 `create_agent` 与 LangGraph | `LCEL` 一词在 `/oss` 整个 Python 文档中 0 命中 |
| `langgraph.prebuilt.create_react_agent` | `from langchain.agents import create_agent` | `/oss/python/releases/langgraph-v1` 明确标注 deprecated |
| "ReAct 框架"作为 agent 实现方式 | **Agent = Model + Harness**，harness 由 prompt + tools + middleware 组成 | `/oss/python/langchain/agents.mdx` |
| 先检索再回答的固定 RAG 管道 | **Agentic RAG**：检索是 agent 的一个 tool，由模型决定何时检索、检索几次 | `/oss/python/deepagents/retrieval.mdx` |
| AutoGen / CrewAI 作为多智能体主线 | LangGraph / LangChain 的四种模式：Subagents、Handoffs、Skills、Router（+ Custom workflow）；上层有 Deep Agents | `/oss/python/langchain/multi-agent/index.mdx` |

## 四、旧概念怎么处理

用一到两句话交代清楚三件事就够：**它是什么、为什么被取代、面试被问到怎么答**。不要围绕已废弃的 API 写练习代码，不要在 demo 里使用它。

在 `CLAUDE.md` 路线图和 `study-log/plan.md` 里，过时技术**保留但用删除线标出**（`~~旧技术~~`），不删除——用户要保留原路线图的历史面貌。如果用户下达学习目标时用了被划掉的旧说法（例如"今天学 LCEL"），按同一位置现行的技术来教，并顺带说明这个替换关系。

## 四点五、待接入的运维类 MCP（用到时再加）

以下是**操作类**而非文档类 MCP，需要对应服务实际在跑才有意义，等到相应阶段再接：

| MCP | 何时接 | 备注 |
|---|---|---|
| Milvus（Zilliz 官方） | Week 2 部署 Milvus 后 | 需要 Milvus 实例（默认 `localhost:19530`） |
| Redis（Redis 官方） | Week 4 接入缓存后 | 用于查看 cache/queue 实际状态 |
| Grafana（`grafana/mcp-grafana` 官方） | Week 5 搭好面板后 | 可查 dashboard、查询 Prometheus 数据源 |
| LangSmith Remote MCP（`https://api.smith.langchain.com/mcp`，OAuth） | Week 5 接入 LangSmith 后 | 查 trace、dataset、experiment |

接入命令形如 `claude mcp add --transport http --scope user <名字> <URL>`，接完用 `claude mcp list` 确认 Connected。

## 五、输出时的标注要求

- 开头一行写明本次依据的**文档来源 + 版本 + 查证日期**，例如：
  `依据：docs.langchain.com /oss/python/langchain/agents（langchain 1.3.15，查证于 2026-09-04）`
- 引用具体结论时给出页面路径或链接，让用户能自己翻回去核对。
- 如果查不到权威依据，直说"官方文档未覆盖，以下是社区做法/我的推断"，不要含混带过。
