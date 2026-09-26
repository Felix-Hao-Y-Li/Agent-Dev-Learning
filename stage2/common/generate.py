"""
common/generate.py —— RAG 的"生成"一步：把检索到的文档交给大模型，让它据此回答

为什么需要它：
    stage2 之前的几课只做到"检索出几块"就停了。
    但 RAGAs 的忠实度、回答相关性评的是"回答"，没有回答就没法算。
    所以把生成这一步单独补在 common/ 里，评估课和以后的课都能共用。

为什么 prompt 不自己写：
    eval_02 用的测试集是 RGB（arXiv 2309.01431）。RGB 官方仓库的 config/instruction.yaml
    给出了它评测时用的中文 system prompt 和问题模板，这里逐字照搬。
    这样我们的生成方式和数据集作者的设定一致，结果才能和论文口径对得上。

函数索引
    answer_with_contexts() —— 问题 + 文档列表 → 回答字符串

阅读地图
    必读：RGB_SYSTEM / RGB_INSTRUCTION —— 模型被要求怎么用文档
          answer_with_contexts()        —— 生成的全过程只有三步
"""

# [了解] init_chat_model：LangChain v1 统一的"按名字创建聊天模型"入口，stage1 起一直在用。
#        SystemMessage / HumanMessage：分别对应"系统设定"和"用户这一轮说的话"两种消息角色。
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# [核心] RGB 官方中文 system prompt（逐字照搬 config/instruction.yaml 的 zh.system）。
#        它交代了三件事：
#          1) 借助外部文档回答，并提醒文档里可能有噪声；
#          2) 文档里没有答案时，固定输出"文档信息不足……"——这叫"拒答"，防止模型用自己的知识硬编；
#          3) 文档与事实不符时先指出错误（这条是给 RGB 的 _fact 数据集用的，zh_refine 里基本用不到）。
RGB_SYSTEM = (
    "你是一个准确和可靠的人工智能助手，能够借助外部文档回答问题，请注意外部文档可能存在噪声事实性错误。"
    "如果文档中的信息包含了正确答案，你将进行准确的回答。"
    "如果文档中的信息不包含答案，你将生成“文档信息不足，因此我无法基于提供的文档回答该问题。”。"
    "如果部分文档中存在与事实不一致的错误，请先生成“提供文档的文档存在事实性错误。”，并生成正确答案。"
)

# [核心] RGB 官方中文问题模板（逐字照搬 zh.instruction，包括 {DOCS} 后面那个空格）。
RGB_INSTRUCTION = "文档：\n{DOCS} \n\n问题：\n{QUERY}"


def build_generator() -> BaseChatModel:
    """[了解] 造一个生成回答用的 DeepSeek 模型。"""
    # 1. [核心] temperature=0 本意是让同一个输入尽量得到同样的回答。
    #    但 2026-09-26 查证：DeepSeek 默认开启思考模式，思考模式下 temperature 被静默忽略
    #    （api-docs.deepseek.com/guides/thinking_mode："setting these parameters will not trigger
    #    an error but will also have no effect"）。eval_02 两次运行，60 条回答只有 12 条完全相同，
    #    就是这个原因。这里保留参数，只为说明意图；实际并不起作用。
    return init_chat_model("deepseek:deepseek-v4-flash", temperature=0)


def answer_with_contexts(llm: BaseChatModel, question: str, contexts: list[str]) -> str:
    """[核心] 问题 + 检索到的文档 → 回答字符串。"""
    # 1. [核心] 文档之间用换行连接——RGB 官方 evalue.py 里就是 '\n'.join(docs)。
    docs = "\n".join(contexts)
    # 2. [核心] 两条消息：system 定规矩，human 给文档和问题。
    messages = [
        SystemMessage(content=RGB_SYSTEM),
        HumanMessage(content=RGB_INSTRUCTION.format(DOCS=docs, QUERY=question)),
    ]
    # 3. [了解] invoke 返回 AIMessage，回答正文在 .content 里。
    return llm.invoke(messages).content
