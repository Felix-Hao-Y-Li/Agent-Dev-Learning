from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

def build_chain() -> Runnable:
    """组装 prompt | model 的 LCEL chain，返回一个可执行的 Runnable。"""
    model = init_chat_model("deepseek:deepseek-chat")
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{question}")
    ])
    chain = prompt | model
    return chain
def ask(chain: Runnable, question : str) -> str:
    """用chain回答一个问题，返回模型输出的纯文本内容"""
    result = chain.invoke({"question":question})
    return result.content

if __name__ == "__main__":
    load_dotenv()
    chain = build_chain()
    answer = ask(chain,"你好，请用一句话介绍你自己")
    print(answer)