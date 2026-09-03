from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

model = init_chat_model("deepseek:deepseek-v4-flash")
parser = StrOutputParser()

def make_prompt(topic:str) -> str:
    return f"用一句话解释{topic}，控制在20字内"

def shout(text):
    return text.upper() + "🔊"

print(type(RunnableLambda(make_prompt)))

chain = RunnableLambda(make_prompt) | model | parser | RunnableLambda(shout)

out = chain.invoke("什么是向量数据库")
print(type(out),"|",out)

# 预测 3：把第一个 RunnableLambda(...) 换成裸函数 make_prompt，还能跑吗？
chain2 = make_prompt | model | parser
print(chain2.invoke("向量检索"))
# 只要管道里至少有一个真 Runnable，普通函数会被自动转成 RunnableLambda
# 预测 4：裸函数单独能 invoke 吗？下面这行会报错吗？
# print(make_prompt.invoke("test"))