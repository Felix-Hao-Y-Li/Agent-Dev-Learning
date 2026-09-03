import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()

model = init_chat_model("deepseek:deepseek-v4-flash")
parser = StrOutputParser()

def slow_a(x): time.sleep(2); return f"slow_a({x})"
def slow_b(x): time.sleep(2); return f"slow_b({x})"

par = RunnableParallel(a=RunnableLambda(slow_a), b=RunnableLambda(slow_b))
t = time.time()
print(par.invoke("hi"))
print("耗时：",time.time()-t)

par2 = {"a":RunnableLambda(slow_a),"b":RunnableLambda(slow_b)}
par2 = RunnableParallel(par2)
t = time.time()
print(type(par2))

zh = ChatPromptTemplate.from_messages([("user", "用中文一句话解释{topic}")]) | model | parser
en = ChatPromptTemplate.from_messages([("user", "Explain {topic} in one English sentence")]) | model | parser
both = RunnableParallel(chinese=zh, english=en)
print(both.invoke({"topic": "vector database"}))

# 预测 4：把 both 接一个后续步骤，后续函数收到的输入是什么类型/结构？
def combine(d: dict) -> str:
    return d["chinese"] + " ||| " + d["english"]
chain = both | RunnableLambda(combine)
print(chain.invoke({"topic": "RAG"}))

