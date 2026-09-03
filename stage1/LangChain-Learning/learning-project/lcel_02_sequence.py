from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = init_chat_model("deepseek:deepseek-v4-flash")
prompt = ChatPromptTemplate.from_messages([("system","你是一个只用中文回答的助手，回答控制在20字内"),("user","{topic}")])
parser = StrOutputParser()

chain = prompt | model | parser

# 预测 1： 
print(type(chain))

out = chain.invoke({"topic":"什么是向量数据库"})
print(type(out),"|",out)

outs = chain.batch([{"topic":"Rag"},{"topic":"Function Calling"}])

print(type(outs),"|",outs)

chain2 = prompt | model
print(type(chain2.invoke({"topic": "test"})))