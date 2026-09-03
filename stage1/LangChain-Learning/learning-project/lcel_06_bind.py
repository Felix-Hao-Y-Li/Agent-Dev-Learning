from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = init_chat_model("deepseek:deepseek-v4-flash")
prompt = ChatPromptTemplate.from_messages([("user", "列举3个{topic}的例子，用顿号分隔")])
parser = StrOutputParser()

# 预测 1：bound 的类型是 ChatModel 还是别的？
bound = model.bind(stop=["、"])
print(type(bound))

# 预测 2：加了 stop=["、"] 后，输出会被截断成什么样？
chain = prompt | bound | parser
print(chain.invoke({"topic": "向量数据库"}))

# 预测 3：不加 stop 的对照
chain_full = prompt | model | parser
print(chain_full.invoke({"topic": "向量数据库"}))