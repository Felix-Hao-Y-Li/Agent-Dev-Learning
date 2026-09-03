from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

model = init_chat_model("deepseek:deepseek-chat")

r1 = model.invoke("用一个词回答中国的首都")
print(type(r1),"|",r1.content)
#

r2 = model.batch(["1+1=?","2+2=?"])
print(type(r2),len(r2),[x.content for x in r2])
# 

for chunk in model.stream("从1数到5"):
    print(repr(chunk.content), end = " ")
