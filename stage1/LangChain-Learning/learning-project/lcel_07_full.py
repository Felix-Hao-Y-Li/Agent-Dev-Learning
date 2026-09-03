from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()

model = init_chat_model("deepseek:deepseek-v4-flash")
parser = StrOutputParser()

def clean_word(raw: str) ->str:
    return raw.strip().lower()

prompt = ChatPromptTemplate.from_messages([("system", "你是英语词典。用户给一个单词，你输出：释义 / 词性 / 一个例句。"),
                                           ("user", "单词: {word} (长度 {length} 个字符)")])

chain = (RunnableLambda(clean_word)) | {"word": RunnablePassthrough(), "length": RunnableLambda(lambda word: len(word))} | prompt | model | parser

print(chain.invoke("  Serendipity  "))

print("--- stream ---")
for chunk in chain.stream("  Ephemeral "):
    print(chunk, end="", flush=True)
print()

print(chain.batch(["  Nostalgia ", "RESILIENCE"]))