from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel

p = RunnablePassthrough()
print(p.invoke({"question":"什么是RAG"}))

chain = RunnableParallel(
    original=RunnablePassthrough(),
    length=RunnableLambda(lambda x: len(x)),
)
print(chain.invoke("hello"))

print(chain.invoke("hello"))

chain2 = RunnablePassthrough.assign(upper = RunnableLambda(lambda d: d["text"].upper()))
print(chain2.invoke({"text":"abc","id":1}))


# 预测 4：这里 question 分支拿到的输入是整个 dict 还是某个字段？会报错吗？
chain3 = {
    "question": RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split())),
}
print(RunnableParallel(chain3).invoke("the quick brown fox"))