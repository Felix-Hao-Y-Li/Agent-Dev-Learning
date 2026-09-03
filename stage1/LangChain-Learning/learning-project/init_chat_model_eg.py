from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

model = init_chat_model("google_genai:gemini-3.7-flash")
# invoke
# response = model.invoke("Why do parrots talk?")

# stream()
# for chunk in model.stream("Why do parrots have colorful feathers?"):
#     print(chunk.text, end="", flush=True)

# batch
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
], config={"max_concurrency": 5})

for res in responses:
    print(res.text)