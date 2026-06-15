from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model

llm = ChatOllama(model="llama3.2", temperature=0)  # 0 = deterministic

# invoke: wait for full reply
response = llm.invoke("Tell me a joke.")
print("invoke:", response.content)

# stream: print tokens as they arrive
print("\nstream: ", end="")
for chunk in llm.stream("Count from 1 to 5, one number per line."):
    print(chunk.content, end="", flush=True)

# batch: multiple inputs, results in same order as inputs
print("\n\nbatch:")
responses = llm.batch(["Capital of India?", "Capital of Japan?"])
for r in responses:
    print(" -", r.content)

# batch_as_completed: results arrive as each finishes (order not guaranteed)
print("\nbatch_as_completed:")
for idx, r in llm.batch_as_completed(["Capital of France?", "Capital of Brazil?", "Capital of Australia?"]):
    print(f"  [{idx}]", r.content)

# init_chat_model: switch providers without changing imports
print("\ninit_chat_model (ollama):")
llm2 = init_chat_model("llama3.2", model_provider="ollama", temperature=0)
print(llm2.invoke("What is 3+3?").content)
