from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)  # 0 = deterministic

# invoke: wait for full reply
response = llm.invoke("Tell me a joke.")
print("invoke:", response.content)  # AIMessage → .content gets the text

# stream: print tokens as they arrive
print("\nstream: ", end="")
for chunk in llm.stream("Count from 1 to 5, one number per line."):
    print(chunk.content, end="", flush=True)

# batch: multiple inputs at once
responses = llm.batch(["Capital of India?", "Capital of Japan?"])
print("\n\nbatch:")
for r in responses:
    print(" -", r.content)
