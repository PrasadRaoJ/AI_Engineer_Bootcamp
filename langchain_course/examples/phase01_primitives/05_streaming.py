import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2", temperature=0)

# --- 1. sync streaming ---
print("--- sync stream ---")
for chunk in llm.stream("Tell me about India in 3 sentences."):
    print(chunk.content, end="", flush=True)  # flush=True prints each token immediately
print()

# --- 2. sync streaming with messages ---
print("\n--- sync stream with messages ---")
messages = [
    SystemMessage("You are a Slipkart customer support agent. Be concise."),
    HumanMessage("What is your return policy?"),
]
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
print()

# --- 3. AIMessageChunk — content vs content_blocks ---
print("\n--- AIMessageChunk content_blocks ---")
chunks = list(llm.stream("What is 2+2?"))
for chunk in chunks:
    if chunk.content:                       # skip empty first/last chunks
        print("content     :", repr(chunk.content))
        print("content_blocks:", chunk.content_blocks)  # normalized: [{"type": "text", "text": ...}]
        break

# --- 4. async streaming ---
print("\n--- async stream ---")

async def stream_async():
    async for chunk in llm.astream("Name 3 famous Indian monuments."):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_async())
