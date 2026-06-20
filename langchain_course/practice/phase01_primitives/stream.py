from dotenv import load_dotenv
import os
load_dotenv()
import asyncio

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)


print("\n")

for chunk in llm.stream("What is your name?"):
    print(chunk.content, end="", flush=True)
print("\n")

async def stream_async():
    async for chunk in llm.astream("What is your name?"):
        print(chunk.content, end="", flush=True)
    print("\n")

asyncio.run(stream_async())