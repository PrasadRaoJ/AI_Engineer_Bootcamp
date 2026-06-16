from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

# Phase 0 — verify LangChain + Ollama works
# Run: python examples/phase00_setup/hello_trace.py
# Expected: prints 'pong'


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke("Reply with the single word: pong")
print(response.content)
