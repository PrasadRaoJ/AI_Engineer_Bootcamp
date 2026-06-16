from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

# Phase 0 — verify LangChain + Ollama works
# Run: python examples/phase00_setup/hello_trace.py
# Expected: prints 'pong'


llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2
response = llm.invoke("Reply with the single word: pong")
print(response.content)
