# Phase 0 — verify LangChain + Ollama works
# Run: python examples/phase00_setup/hello_trace.py
# Expected: prints 'pong'

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
response = llm.invoke("Reply with the single word: pong")
print(response.content)
