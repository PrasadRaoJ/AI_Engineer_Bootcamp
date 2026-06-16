from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# ── Ollama (local) ────────────────────────────────────────────────────────────
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

# ── Groq (cloud, fast, free tier) ────────────────────────────────────────────
# pip install langchain-groq
# set GROQ_API_KEY in .env

print("\n--- Groq ---")
groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# invoke
response = groq.invoke("Tell me a joke.")
print("invoke:", response.content)

# stream
print("\nstream: ", end="")
for chunk in groq.stream("Count from 1 to 5, one number per line."):
    print(chunk.content, end="", flush=True)

# init_chat_model with groq
print("\n\ninit_chat_model (groq):")
llm3 = init_chat_model("llama-3.3-70b-versatile", model_provider="groq", temperature=0)
print(llm3.invoke("What is 3+3?").content)

# ── OpenAI (cloud, paid) ──────────────────────────────────────────────────────
# pip install langchain-openai
# set OPENAI_API_KEY in .env

print("\n--- OpenAI ---")
oai = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# invoke
response = oai.invoke("Tell me a joke.")
print("invoke:", response.content)

# stream
print("\nstream: ", end="")
for chunk in oai.stream("Count from 1 to 5, one number per line."):
    print(chunk.content, end="", flush=True)

# init_chat_model with openai
print("\n\ninit_chat_model (openai):")
llm4 = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
print(llm4.invoke("What is 3+3?").content)
