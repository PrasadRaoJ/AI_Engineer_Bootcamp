from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2", temperature=0)

# --- 1. SystemMessage sets the persona ---
messages = [
    SystemMessage("You are a formal customer service representative for Slipkart. Be professional and Friendly."),
    HumanMessage("My order has not arrived yet. It has been 5 days."),
]
response = llm.invoke(messages)
print("AI:", response.content)

# --- 2. Multi-turn conversation (manual history) ---
history = [SystemMessage("You are a helpful assistant.")]

history.append(HumanMessage("My name is JP."))
history.append(llm.invoke(history))  # AIMessage stored in history

history.append(HumanMessage("What's my name?"))
reply = llm.invoke(history)          # model remembers context
print("\nmemory:", reply.content)
# expected: "Your name is JP."
