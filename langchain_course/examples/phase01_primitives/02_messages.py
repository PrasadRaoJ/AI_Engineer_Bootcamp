from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

llm = ChatOllama(model="llama3.2", temperature=0)

# --- 1. SystemMessage sets the persona ---
messages = [
    SystemMessage("You are a formal customer service representative for Slipkart. Be professional and Friendly."),
    HumanMessage("My order has not arrived yet. It has been 5 days."),
]
response = llm.invoke(messages)
print("AI:", response.content)

# --- 2. usage_metadata — token counts on every AIMessage ---
print("\nusage_metadata:", response.usage_metadata)
# {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}

# --- 3. Multi-turn conversation (manual history) ---
history = [SystemMessage("You are a helpful assistant.")]

history.append(HumanMessage("My name is JP."))
history.append(llm.invoke(history))  # AIMessage stored in history

history.append(HumanMessage("What's my name?"))
reply = llm.invoke(history)          # model remembers context
print("\nmemory:", reply.content)

# --- 4. ToolMessage fields: content, tool_call_id, artifact ---
tool_msg = ToolMessage(
    content="Order ORD123 is out for delivery.",  # text the model sees
    tool_call_id="abc123",                         # must match AIMessage tool call id
    artifact={"eta": "6pm", "carrier": "BlueDart"},  # raw data — invisible to model
)
print("\nToolMessage content:", tool_msg.content)
print("ToolMessage artifact:", tool_msg.artifact)  # for your code only

# --- 5. HumanMessage name field ---
hm = HumanMessage("Hello, I need help.", name="JP")
print("\nHumanMessage name:", hm.name)
