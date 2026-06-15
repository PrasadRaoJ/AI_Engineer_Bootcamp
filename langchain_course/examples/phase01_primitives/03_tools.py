from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

llm = ChatOllama(model="llama3.2", temperature=0)

# --- define tools ---

@tool
def get_order_status(order_id: str) -> str:
    """Returns the current delivery status of a Slipkart order given its order ID."""
    statuses = {
        "ORD123": "Out for delivery. Expected by 6 PM today.",
        "ORD456": "Delivered on 12 Jun 2026.",
        "ORD789": "Delayed. New expected date: 16 Jun 2026.",
    }
    return statuses.get(order_id, f"Order {order_id} not found.")

@tool
def cancel_order(order_id: str) -> str:
    """Cancels a Slipkart order given its order ID."""
    return f"Order {order_id} has been successfully cancelled. Refund will be processed in 3-5 days."

# --- bind tools to model ---
llm_with_tools = llm.bind_tools([get_order_status, cancel_order])
TOOL_MAP = {t.name: t for t in [get_order_status, cancel_order]}  # name → function lookup

# --- tool-calling loop ---
# FLOW:
# Step 1 — user sends a message
# Step 2 — model decides which tool to call and returns tool_calls (not text)
# Step 3 — we run the actual Python function with the model's args
# Step 4 — we send the result back as a ToolMessage
# Step 5 — model reads the result and gives the final text answer

messages = [
    SystemMessage("You are a formal customer service representative for Slipkart. Be professional and friendly."),
    #HumanMessage("What is the status of my order ORD123?"),
    HumanMessage("Please cancel my order ORD123."),
]

response = llm_with_tools.invoke(messages)  # step 2: model returns tool_call

if response.tool_calls:
    messages.append(response)                                          # step 4a: AIMessage added ONCE
    for tc in response.tool_calls:
        print(f"tool called: {tc['name']} with args {tc['args']}")
        result = TOOL_MAP[tc["name"]].invoke(tc["args"])               # step 3: run via TOOL_MAP
        messages.append(ToolMessage(result, tool_call_id=tc["id"]))    # step 4b: add tool result

    final = llm_with_tools.invoke(messages)  # step 5: model gives final answer
    print("\nAI:", final.content)
    # expected: "Your order ORD123 is out for delivery. Expected by 6 PM today."
else:
    print("AI:", response.content)  # model answered directly, no tool needed
