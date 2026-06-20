from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
import os
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

# llm = init_chat_model("local", model_provider="openai", base_url="http://localhost:8080/v1", api_key="none", temperature=0)
llm = init_chat_model("llama3.2", model_provider="ollama", temperature=0)



ORDERS = {
    "ORD123": {"status": "Out for delivery.", "amount": 1299},
    "ORD456": {"status": "Delivered on 12 Jun 2026.", "amount": 3499},
    "ORD789": {"status": "Delayed. New date: 16 Jun 2026.", "amount": 899},
}


# Context schema

class Context(BaseModel):
    user_id: str
    role: str   # "admin" | "customer"



@tool
def get_order_status(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Get the delivery status of a Slipkart order."""
    order = ORDERS.get(order_id)
    if not order:
        return f"No order found with ID {order_id}."
    return f"[user={runtime.context.user_id}] {order['status']}"

@tool
def cancel_order(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Cancel a Slipkart order. Admin only."""
    if runtime.context.role != "admin":
        return "Permission denied. Only admins can cancel orders."
    order = ORDERS.get(order_id)
    if not order:
        return f"No order found with ID {order_id}."
    return f"Order {order_id} cancelled. Refund of ₹{order['amount']} in 3-5 days."



agent = create_agent(
    model=llm,
    tools=[get_order_status, cancel_order],
    system_prompt="You are a helpful customer support agent for Slipkart, ",
    context_schema=Context,
)


result = agent.invoke(
    {"messages": [{
        "role":"user",
        #"content":"I want to check the status of my order ORD123.",
        "content":"I want to Cancel my order ORD123.",
    }]},
    context={"user_id": "user_123", "role": "customer"}

)

def print_flow(result):
    for i, msg in enumerate(result["messages"]):
        name = type(msg).__name__
        if name == "HumanMessage":
            print(f"[{i}] USER        : {msg.content}")
        elif name == "AIMessage":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[{i}] LLM DECIDES : call {tc['name']}({tc['args']})")
            else:
                print(f"[{i}] LLM REPLY   : {msg.content[:80]}")
        elif name == "ToolMessage":
            print(f"[{i}] TOOL RESULT : {msg.content}")
    print()

print("=== customer cancel ===")
print_flow(result)

result = agent.invoke(
    {"messages": [{
        "role":"user",
        "content":"check the status of order ORD123",
    }]},
    context={"user_id": "Admin123", "role": "admin"}
)

print("=== admin status ===")
print_flow(result)

print(result['messages'][-1].content)
print()