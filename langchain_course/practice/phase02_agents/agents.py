from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent
import os

llm = init_chat_model(os.getenv("LLM_MODEL", "qwen3.5:2b"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)
# llama.cpp: LLM_PROVIDER=openai  LLM_MODEL=local  +  OPENAI_BASE_URL=http://localhost:8080/v1  OPENAI_API_KEY=none  (start: ~/models/switch-model.sh qwen3-4b)


ORDERS = {
    "ORD123": "Out for delivery. Expected by 6 PM today.",
    "ORD456": "Delivered on 12 Jun 2026.",
    "ORD789": "Delayed. New expected date: 16 Jun 2026.",
}


@tool
def get_order_status(order_id: str) -> str:
    """Get the current delivery status of a Slipkart order."""
    return ORDERS.get(order_id, f"No order found with ID {order_id}.")

@tool
def cancel_order(order_id: str) -> str:
    """Cancel a Slipkart order and initiate a refund."""
    if order_id in ORDERS:
        return f"Order {order_id} has been cancelled. Refund in 3-5 business days."
    return f"No order found with ID {order_id}."

agent = create_agent( model = llm, tools = [get_order_status, cancel_order],
                     system_prompt = "You are a helpful customer support agent for Slipkart, " \
                     "an online shopping platform. You can assist customers with checking their order status "
                     "and cancelling orders if needed. Always provide clear and concise responses to the customers' inquiries.")

result = agent.invoke({"messages":[{"role":"user", 
                                    "content":"I want to check the status of my order ORD123."}]})

print(result)

print("\n=== call flow ===")
for i, msg in enumerate(result["messages"]):
    name = type(msg).__name__
    if name == "HumanMessage":
        print(f"[{i}] USER       : {msg.content}")
    elif name == "AIMessage":
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[{i}] LLM DECIDES: call {tc['name']}({tc['args']})")
        else:
            print(f"[{i}] LLM REPLY  : {msg.content}")
    elif name == "ToolMessage":
        print(f"[{i}] TOOL RESULT: {msg.content}")

print()
print("=== final answer ===")
print(result["messages"][-1].content)

#------------------------------------- Tool Runtime Example & permissions -------------------------------------#
