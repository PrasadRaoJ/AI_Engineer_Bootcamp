from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Literal

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

ORDERS = {
    "ORD123": "Out for delivery. Expected by 6 PM today.",
    "ORD456": "Delivered on 12 Jun 2026.",
    "ORD789": "Delayed. New expected date: 16 Jun 2026.",
}

# --- 1. Basic tool — docstring as description ---

@tool
def get_order_status(order_id: str) -> str:
    """Returns the current delivery status of a Slipkart order given its order ID."""
    return ORDERS.get(order_id, f"Order {order_id} not found.")


# --- 2. args_schema — Pydantic for constrained inputs ---

class CancelInput(BaseModel):
    order_id: str = Field(description="The Slipkart order ID, e.g. ORD123")
    reason: Literal["changed_mind", "wrong_item", "delay"] = Field(
        description="Reason for cancellation"
    )

@tool(args_schema=CancelInput)
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a Slipkart order. Reason must be one of: changed_mind, wrong_item, delay."""
    return f"Order {order_id} cancelled. Reason: {reason}. Refund in 3-5 days."


# --- 3. return_direct — skip final LLM summarization ---

@tool(return_direct=True)
def get_store_hours() -> str:
    """Returns Slipkart customer support hours."""
    return "Slipkart support is available Mon–Sat, 9 AM to 9 PM IST."


# --- show tool schema the model sees ---
print("=== tool schemas ===")
print("get_order_status args:", get_order_status.args)
print("cancel_order args:", cancel_order.args)
print("get_store_hours return_direct:", get_store_hours.return_direct)

# --- TOOL_MAP for safe dispatch ---
ALL_TOOLS = [get_order_status, cancel_order, get_store_hours]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}  # name → function lookup

llm_with_tools = llm.bind_tools(ALL_TOOLS)

# --- 4. Full tool-calling loop ---

print("\n=== tool-calling loop ===")
messages = [
    SystemMessage("You are a formal Slipkart support agent. Be professional and friendly."),
    HumanMessage("Please cancel my order ORD123, I changed my mind."),
]

response = llm_with_tools.invoke(messages)

if response.tool_calls:
    messages.append(response)                                        # AIMessage added ONCE
    for tc in response.tool_calls:
        print(f"tool called: {tc['name']} | args: {tc['args']}")
        result = TOOL_MAP[tc["name"]].invoke(tc["args"])             # dispatch via TOOL_MAP
        messages.append(ToolMessage(result, tool_call_id=tc["id"]))  # add tool result

    final = llm_with_tools.invoke(messages)
    print("\nAI:", final.content)
else:
    print("AI (no tool):", response.content)
