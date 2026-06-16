from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

"""
Phase 2 — Topic 1: Agents
create_agent replaces the manual tool-call loop from Phase 1.
"""
from pydantic import BaseModel
from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

# ── Tools ─────────────────────────────────────────────────────────────────────

ORDERS = {
    "ORD123": "Out for delivery. Expected by 6 PM today.",
    "ORD456": "Delivered on 12 Jun 2026.",
    "ORD789": "Delayed. New expected date: 16 Jun 2026.",
}


def get_order_status(order_id: str) -> str:
    """Get the current delivery status of a Slipkart order."""
    return ORDERS.get(order_id, f"No order found with ID {order_id}.")


def cancel_order(order_id: str) -> str:
    """Cancel a Slipkart order and initiate a refund."""
    if order_id in ORDERS:
        return f"Order {order_id} has been cancelled. Refund in 3-5 business days."
    return f"No order found with ID {order_id}."


# ── Example 1: basic .invoke() ────────────────────────────────────────────────

print("=== invoke() ===")
agent = create_agent(
    model=llm,
    tools=[get_order_status, cancel_order],
    system_prompt="You are a formal Slipkart support agent. Be concise.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Where is ORD123?"}]})
print(result["messages"][-1].content)

print()

# ── Example 2: all messages — see the full loop ────────────────────────────────

print("=== all messages ===")
result = agent.invoke({"messages": [{"role": "user", "content": "Cancel order ORD456 please."}]})
for msg in result["messages"]:
    label = type(msg).__name__
    content = msg.content if msg.content else "<tool call>"
    print(f"[{label}] {content[:80]}")

print()

# ── Example 3: per-call context injection ─────────────────────────────────────

print("=== context_schema + ToolRuntime ===")

class Context(BaseModel):
    user_id: str
    role: str   # "admin" or "customer"


def cancel_order_ctx(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Cancel a Slipkart order. Requires admin role."""
    if runtime.context.role != "admin":
        return f"Permission denied. User {runtime.context.user_id} cannot cancel orders."
    return f"Order {order_id} cancelled by admin {runtime.context.user_id}."


agent_ctx = create_agent(
    model=llm,
    tools=[cancel_order_ctx],
    system_prompt="You are a Slipkart support agent.",
    context_schema=Context,
)

# customer — should be denied
result = agent_ctx.invoke(
    {"messages": [{"role": "user", "content": "Cancel order ORD123."}]},
    context=Context(user_id="U001", role="customer"),
)
print("Customer:", result["messages"][-1].content)

print()

# ── Example 4: .stream() with updates ─────────────────────────────────────────

print("=== stream() ===")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the status of ORD789?"}]},
    stream_mode="updates",
):
    if "model" in chunk:
        msg = chunk["model"]["messages"][-1]
        if msg.content:                       # skip empty tool-call AIMessage
            print(msg.content)
