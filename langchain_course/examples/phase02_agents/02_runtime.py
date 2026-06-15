"""
Phase 2 — Topic 2: Runtime (context, store, config & callbacks)
Runtime injects user data and session info into tools automatically.
config= handles tracing labels and callbacks separately.
"""
from pydantic import BaseModel
from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler

llm = ChatOllama(model="llama3.2")

ORDERS = {
    "ORD123": {"status": "Out for delivery.", "amount": 1299},
    "ORD456": {"status": "Delivered on 12 Jun 2026.", "amount": 3499},
    "ORD789": {"status": "Delayed. New date: 16 Jun 2026.", "amount": 899},
}

# ── Context schema ─────────────────────────────────────────────────────────────

class Context(BaseModel):
    user_id: str
    role: str   # "admin" | "customer"

# ── Tools that use runtime.context ────────────────────────────────────────────

def get_order_status(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Get the delivery status of a Slipkart order."""
    order = ORDERS.get(order_id)
    if not order:
        return f"No order found with ID {order_id}."
    return f"[user={runtime.context.user_id}] {order['status']}"


def cancel_order(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Cancel a Slipkart order. Admin only."""
    if runtime.context.role != "admin":
        return "Permission denied. Only admins can cancel orders."
    order = ORDERS.get(order_id)
    if not order:
        return f"No order found with ID {order_id}."
    return f"Order {order_id} cancelled. Refund of ₹{order['amount']} in 3-5 days."


# ── Callback ───────────────────────────────────────────────────────────────────

class SupportLogger(BaseCallbackHandler):
    """Logs LLM and tool events during execution."""

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(f"  [LLM] {serialized['name']} starting...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"  [TOOL] {serialized['name']} | args: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"  [TOOL] result: {output.content}")


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=llm,
    tools=[get_order_status, cancel_order],
    system_prompt="You are a formal Slipkart support agent.",
    context_schema=Context,
)

# ── Example 1: context= injects user data into tools ──────────────────────────

print("=== context injection (customer — cancel denied) ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Cancel my order ORD123."}]},
    context=Context(user_id="U001", role="customer"),
)
print(result["messages"][-1].content)

print()

print("=== context injection (admin — cancel allowed) ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Cancel order ORD123."}]},
    context=Context(user_id="ADMIN01", role="admin"),
)
print(result["messages"][-1].content)

print()

# ── Example 2: config= for callbacks + LangSmith labels ───────────────────────

print("=== config: callbacks + tags + metadata ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Where is order ORD789?"}]},
    context=Context(user_id="U002", role="customer"),
    config={
        "callbacks": [SupportLogger()],
        "tags": ["slipkart", "support"],
        "metadata": {"user_id": "U002", "channel": "mobile"},
        "run_name": "slipkart-status-run",
    },
)
print("Reply:", result["messages"][-1].content)

print()

# ── Example 3: recursion_limit — cap runaway tool loops ───────────────────────

print("=== recursion_limit ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Status of ORD456?"}]},
    context=Context(user_id="U003", role="customer"),
    config={"recursion_limit": 5},
)
print(result["messages"][-1].content)
