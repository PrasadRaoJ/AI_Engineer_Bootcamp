from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

"""
Phase 2 — Topic 3: Context Engineering
Shape what the model sees (prompt, tools, messages) using middleware.
"""
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call
from langchain.agents.middleware.types import ModelRequest
from langchain_core.tools import tool

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

# ── Context schema ─────────────────────────────────────────────────────────────

class Context(BaseModel):
    user_name: str
    role: str       # "admin" | "customer"
    language: str   # "English" | "Hindi"

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_order_status(order_id: str) -> str:
    """Get the current delivery status of a Slipkart order."""
    statuses = {
        "ORD123": "Out for delivery. Expected by 6 PM today.",
        "ORD456": "Delivered on 12 Jun 2026.",
    }
    return statuses.get(order_id, f"Order {order_id} not found.")

get_order_status.metadata = {"roles": ["admin", "customer"]}


@tool
def cancel_order(order_id: str) -> str:
    """Cancel a Slipkart order. Admin only."""
    return f"Order {order_id} cancelled. Refund in 3-5 days."

cancel_order.metadata = {"roles": ["admin"]}


# ── Example 1: @dynamic_prompt — personalize system prompt per user ────────────

@dynamic_prompt
def personalized_prompt(request: ModelRequest[Context]) -> str:
    name = request.runtime.context.user_name
    lang = request.runtime.context.language
    return (
        f"You are a formal Slipkart support agent. "
        f"Address the customer as {name}. Reply in {lang}."
    )


agent_dynamic = create_agent(
    model=llm,
    tools=[get_order_status],
    context_schema=Context,
    middleware=[personalized_prompt],   # replaces system_prompt=
)

print("=== @dynamic_prompt — personalized system prompt ===")
result = agent_dynamic.invoke(
    {"messages": [{"role": "user", "content": "What is your return policy?"}]},
    context=Context(user_name="Ravi", role="customer", language="English"),
)
print(result["messages"][-1].content)

print()

# ── Example 2: @wrap_model_call — filter tools by role ────────────────────────

@wrap_model_call
def filter_tools_by_role(request: ModelRequest[Context], handler):
    role = request.runtime.context.role
    filtered = [t for t in request.tools if role in t.metadata.get("roles", [role])]
    request = request.override(tools=filtered)
    print(f"  [middleware] tools visible to LLM: {[t.name for t in request.tools]}")
    return handler(request)


agent_filtered = create_agent(
    model=llm,
    tools=[get_order_status, cancel_order],
    context_schema=Context,
    middleware=[filter_tools_by_role],
    system_prompt="You are a Slipkart support agent.",
)

print("=== @wrap_model_call — tool filtering by role ===")

print("-- customer (cancel_order filtered out) --")
result = agent_filtered.invoke(
    {"messages": [{"role": "user", "content": "What is the status of ORD123?"}]},
    context=Context(user_name="Priya", role="customer", language="English"),
)
print(result["messages"][-1].content)

print()

print("-- admin (all tools available) --")
result = agent_filtered.invoke(
    {"messages": [{"role": "user", "content": "Cancel order ORD456."}]},
    context=Context(user_name="Admin01", role="admin", language="English"),
)
print(result["messages"][-1].content)
