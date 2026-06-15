"""
Phase 2 — Topic 5: Long-term Memory
Store persists user data across threads and sessions — unlike checkpointer which
only keeps message history within one thread.
"""
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

llm = ChatOllama(model="llama3.2", temperature=0)

# ── Store + Context setup ──────────────────────────────────────────────────────

store = InMemoryStore()

# Pre-populate with Ravi's profile
store.put(("users", "U001"), "profile", {"name": "Ravi", "city": "Nellore", "vip": True})

class Context(BaseModel):
    user_id: str

# ── Tools that read/write the store ───────────────────────────────────────────

@tool
def get_my_profile(runtime: ToolRuntime[Context]) -> str:
    """Get the saved profile of the current user."""
    item = runtime.store.get(("users", runtime.context.user_id), "profile")
    if item is None:
        return f"No profile found for user {runtime.context.user_id}."
    p = item.value
    return f"Name: {p['name']}, City: {p['city']}, VIP: {p['vip']}"


@tool
def save_city(city: str, runtime: ToolRuntime[Context]) -> str:
    """Save or update the city for the current user."""
    uid = runtime.context.user_id
    item = runtime.store.get(("users", uid), "profile")
    profile = item.value if item else {}
    profile["city"] = city
    runtime.store.put(("users", uid), "profile", profile)
    return f"City updated to {city} for user {uid}."


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=llm,
    tools=[get_my_profile, save_city],
    system_prompt="You are a Slipkart support agent. Be concise.",
    context_schema=Context,
    store=store,
    checkpointer=InMemorySaver(),   # both short + long term together
)

cfg = {"configurable": {"thread_id": "u001-session-1"}}
ctx = Context(user_id="U001")

# ── Example 1: read from store ─────────────────────────────────────────────────

print("=== get profile ===")
r = agent.invoke({"messages": [{"role": "user", "content": "What is my profile?"}]}, config=cfg, context=ctx)
print(r["messages"][-1].content)

print()

# ── Example 2: write to store ──────────────────────────────────────────────────

print("=== update city ===")
r = agent.invoke({"messages": [{"role": "user", "content": "Update my city to Hyderabad."}]}, config=cfg, context=ctx)
print(r["messages"][-1].content)

print()

# ── Example 3: long-term persists across threads ───────────────────────────────
# New thread_id (new conversation) but SAME store — city should still be Hyderabad
# Using qwen3.5:2b which handles tool calls more reliably for this query

print("=== new thread — store persists ===")
agent_q = create_agent(
    model=ChatOllama(model="qwen3.5:2b", temperature=0),
    tools=[get_my_profile, save_city],
    system_prompt="You are a Slipkart support agent. Be concise.",
    context_schema=Context,
    store=store,
    checkpointer=InMemorySaver(),
)
cfg_new = {"configurable": {"thread_id": "u001-session-2"}}   # different thread, same store
r = agent_q.invoke({"messages": [{"role": "user", "content": "Look up my profile and tell me my city."}]}, config=cfg_new, context=ctx)
print(r["messages"][-1].content)

print()

# ── Example 4: direct store operations ────────────────────────────────────────

print("=== direct store put/get/search ===")

store.put(("users", "U002"), "profile", {"name": "Priya", "city": "Mumbai", "vip": False})

# get
item = store.get(("users", "U002"), "profile")
print("get:", item.value)

# get missing
missing = store.get(("users", "U999"), "profile")
print("missing:", missing)   # None

# search all users
results = store.search(("users",))
print("search results:")
for r in results:
    print(f"  {r.namespace} / {r.key} → {r.value}")
