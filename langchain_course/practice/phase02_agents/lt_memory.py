from dotenv import load_dotenv
load_dotenv()
import os

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent,AgentState
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool 
from pydantic import BaseModel


llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)

# ── Store + Context setup ──────────────────────────────────────────────────────

store = InMemoryStore()

class Context(BaseModel):
    user_id: str
    name: str



# ── Tools that read/write the store ───────────────────────────────────────────

@tool
def save_fitness_profile(weight: float, goal: float, runtime: ToolRuntime[Context]) -> str:
    """Save or update the user's current weight and goal weight."""
    uid = runtime.context.user_id
    runtime.store.put(("users", uid), "fitness", {"name": runtime.context.name, "weight": weight, "goal": goal})
    return f"Profile saved. Name: {runtime.context.name}, Weight: {weight}kg, Goal: {goal}kg."


@tool
def get_fitness_profile(runtime: ToolRuntime[Context]) -> str:
    """Get the saved fitness profile of the current user."""
    uid = runtime.context.user_id
    item = runtime.store.get(("users", uid), "fitness")
    if item is None:
        return "No profile found. Please share your weight and goal."
    v = item.value
    return f"Weight: {v['weight']}kg, Goal: {v['goal']}kg"


@tool
def log_workout(date: str, activity: str, runtime: ToolRuntime[Context]) -> str:
    """Log a workout activity for the current user on a given date."""
    uid = runtime.context.user_id
    runtime.store.put(("users", uid), f"workout_{date}", {"activity": activity})
    return f"Logged: {activity} on {date}."


tools = [save_fitness_profile, get_fitness_profile, log_workout]

ctx = Context(user_id="U001", name="Prasad")

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=f"You are a fitness coach assistant. The user's name is {ctx.name}. Always address them by name. Be concise.",
    context_schema=Context,
    store=store,
    checkpointer=InMemorySaver(),   # both short + long term together
)


# ── Example 1: save fitness profile ───────────────────────────────────────────

print("=== save fitness profile ===")
cfg1 = {"configurable": {"thread_id": "u001-session-1"}}


r = agent.invoke(
    {"messages": [{"role": "user", "content": "I weigh 75kg and my goal is 65kg."}]},
    config=cfg1, context=ctx,
)
print(r["messages"][-1].content)

print()



# ── Example 2: log a workout ──────────────────────────────────────────────────

print("=== log workout ===")
r = agent.invoke(
    {"messages": [{"role": "user", "content": "I did 30 mins of running today, 2026-06-21."}]},
    config=cfg1, context=ctx,
)
print(r["messages"][-1].content)

print()

# ── Example 3: new session — store persists ────────────────────────────────────
# New thread_id (new conversation) but SAME store — profile still there

print("=== new session — store persists ===")
cfg2 = {"configurable": {"thread_id": "u001-session-2"}}   # different thread, same store
r = agent.invoke(
    {"messages": [{"role": "user", "content": "How am I doing with my fitness goal?"}]},
    config=cfg2, context=ctx,
)
print(r["messages"][-1].content)

print()

# ── Example 4: direct store operations ────────────────────────────────────────
print("Complete Store Data:",store)
print()

print("=== direct store put/get/search ===")

store.put(("users", "U002"), "fitness", {"weight": 70, "goal": 65})

item = store.get(("users", "U002"), "fitness")
print("get:", item.value)

missing = store.get(("users", "U999"), "fitness")
print("missing:", missing)   # None

results = store.search(("users",))
print("search results:")
for r in results:
    print(f"  {r.namespace} / {r.key} → {r.value}")



store.delete(("users", "U002"), "fitness")
deleted = store.get(("users", "U002"), "fitness")
print("after delete:", deleted)   # None