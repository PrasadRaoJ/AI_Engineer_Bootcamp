from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

"""
Phase 2 — Topic 4: Short-term Memory
InMemorySaver + thread_id keeps conversation alive across turns.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, wrap_model_call, SummarizationMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import RemoveMessage, AIMessage

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

# ── Example 1: basic multi-turn memory ────────────────────────────────────────

print("=== multi-turn memory ===")

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "session-001"}}

r1 = agent.invoke({"messages": [{"role": "user", "content": "My name is Ravi."}]}, config=cfg)
print("Turn 1:", r1["messages"][-1].content[:80])

r2 = agent.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=cfg)
print("Turn 2:", r2["messages"][-1].content[:80])

# different thread — no memory
r3 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config={"configurable": {"thread_id": "session-999"}},
)
print("New thread:", r3["messages"][-1].content[:80])

print()

# ── Example 2: @before_model trim — keep last N messages ──────────────────────

print("=== @before_model trim ===")

@before_model
def trim_old_messages(state: AgentState, runtime):
    messages = state["messages"]
    if len(messages) > 4:                               # keep last 4 only
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-4]]}

agent_trim = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
    middleware=[trim_old_messages],
)

cfg2 = {"configurable": {"thread_id": "trim-session"}}
agent_trim.invoke({"messages": [{"role": "user", "content": "My name is Ravi."}]}, config=cfg2)
agent_trim.invoke({"messages": [{"role": "user", "content": "I work at Infosys."}]}, config=cfg2)
agent_trim.invoke({"messages": [{"role": "user", "content": "I live in Nellore."}]}, config=cfg2)

r = agent_trim.invoke({"messages": [{"role": "user", "content": "What do you know about me?"}]}, config=cfg2)
print(r["messages"][-1].content[:200])

print()

# ── Example 3: SummarizationMiddleware — compress history ─────────────────────

print("=== SummarizationMiddleware ===")

summarizer = SummarizationMiddleware(
    model=llm,
    trigger=("messages", 5),    # summarize when > 5 messages
    keep=("messages", 2),       # keep last 2 after summarizing
)

agent_summ = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
    middleware=[summarizer],
)

cfg3 = {"configurable": {"thread_id": "summ-session"}}
agent_summ.invoke({"messages": [{"role": "user", "content": "My name is Ravi."}]}, config=cfg3)
agent_summ.invoke({"messages": [{"role": "user", "content": "I work at Infosys."}]}, config=cfg3)
agent_summ.invoke({"messages": [{"role": "user", "content": "I live in Nellore."}]}, config=cfg3)

r = agent_summ.invoke({"messages": [{"role": "user", "content": "What do you know about me?"}]}, config=cfg3)
print(r["messages"][-1].content[:200])

print()

# ── Example 4: state_schema — rate limiting with turn_count ───────────────────

print("=== state_schema (rate limiting) ===")

FREE_TURN_LIMIT = 2

class CustomState(AgentState):
    turn_count: int = 0
    user_plan: str = "free"   # "free" | "pro"

@wrap_model_call
def rate_limit(request: ModelRequest[CustomState], handler):
    if request.state.get("turn_count", 0) >= FREE_TURN_LIMIT and request.state.get("user_plan", "free") == "free":
        return AIMessage(content="Free limit reached. Upgrade to pro.")
    return handler(request)

agent_state = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
    state_schema=CustomState,
    middleware=[rate_limit],
)

cfg4 = {"configurable": {"thread_id": "state-session"}}

r1 = agent_state.invoke(
    {"messages": [{"role": "user", "content": "What is Python?"}], "turn_count": 1},
    config=cfg4,
)
print(f"Turn {r1['turn_count']}: {r1['messages'][-1].content[:80]}")

r2 = agent_state.invoke(
    {"messages": [{"role": "user", "content": "What is LangChain?"}], "turn_count": r1["turn_count"] + 1},
    config=cfg4,
)
print(f"Turn {r2['turn_count']}: {r2['messages'][-1].content[:80]}")

print()

# ── Example 5: REMOVE_ALL_MESSAGES — clear thread history ─────────────────────

print("=== REMOVE_ALL_MESSAGES ===")

agent_clear = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),
)

cfg5 = {"configurable": {"thread_id": "clear-session"}}
agent_clear.invoke({"messages": [{"role": "user", "content": "My name is Ravi."}]}, config=cfg5)

r = agent_clear.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=cfg5)
print("Before clear:", r["messages"][-1].content[:80])

# clear all history — one line
agent_clear.update_state(cfg5, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})

r = agent_clear.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=cfg5)
print("After clear:", r["messages"][-1].content[:80])
