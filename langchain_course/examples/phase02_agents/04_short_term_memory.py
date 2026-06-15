"""
Phase 2 — Topic 4: Short-term Memory
InMemorySaver + thread_id keeps conversation alive across turns.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, SummarizationMiddleware
from langchain_core.messages import RemoveMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)

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
for msg in ["My name is Ravi.", "I work at Infosys.", "I live in Nellore."]:
    agent_summ.invoke({"messages": [{"role": "user", "content": msg}]}, config=cfg3)

r = agent_summ.invoke({"messages": [{"role": "user", "content": "What do you know about me?"}]}, config=cfg3)
print(r["messages"][-1].content[:200])
