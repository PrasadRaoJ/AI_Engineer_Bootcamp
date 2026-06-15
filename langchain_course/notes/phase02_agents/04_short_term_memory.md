# Short-term Memory

## Concept

By default each `.invoke()` call is stateless — the agent forgets everything after it returns. **Short-term memory** keeps the conversation alive across multiple turns using a **checkpointer** and a **thread ID**.

```
Without memory:
  Turn 1: "My name is Ravi" → agent replies → state gone
  Turn 2: "What is my name?" → agent has no idea

With memory (same thread_id):
  Turn 1: "My name is Ravi" → state saved to checkpointer
  Turn 2: "What is my name?" → state loaded → "Your name is Ravi"
```

## Flow

```
agent.invoke(input, config={"configurable": {"thread_id": "sess-001"}})
                                                      │
                                     ┌────────────────▼───────────────┐
                                     │         Checkpointer           │
                                     │   saves state after each turn  │
                                     │   loads state before each turn │
                                     └────────────────┬───────────────┘
                                                      │
                              ┌───────────────────────┼──────────────────────┐
                              │ same thread_id        │                      │ different thread_id
                              ▼                       │                      ▼
                    ┌──────────────────┐              │           ┌──────────────────┐
                    │  history loaded  │              │           │   fresh start    │
                    │  model remembers │              │           │   no memory      │
                    └──────────────────┘              │           └──────────────────┘
```

## Basic setup

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

checkpointer = InMemorySaver()   # dev/testing only — resets on restart

agent = create_agent(
    model=llm,
    tools=[...],
    system_prompt="You are a helpful assistant.",
    checkpointer=checkpointer,
)
```

## Multi-turn conversation

```python
cfg = {"configurable": {"thread_id": "session-001"}}

# turn 1
agent.invoke({"messages": [{"role": "user", "content": "My name is Ravi."}]}, config=cfg)

# turn 2 — same thread_id, model remembers
agent.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=cfg)
# → "Your name is Ravi."

# different thread_id — fresh start, no memory
agent.invoke({"messages": [{"role": "user", "content": "What is my name?"}]},
             config={"configurable": {"thread_id": "session-999"}})
# → "I don't know your name."
```

## Checkpointer types

| Checkpointer | Use case |
|-------------|---------|
| `InMemorySaver` | Dev / testing — lost on process restart |
| `PostgresSaver` | Production — persists across restarts (`pip install langgraph-checkpoint-postgres`) |
| `SqliteSaver` | Lightweight production alternative |

## Custom state with `state_schema`

Extend `AgentState` to store extra fields alongside messages:

```python
from langchain.agents import AgentState

class CustomState(AgentState):
    user_name: str = ""
    turn_count: int = 0

agent = create_agent(
    model=llm,
    tools=[...],
    checkpointer=InMemorySaver(),
    state_schema=CustomState,
)
```

## Managing message history — three approaches

### 1. Trim with `@before_model` + `RemoveMessage`

Run before each LLM call — delete old messages to stay within token limits:

```python
from langchain.agents.middleware import before_model
from langchain.agents import AgentState
from langchain_core.messages import RemoveMessage

@before_model
def trim_old_messages(state: AgentState, runtime):
    messages = state["messages"]
    if len(messages) > 10:                            # keep last 10
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-10]]}
```

### 2. Summarize with `SummarizationMiddleware`

Condenses old messages into a summary when a threshold is hit — the model still has context, just compressed:

```python
from langchain.agents.middleware import SummarizationMiddleware

summarizer = SummarizationMiddleware(
    model=llm,
    trigger=("messages", 20),   # summarize when > 20 messages
    keep=("messages", 5),       # keep last 5 after summarizing
)

agent = create_agent(model=llm, tools=[...], checkpointer=InMemorySaver(), middleware=[summarizer])
```

### 3. Delete all with `REMOVE_ALL_MESSAGES`

Clear the entire history for a thread:

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

agent.invoke({"messages": REMOVE_ALL_MESSAGES}, config=cfg)
```

## Approach comparison

| Approach | Memory retained | Token cost | Use when |
|----------|----------------|-----------|---------|
| No trimming | full history | grows forever | short conversations |
| `RemoveMessage` trim | last N messages | capped | you need only recent context |
| `SummarizationMiddleware` | compressed summary | moderate | need full context, long sessions |
| `REMOVE_ALL_MESSAGES` | none | reset to 0 | user logs out / new session |

## Gotchas

- `checkpointer=` is required for memory — without it, every `.invoke()` starts fresh.
- `thread_id` is required in config — without it the checkpointer does not save state.
- `InMemorySaver` is for development only — data is lost when the process restarts.
- Each `thread_id` is an isolated conversation — users should always get their own thread ID.
- `AgentState` is imported from `langchain.agents`, not `langchain.agents.state`.
- `REMOVE_ALL_MESSAGES` is from `langgraph.graph.message`, not `langchain_core`.
