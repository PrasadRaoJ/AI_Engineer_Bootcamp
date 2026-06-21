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

Extend `AgentState` to store extra fields alongside messages. This is **for your backend code**, not for the user — the user only sees LLM replies, your code reads the state fields.

```python
from langchain.agents import AgentState

class CustomState(AgentState):
    turn_count: int = 0
    user_plan: str = "free"    # "free" | "pro"
    escalated: bool = False

agent = create_agent(
    model=llm,
    tools=[...],
    checkpointer=InMemorySaver(),
    state_schema=CustomState,
)

r = agent.invoke(
    {"messages": [...], "turn_count": 1, "user_plan": "free"},
    config=cfg,
)
# your backend reads this, not the user
if r["turn_count"] > 10:
    print("Free limit reached.")
```

**When to use custom state:**

| Field | Use |
|---|---|
| `turn_count` | Rate limiting — block after N turns |
| `tokens_used` | Billing — track cost per session |
| `user_plan` | Feature gating — free vs pro |
| `booking_stage` | Workflow — track step in multi-turn flow |
| `escalated` | Flags — notify human agent |

**Trim does NOT delete custom state fields:**

`RemoveMessage` only touches the `messages` list. Custom fields survive trimming:

```
After trim:
  messages    → [last 4 only]   ← trimmed
  turn_count  → 42              ← intact
  user_plan   → "pro"           ← intact
```

So you can safely combine `@before_model` trim + `CustomState` — messages get pruned, your backend data stays.

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

**What counts as a message?**
Every entry in history — both Human and AI — is one message. One conversation turn = 2 messages.

```
messages = [
    HumanMessage("My name is Ravi."),      # index 0  ← removed (old)
    AIMessage("Nice to meet you, Ravi."),  # index 1  ← removed (old)
    HumanMessage("I work at Infosys."),    # index 2  ← kept (last 4)
    AIMessage("Got it."),                  # index 3  ← kept
    HumanMessage("What do you know?"),     # index 4  ← kept
    AIMessage("You are Ravi..."),          # index 5  ← kept
]
messages[:-4]  →  removes index 0, 1      (old messages)
messages[-4:]  →  keeps  index 2, 3, 4, 5 (last 4 = last 2 turns)
```

So `keep last 4 messages` = `keep last 2 turns` of conversation.

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

**Important:**
- `SummarizationMiddleware` makes a real LLM call using the `model=` you pass — it costs tokens
- No system prompt needed — it has its own built-in summarization prompt (`DEFAULT_SUMMARY_PROMPT`)
- You can override it if needed: `summary_prompt="Summarize briefly in bullet points."`

### 3. Delete all messages

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.messages import RemoveMessage

agent.update_state(cfg, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})
```

`REMOVE_ALL_MESSAGES` is a special sentinel string `"__remove_all__"`. It must be wrapped in `RemoveMessage(id=REMOVE_ALL_MESSAGES)` and passed via `update_state` — **not** via `invoke`.

> Passing `{"messages": REMOVE_ALL_MESSAGES}` directly to `invoke` does NOT work — it treats the string as a new message.

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
