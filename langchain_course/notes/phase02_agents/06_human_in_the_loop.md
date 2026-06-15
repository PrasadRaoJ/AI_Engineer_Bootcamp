# Human-in-the-loop (HITL)

## Concept

By default agents execute tool calls automatically. **Human-in-the-loop** pauses the agent before (or during) risky tool calls so a human can approve, edit, or reject them before anything is executed.

```
Without HITL:
  agent decides to call delete_records() → executes immediately → data gone

With HITL:
  agent decides to call delete_records()
       ↓
  middleware interrupts
       ↓
  human sees: "Agent wants to delete records. Approve / Edit / Reject?"
       ↓
  human approves → delete_records() runs
  (or) human rejects → agent hears feedback, stops
  (or) human edits args → modified call runs
```

HITL requires a **checkpointer** (to persist state while paused) and a **thread_id** (to resume the right conversation).

## Setup

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model=llm,
    tools=[delete_records, read_table],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_records": True,    # always pause
                "read_table": False,       # never pause (safe)
            },
        )
    ],
    checkpointer=InMemorySaver(),          # required — persists state while paused
)
```

## `interrupt_on` values

| Value | Behavior |
|-------|---------|
| `True` | Always interrupt on this tool |
| `False` | Never interrupt (auto-approve) |
| `dict` / `InterruptOnConfig` | Fine-grained config — see below |

## `InterruptOnConfig` options

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "write_file": {
            "allowed_decisions": ["approve", "reject"],  # which decisions to offer
            "description": "File write requires approval",  # shown to human
            "when": my_predicate,   # callable — interrupt only when True
        },
    },
)
```

| Field | Type | Description |
|-------|------|-------------|
| `allowed_decisions` | `list[str]` | Which decision types to show (`"approve"`, `"edit"`, `"reject"`, `"respond"`) |
| `description` | `str` or callable | Label shown to the human reviewer |
| `when` | callable → `bool` | Predicate receiving `ToolCallRequest`; interrupt only when it returns `True` |

## Four decision types

| Type | What happens |
|------|-------------|
| `"approve"` | Execute the tool call exactly as-is |
| `"edit"` | Human modifies the args; modified call runs |
| `"reject"` | Tool is not called; rejection message guides agent |
| `"respond"` | Human directly provides the tool's return value (human acts as the tool) |

## Invoke + interrupt pattern

```python
cfg = {"configurable": {"thread_id": "hitl-session-001"}}

# Step 1 — agent runs until interrupt
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete cancelled orders."}]},
    config=cfg,
    version="v2",
)

# When interrupted, result has .interrupts (list of paused calls)
print(result.interrupts)
```

## Resume with `Command(resume=...)`

### Approve

```python
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=cfg,
    version="v2",
)
```

### Edit (change args before executing)

```python
agent.invoke(
    Command(resume={
        "decisions": [{
            "type": "edit",
            "edited_action": {
                "name": "delete_records",
                "args": {"table": "orders", "condition": "status='cancelled' AND created_at < '2025-01-01'"},
            }
        }]
    }),
    config=cfg,
    version="v2",
)
```

### Reject (tool not called, message guides agent)

```python
agent.invoke(
    Command(resume={
        "decisions": [{"type": "reject", "message": "User rejected. Do not retry this deletion."}]
    }),
    config=cfg,
    version="v2",
)
```

### Respond (human substitutes for the tool)

```python
agent.invoke(
    Command(resume={
        "decisions": [{"type": "respond", "message": "Records count: 42"}]
    }),
    config=cfg,
    version="v2",
)
```

### Multiple decisions (when agent called several tools at once)

Provide decisions in the same order as the interrupted calls:

```python
Command(resume={
    "decisions": [
        {"type": "approve"},
        {"type": "reject", "message": "Too dangerous."},
    ]
})
```

## Conditional interrupts with `when`

Interrupt only when the tool call meets a condition:

```python
from langchain.agents.middleware import ToolCallRequest

def writes_outside_workspace(request: ToolCallRequest) -> bool:
    path = request.tool_call["args"].get("path", "")
    return not path.startswith("/workspace/")

HumanInTheLoopMiddleware(
    interrupt_on={
        "write_file": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "when": writes_outside_workspace,   # only interrupt for paths outside /workspace/
        },
    },
)
```

When `when` returns `False`, the call proceeds without pausing.

## Streaming with HITL

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Delete records."}]},
    config=cfg,
    stream_mode=["updates", "messages"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, _ = chunk["data"]
        if token.content:
            print(token.content, end="", flush=True)
    elif "__interrupt__" in chunk.get("data", {}):
        print("\nInterrupted — waiting for decision.")
```

## Gotchas

- `checkpointer=` is **required** — HITL state cannot persist without it.
- `thread_id` must be in config — without it the checkpointer cannot store/resume state.
- `version="v2"` is required on `agent.invoke()` for HITL to work.
- After a successful resume, access output via `result.value["messages"]` — `result["messages"]` is deprecated since LangGraph v1.1.
- `result.interrupts` is truthy when interrupted, empty tuple when not — always check `if result.interrupts:`, not `hasattr`.
- `reject` message guides agent behavior (e.g. "don't retry") — it does not call the tool at all.
- `respond` treats the human's message as the tool's return value — the agent sees it as tool success, not rejection.
- Multiple tool calls in one turn → provide decisions in the same order as interrupted calls.
- `InMemorySaver` is dev-only — use `AsyncPostgresSaver` in production.
- `edit` with significantly different args may cause the model to re-evaluate and call tools again.
