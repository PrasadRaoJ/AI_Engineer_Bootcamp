# Agents

## Concept

In Phase 1 you wrote the tool-calling loop manually:

```python
# Phase 1 — manual loop
response = llm.bind_tools(tools).invoke(messages)
if response.tool_calls:
    messages.append(response)
    for tc in response.tool_calls:
        result = TOOL_MAP[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(result, tool_call_id=tc["id"]))
response = llm.bind_tools(tools).invoke(messages)  # call again after tools
```

**`create_agent` does this loop for you** — it keeps calling the model, running tools, and feeding results back until the model stops producing tool calls.

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=[my_tool], system_prompt="You are helpful.")
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

## Flow

```
┌──────────────────────────────────────────────────────────────┐
│  agent.invoke({"messages": [HumanMessage]})                  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │   model node             │  LLM reads messages list
          │   (LLM call)             │  returns AIMessage
          └──────────────┬───────────┘
                         │
              ┌──────────▼──────────┐
              │  tool_calls?        │
              └──────────┬──────────┘
                         │
           ┌─────────────┴──────────────┐
           │ YES                        │ NO
           ▼                            ▼
┌──────────────────────┐      ┌──────────────────────────┐
│   tools node         │      │   DONE                   │
│   run each tool      │      │   return {"messages": []} │
│   add ToolMessage(s) │      └──────────────────────────┘
└──────────┬───────────┘
           │
           └──────► back to model node ──────► loops until no tool_calls
```

The loop runs automatically. You never write `if response.tool_calls` again.

## Basic usage

```python
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

def get_order_status(order_id: str) -> str:
    """Get the current status of an order."""
    return f"Order {order_id} is out for delivery."

agent = create_agent(
    model=llm,
    tools=[get_order_status],
    system_prompt="You are a Slipkart support agent.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "Where is ORD123?"}]})
print(result["messages"][-1].content)  # final AIMessage
```

## Output shape

```python
result = agent.invoke({"messages": [...]})

result["messages"]           # ALL messages — Human, AI (tool call), Tool, AI (final)
result["messages"][-1].content  # the final reply text

# When response_format= is set (structured output):
result["structured_response"]   # validated Pydantic object
```

All intermediate messages (tool calls, tool results) are included — useful for logging.

## Streaming

```python
# stream_mode="updates" — yields each node's output as it runs
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Where is ORD123?"}]},
    stream_mode="updates",
):
    # chunk is {"model": {...}} or {"tools": {...}}
    if "model" in chunk:
        msg = chunk["model"]["messages"][-1]
        if msg.content:               # skip empty AIMessage (tool-call step)
            print(msg.content)

# stream_mode="values" — yields full state at each step
for chunk in agent.stream(inputs, stream_mode="values"):
    latest = chunk["messages"][-1]
```

## `create_agent` parameters

| Param | What it does |
|-------|-------------|
| `model` | `ChatOllama` instance or model string like `"openai:gpt-4o"` |
| `tools` | list of `@tool` functions or plain callables with docstrings |
| `system_prompt` | sets the agent's persona / instructions |
| `context_schema` | Pydantic schema for per-call context (user_id, API keys, etc.) |
| `middleware` | list of middleware instances — HITL, summarization, guardrails, etc. |
| `checkpointer` | adds conversation memory across turns (covered in Short-term Memory) |
| `store` | cross-thread long-term memory (covered in Long-term Memory) |
| `response_format` | Pydantic schema → final answer in `result["structured_response"]` |
| `name` | identifier when used as a subagent inside another agent |
| `debug` | print node execution details — useful when the agent seems stuck |

## Per-call context injection

Use `context_schema=` to define what data each call can carry (user ID, permissions, API keys).
Tools read this via `runtime.context`.

```python
from pydantic import BaseModel
from langgraph.prebuilt import ToolRuntime

class Context(BaseModel):
    user_id: str
    channel: str

def get_order_status(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Get order status for the current user."""
    return f"Order {order_id} status for user {runtime.context.user_id}: Out for delivery."

agent = create_agent(
    model=llm,
    tools=[get_order_status],
    system_prompt="You are a Slipkart support agent.",
    context_schema=Context,
)

# context= is separate from config=
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Where is ORD123?"}]},
    context=Context(user_id="U001", channel="web"),
)
```

`ToolRuntime` is auto-injected — the LLM never sees it as a tool argument.

## Phase 1 vs Phase 2 comparison

| | Phase 1 (manual) | Phase 2 (create_agent) |
|--|-----------------|----------------------|
| Tool loop | you write it | automatic |
| Multi-step tool calls | possible but verbose | handled |
| Per-call user data | passed manually | `context_schema=` + `context=` |
| Memory across turns | you manage `history` list | `checkpointer=` |
| HITL | manual check | `HumanInTheLoopMiddleware` |
| Guardrails, PII | manual | `middleware=[]` |

## Gotchas

- Input must be `{"messages": [...]}` — not a plain string or message list.
- `result["messages"][-1].content` is the final reply — not `result.content`.
- The first AIMessage often has empty `.content` (it's the tool-call step) — always use `[-1]`.
- `context=` is passed separately from `config=` — they are two different params.
- `ToolRuntime` import is from `langgraph.prebuilt`, not `langchain`.
- `ToolRuntime` must be type-annotated exactly — the framework injects it by type hint, not name.
- Small local models (llama3.2) sometimes loop or fail to emit clean tool calls — add `debug=True` to diagnose.
