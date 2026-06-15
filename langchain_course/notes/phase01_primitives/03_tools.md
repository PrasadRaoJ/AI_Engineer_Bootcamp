# Tools

## Concept

A **Tool** is a Python function the LLM can choose to call. You define what it does; the model decides when to call it based on the user's message.

## Flow

```
┌─────────────────────┐
│     User message    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LLM + bound tools  │
└──────────┬──────────┘
           │
     ┌─────┴──────────────────────────────┐
     │ no tool needed                     │ tool needed
     ▼                                    ▼
┌──────────────┐              ┌───────────────────────┐
│  text reply  │ ✓            │  tool_call {name,args}│
└──────────────┘              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  Python function runs │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  ToolMessage {result} │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │    LLM reads result   │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │    final text reply   │ ✓
                              └───────────────────────┘
```

## Defining a tool

### Basic — docstring as description

```python
from langchain_core.tools import tool

@tool
def get_order_status(order_id: str) -> str:
    """Returns the current delivery status of a Slipkart order given its order ID."""
    return f"Order {order_id} is out for delivery."
```

- The **docstring** is what the LLM reads to decide when and how to call the tool — write it clearly.
- **Type hints are required** — LangChain uses them to build the JSON schema sent to the model.
- Use **snake_case** names — some providers reject names with spaces or special characters.

### With `args_schema` — Pydantic for richer input control

Use `args_schema=` when you need field descriptions, constrained values (`Literal`), or optional fields that a plain function signature cannot express:

```python
from pydantic import BaseModel, Field
from typing import Literal

class CancelInput(BaseModel):
    order_id: str = Field(description="The Slipkart order ID, e.g. ORD123")
    reason: Literal["changed_mind", "wrong_item", "delay"] = Field(
        description="Reason for cancellation"
    )

@tool(args_schema=CancelInput)
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a Slipkart order."""
    return f"Order {order_id} cancelled. Reason: {reason}. Refund in 3-5 days."
```

The model now sees `reason` as a constrained enum — it cannot pass an arbitrary string.

### Custom name and description

```python
@tool("check_status", description="Check the live delivery status of any Slipkart order.")
def get_order_status(order_id: str) -> str:
    """Fallback docstring — description= overrides this."""
    return f"Order {order_id} is out for delivery."
```

### `return_direct` — skip the final LLM call

```python
@tool(return_direct=True)
def get_order_status(order_id: str) -> str:
    """Returns the delivery status of a Slipkart order."""
    return f"Order {order_id} is out for delivery."
```

When `return_direct=True`, the agent returns the tool's output directly to the user — the model does **not** get a chance to rephrase or summarize it. Use only when the raw tool output is already the final answer.

## Binding tools to the model

```python
llm_with_tools = llm.bind_tools([get_order_status, cancel_order])
```

## Full tool-calling loop

```python
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

messages = [
    SystemMessage("You are a formal Slipkart support agent. Be professional and friendly."),
    HumanMessage("Where is my order ORD123?"),
]

TOOL_MAP = {t.name: t for t in [get_order_status, cancel_order]}  # name → function lookup

response = llm_with_tools.invoke(messages)

if response.tool_calls:
    messages.append(response)                              # AIMessage added ONCE before the loop
    for tc in response.tool_calls:
        result = TOOL_MAP[tc["name"]].invoke(tc["args"])   # use TOOL_MAP, not locals()
        messages.append(ToolMessage(result, tool_call_id=tc["id"]))
    final = llm_with_tools.invoke(messages)
    print(final.content)
else:
    print(response.content)                                # model answered directly
```

## Tool return types

| Return | What happens |
|--------|-------------|
| `str` | Converted to ToolMessage content — model sees it |
| `dict` | Serialized to JSON — model reasons over specific fields |
| `Command` | Updates graph state directly — used inside `create_agent` (Phase 2) |

## Gotchas

- If the model decides no tool is needed, `response.tool_calls` is an empty list — always check before looping.
- `ToolMessage` requires `tool_call_id` to match `tc["id"]` exactly — wrong id means the model ignores the result.
- The docstring is the tool's "prompt" to the model — vague or missing docstrings cause wrong tool selection or no tool call at all.
- Never use `config` or `runtime` as argument names in a tool — they are reserved by LangChain for internal injection.
- `return_direct=True` skips the final LLM summarization — the raw tool string goes straight to the user.
- `args_schema=` overrides individual type hints — use it when you need `Field(description=...)` or `Literal` on inputs.
