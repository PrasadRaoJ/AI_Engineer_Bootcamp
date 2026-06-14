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

```python
from langchain_core.tools import tool

@tool
def get_order_status(order_id: str) -> str:
    """Returns the current status of an order given its order ID."""
    # real code would query a database
    return f"Order {order_id} is out for delivery."
```

- The **docstring** is what the LLM reads to decide when to call this tool — write it clearly.
- Type hints are required — LangChain uses them to build the JSON schema sent to the model.

## Binding tools to the model

```python
llm_with_tools = llm.bind_tools([get_order_status])
```

## Full tool-calling loop

```python
messages = [
    SystemMessage("You are a formal customer service representative for Slipkart. Be professional and friendly."),
    HumanMessage("Where is my order ORD123?"),
]
response = llm_with_tools.invoke(messages)

if response.tool_calls:
    for tc in response.tool_calls:
        result = get_order_status.invoke(tc["args"])
        messages.append(response)                          # AIMessage with tool_call
        messages.append(ToolMessage(result, tool_call_id=tc["id"]))
    final = llm_with_tools.invoke(messages)
    print(final.content)
```

## Gotchas

- If the model decides no tool is needed, `response.tool_calls` is an empty list — always check.
- `ToolMessage` requires `tool_call_id` to match the model's request — use `tc["id"]`.
- The docstring is the tool's "prompt" — vague docstrings cause the model to call the wrong tool or miss it entirely.
