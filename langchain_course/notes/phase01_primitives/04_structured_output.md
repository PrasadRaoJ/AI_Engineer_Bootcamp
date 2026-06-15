# Structured Output

## Concept

By default the LLM returns free text. **Structured output** forces the model to return a specific shape so you can use the result directly in code without parsing.

## Flow

```
┌──────────────────────────┐
│  1. Define schema        │  Pydantic / TypedDict / JSON Schema
└────────────┬─────────────┘
             │
             │  llm.with_structured_output(Schema)
             ▼
┌──────────────────────────┐
│  2. Bind schema to LLM   │  Provider picks its native method:
│                          │  • Ollama → format= param (JSON mode)
│                          │  • OpenAI/Anthropic → tool injection or native structured output
└────────────┬─────────────┘
             │
             │  .invoke([SystemMessage, HumanMessage])
             ▼
┌──────────────────────────┐
│  3. LLM outputs JSON     │  {"order_id": "ORD123", "issue": "...", "priority": "high"}
└────────────┬─────────────┘
             │
             │  LangChain parses + validates automatically
             ▼
┌──────────────────────────┐
│  4. You get typed object │  result.order_id / result.issue / result.priority
│     (NOT AIMessage)      │
└──────────────────────────┘
```

## Schema types

### 1. Pydantic (recommended — validates + typed)

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class SupportTicket(BaseModel):
    order_id: str                                       # plain string
    days_waiting: int                                   # integer
    is_urgent: bool                                     # boolean
    priority: Literal["low", "medium", "high"]          # constrained to fixed values
    tags: List[str]                                     # list of strings
    notes: Optional[str] = None                         # nullable field
    issue: str = Field(description="Short description of the problem")  # guides the model
```

### 2. TypedDict (lighter — no validation, returns dict)

```python
from typing import TypedDict

class SupportTicket(TypedDict):
    order_id: str
    priority: str
```

Returns a plain dict — access with `result["order_id"]`, not `result.order_id`.

### 3. JSON Schema (raw dict — no class needed)

```python
schema = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "description": "Order ID from the message"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "refund_eligible": {"type": "boolean"},
    },
    "required": ["order_id", "priority", "refund_eligible"],
}
```

Returns a plain dict. Use when you don't want to define a class.

### 4. Dataclass — ⚠️ not supported by Ollama

```python
from dataclasses import dataclass

@dataclass
class SupportTicket:
    order_id: str
    priority: str
```

Ollama raises a `ValidationError` when a dataclass is passed to `with_structured_output`. Use Pydantic instead.

### 5. Union types — ⚠️ not supported by Ollama

```python
from typing import Union

# Works with OpenAI / Anthropic (tool-calling providers), NOT Ollama
result = llm.with_structured_output(Union[StatusQuery, CancelRequest]).invoke(...)
```

Union types let the model choose between multiple schemas. Ollama raises `ValidationError` — requires a provider that supports native tool calling or structured output.

## Binding and calling

```python
structured_llm = llm.with_structured_output(SupportTicket)

result = structured_llm.invoke([
    SystemMessage("You are a Slipkart support classifier."),
    HumanMessage("My order ORD123 hasn't arrived in 10 days, I'm very frustrated!"),
])

print(result.order_id)     # "ORD123"
print(result.days_waiting) # 10
print(result.is_urgent)    # True
print(result.priority)     # "high"
```

## Structured output in agents (Phase 2)

When using `create_agent`, use `response_format=` instead of `with_structured_output`:

```python
agent = create_agent(model=llm, tools=[...], response_format=SupportTicket)
result = agent.invoke({"messages": [...]})
result["structured_response"]  # validated Pydantic object
```

## When to use

- Extracting structured data from user messages (order ID, intent, priority)
- Any time the next step in your code needs a typed field, not free text
- Replacing fragile string parsing / regex

## Schema type comparison

| Schema | Returns | Validation | Ollama support |
|--------|---------|------------|----------------|
| Pydantic | object | ✅ yes — Python-side, rich errors | ✅ yes |
| JSON Schema | dict | ✅ yes — checked against schema | ✅ yes |
| TypedDict | dict | ❌ no — type hints only | ✅ yes |
| Dataclass | object | ❌ no | ❌ ValidationError |
| Union | object | ✅ yes | ❌ ValidationError |

## Pydantic validation example

```python
class SupportTicket(BaseModel):
    order_id: str
    days_waiting: int
    priority: Literal["low", "medium", "high"]

# ✅ valid
SupportTicket(order_id="ORD123", days_waiting=10, priority="high")

# ❌ ValidationError — "ten" is not an int
SupportTicket(order_id="ORD123", days_waiting="ten", priority="high")

# ❌ ValidationError — "critical" not in Literal
SupportTicket(order_id="ORD123", days_waiting=10, priority="critical")
```

Validation fires at object creation — bad data never makes it through.

## Gotchas

- `.with_structured_output()` returns the object directly — there is no `.content`.
- Use `Field(description=...)` to guide the model on what each field means — this is part of the schema sent to the LLM.
- Use `Literal[...]` instead of plain `str` when the value must be one of a fixed set.
- `temperature=0` is recommended — deterministic structured results.
- Dataclass and Union types raise `ValidationError` on Ollama — use Pydantic or JSON Schema instead.
- TypedDict returns a dict, not an object — use `result["field"]` not `result.field`.
