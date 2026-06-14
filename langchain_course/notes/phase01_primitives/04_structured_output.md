# Structured Output

## Concept

By default the LLM returns free text. **Structured output** forces the model to return a specific shape so you can use the result directly in code without parsing.

## Flow

```
┌──────────────────────────┐
│  1. Define schema        │  Pydantic / TypedDict / Dataclass / JSON Schema
└────────────┬─────────────┘
             │
             │  llm.with_structured_output(Schema)
             ▼
┌──────────────────────────┐
│  2. Bind schema to LLM   │  LangChain injects schema as a tool internally
└────────────┬─────────────┘
             │
             │  .invoke([SystemMessage, HumanMessage])
             ▼
┌──────────────────────────┐
│  3. LLM outputs JSON     │  {"order_id": "ORD123", "issue": "...", "priority": "high"}
└────────────┬─────────────┘
             │
             │  LangChain parses + validates JSON automatically
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
    order_id: str                          # plain string
    days_waiting: int                      # integer
    is_urgent: bool                        # boolean
    priority: Literal["low", "medium", "high"]   # constrained string
    tags: List[str]                        # list of strings
    notes: Optional[str] = None            # nullable field
```

### 2. TypedDict (lighter — no validation, returns dict)

```python
from typing import TypedDict

class SupportTicket(TypedDict):
    order_id: str
    priority: str
```

### 3. Dataclass — ⚠️ not supported by Ollama

Dataclasses are listed in the LangChain docs but Ollama's backend rejects them. Use Pydantic instead.

### 4. JSON Schema (raw dict — no class needed, most flexible)

```python
schema = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "refund_eligible": {"type": "boolean"},
    },
    "required": ["order_id", "priority", "refund_eligible"],
}
```

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

## When to use

- Extracting structured data from user messages (order ID, intent, priority)
- Any time the next step in your code needs a typed field, not free text
- Replacing fragile string parsing / regex

## Schema type comparison

| Schema | Returns | Validation | How it validates |
|--------|---------|------------|-----------------|
| Pydantic | object | ✅ yes | Python-side at object creation — rich errors, type coercion |
| JSON Schema | dict | ✅ yes | LangChain checks response against schema using `jsonschema` lib |
| TypedDict | dict | ❌ no | just type hints — nothing enforced at runtime |
| Dataclass | object | ❌ no | ⚠️ not supported by Ollama |

## Pydantic validation example (Python-side)

```python
from pydantic import BaseModel
from typing import Literal

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

# ❌ ValidationError — order_id is missing
SupportTicket(days_waiting=10, priority="high")
```

Validation fires at object creation — bad data never makes it through.

## Gotchas

- `.with_structured_output()` returns the object directly — there is no `.content`.
- Use `Field(description=...)` to guide the model on what each field means.
- Use `Literal[...]` instead of plain `str` when the value must be one of a fixed set.
- `temperature=0` is recommended — deterministic structured results.
