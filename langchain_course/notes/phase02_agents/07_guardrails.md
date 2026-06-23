# Guardrails

## Concept

Guardrails are safety mechanisms that **validate and filter content at key points** in agent execution. They detect sensitive data, enforce policies, and prevent unsafe behaviors — before the model runs, after it responds, or both.

```
User input
    │
    ▼
@before_agent  ─── block early if input violates policy
    │
    ▼
  model
    │
    ▼
 tools
    │
    ▼
@after_agent   ─── validate final response before user sees it
    │
    ▼
User output
```

## Two strategies

| Strategy | Mechanism | Tradeoff |
|----------|-----------|---------|
| **Deterministic** | Regex, keyword matching, explicit rules | Fast, cheap, but misses nuanced violations |
| **Model-based** | LLM or classifier evaluates content | Catches subtle issues, but adds latency/cost |

Both can be mixed in the same agent.

## Built-in: `PIIMiddleware`

Detects and handles Personally Identifiable Information.

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=llm,
    tools=[...],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block", apply_to_input=True),
    ],
)
```

### Supported PII types

| `pii_type` | What it detects |
|-----------|-----------------|
| `"email"` | Email addresses |
| `"credit_card"` | Credit card numbers (Luhn validated) |
| `"ip"` | IP addresses |
| `"mac_address"` | MAC addresses |
| `"url"` | URLs |

Use `detector=r"your-regex"` for custom patterns (e.g. API keys, SSNs).

### Handling strategies

| `strategy` | Output | Use when |
|-----------|--------|---------|
| `"redact"` | `[REDACTED_EMAIL]` | Default masking |
| `"mask"` | `****-****-****-1234` | Show partial value |
| `"hash"` | `a8f5f167...` | Deterministic replacement |
| `"block"` | Raises exception | Strict enforcement — stop everything |

### Where it applies

| Parameter | Default | Effect |
|-----------|---------|--------|
| `apply_to_input` | `True` | Scan user messages before the model runs |
| `apply_to_output` | `False` | Scan AI responses before returning to user |
| `apply_to_tool_results` | `False` | Scan tool return values |

`apply_to_output=True` also redacts streamed output (requires langchain>=1.3.2).

## Custom: `@before_agent`

Runs before every agent execution. Use for input validation, policy enforcement, or early rejection.

```python
from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime):
    if not state["messages"]:
        return None
    first = state["messages"][0]
    if first.type == "human" and "exploit" in first.content.lower():
        return {
            "messages": [{"role": "assistant", "content": "I cannot help with that."}],
            "jump_to": "end",   # short-circuit — skip model entirely
        }
    return None   # None = continue normally

agent = create_agent(model=llm, tools=[...], middleware=[content_filter])
```

### Key points

- `can_jump_to=["end"]` in the decorator enables early termination.
- Return `None` to continue normally.
- Return `{"messages": [...], "jump_to": "end"}` to block and reply immediately without calling the model.
- `state["messages"][0]` is the first message; check `first.type == "human"` before inspecting content.

## Class-based middleware

Use `AgentMiddleware` when the guardrail needs state (e.g. a list of banned keywords, a counter):

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

class ContentFilterMiddleware(AgentMiddleware):
    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        if not state["messages"]:
            return None
        first = state["messages"][0]
        if first.type != "human":
            return None
        content = first.content.lower()
        if any(kw in content for kw in self.banned_keywords):
            return {
                "messages": [{"role": "assistant", "content": "Request blocked by content policy."}],
                "jump_to": "end",
            }
        return None
```

## Custom: `@after_agent`

Runs after every agent execution. Use for output validation, compliance checks, or safety filtering.

```python
from langchain.agents.middleware import after_agent, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

@after_agent(can_jump_to=["end"])
def output_safety(state: AgentState, runtime: Runtime):
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None
    if "password" in last.content.lower():
        last.content = "[Response blocked — contained sensitive data.]"
    return None

agent = create_agent(model=llm, tools=[...], middleware=[output_safety])
```

## Layered protection

Stack multiple guardrails in the order they should run:

```python
agent = create_agent(
    model=llm,
    tools=[search_tool, send_email_tool],
    middleware=[
        # 1. input check — block before the model even sees the message
        ContentFilterMiddleware(banned_keywords=["exploit", "hack"]),

        # 2. PII on input — redact before model runs
        PIIMiddleware("email", strategy="redact", apply_to_input=True),

        # 3. PII on output — redact before user sees response
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # 4. HITL — pause before risky tool calls
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # 5. output safety — final check on model response
        output_safety,
    ],
)
```

Middleware is applied in list order — deterministic checks typically run first (fast), model-based checks last.

## Gotchas

- `@before_agent` / `@after_agent` decorators require `can_jump_to=["end"]` to enable early exit — without it, `"jump_to": "end"` in the return value is ignored.
- Class-based middleware uses `@hook_config(can_jump_to=["end"])` on the method (same concept, different syntax).
- `AgentState` for guardrail hooks is imported from `langchain.agents.middleware` (not `langchain.agents`).
- Returning `None` always means "continue" — never return an empty dict.
- `strategy="block"` raises an exception — wrap invocations in try/except if you need graceful handling.
- `apply_to_output=True` on `PIIMiddleware` adds latency for every response — only enable where needed.
- Multiple `PIIMiddleware` instances can stack (one per PII type is fine).

## When to use which hook — one line each

| Hook | Purpose |
|------|---------|
| `@before_agent` | "banking question కాదు? block చేయి, LLM కి వెళ్ళకు." |
| `@before_model` | "LLM కి వెళ్ళే ముందు card number mask చేయి." |
| `@after_model` | "LLM `delete_account` tool call చేయబోతోందా? cancel చేయి." |
| `@after_agent` | "final response లో 'backend' అనే word ఉందా? replace చేయి." |

**Agent hooks** (`before/after_agent`) = entire execution చుట్టూ — gate/filter, fires once.

**Model hooks** (`before/after_model`) = ప్రతి LLM call చుట్టూ — inspect/modify, fires on every model call including after tool results.
