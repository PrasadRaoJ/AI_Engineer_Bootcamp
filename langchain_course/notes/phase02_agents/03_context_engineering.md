# Context Engineering

## Concept

Context engineering is **"providing the right information and tools in the right format so the LLM can accomplish a task."** The docs call it "the number one job of AI Engineers."

The idea: don't give the model everything all the time. Shape what it sees — the system prompt, the messages, the tools, the model itself — based on who the user is and what they're doing.

## Three types of context

```
┌─────────────────────────────────────────────────────────────────┐
│  MODEL CONTEXT  (transient — enters each LLM call)              │
│   • System prompt   — instructions shaped by state / runtime    │
│   • Messages        — conversation history                      │
│   • Tools           — filtered by role, stage, permissions      │
│   • Model selection — different model based on conversation     │
│   • Response format — Pydantic schema for structured output     │
├─────────────────────────────────────────────────────────────────┤
│  TOOL CONTEXT  (persistent — what tools read/write)             │
│   • runtime.state   — conversation-scoped data                  │
│   • runtime.store   — cross-conversation long-term data         │
│   • runtime.context — per-call static data (user_id, API keys)  │
├─────────────────────────────────────────────────────────────────┤
│  LIFECYCLE CONTEXT  (between steps)                             │
│   • Summarization middleware — condenses old messages           │
│   • Guardrails middleware    — filters input/output             │
│   • Logging middleware       — observability hooks              │
└─────────────────────────────────────────────────────────────────┘
```

## Two middleware decorators

### `@dynamic_prompt` — change the system prompt per request

The decorated function receives a `ModelRequest` and returns a `str` or `SystemMessage`.

```python
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from pydantic import BaseModel

class Context(BaseModel):
    user_name: str
    language: str

@dynamic_prompt
def my_prompt(request: ModelRequest[Context]) -> str:
    name = request.runtime.context.user_name
    lang = request.runtime.context.language
    return f"You are a Slipkart support agent. Address the customer as {name}. Reply in {lang}."

agent = create_agent(
    model=llm,
    tools=[...],
    context_schema=Context,
    middleware=[my_prompt],        # pass as middleware, not system_prompt
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is your return policy?"}]},
    context=Context(user_name="Ravi", language="English"),
)
```

### `@wrap_model_call` — intercept and modify the full model request

Receives `(request, handler)` — call `handler(request)` to run the model, or modify `request` first using `request.override(...)`.

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def filter_tools_by_role(request: ModelRequest[Context], handler):
    if request.runtime.context.role != "admin":
        # Remove admin-only tools for non-admin users
        filtered = [t for t in request.tools if t.name != "cancel_order"]
        request = request.override(tools=filtered)   # ← always use .override(), not direct assignment
    return handler(request)
```

## `ModelRequest` — what you get inside middleware

```python
request.messages          # conversation history (excluding system message)
request.system_message    # current system message
request.tools             # list of available tools
request.model             # the chat model instance
request.response_format   # Pydantic schema if set
request.state             # current AgentState
request.runtime.context   # per-call context (user_id, role, etc.)
request.runtime.store     # long-term memory store

request.override(tools=[...])          # return modified copy — correct pattern
request.override(system_message=...)   # change system prompt
request.override(model=...)            # swap model mid-chain
```

## Flow with middleware

```
agent.invoke(input, context=Context(...))
                │
                ▼
    ┌───────────────────────┐
    │  @dynamic_prompt      │  builds system prompt from context
    │  @wrap_model_call     │  filters tools / modifies request
    └───────────┬───────────┘
                │ modified ModelRequest
                ▼
    ┌───────────────────────┐
    │   model node (LLM)    │  sees only what middleware allowed
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │   tools node          │  only filtered tools available
    └───────────────────────┘
```

## When to use which

| Need | Use |
|------|-----|
| Personalize system prompt per user | `@dynamic_prompt` |
| Filter tools by role / permissions | `@wrap_model_call` + `request.override(tools=...)` |
| Swap model based on context | `@wrap_model_call` + `request.override(model=...)` |
| Inject user data into tools | `context_schema=` + `context=` (covered in Runtime) |
| Summarize long message history | `SummarizationMiddleware` (covered in Short-term Memory) |

## ToolRuntime vs ModelRequest — which one to use?

| | `ToolRuntime[Context]` | `ModelRequest[Context]` |
|---|---|---|
| **Where** | Inside `@tool` functions | Inside middleware (`@dynamic_prompt`, `@wrap_model_call`) |
| **When** | After LLM calls the tool | Before LLM is called |
| **Access** | `runtime.context` | `request.runtime.context` + messages, tools, model |
| **Use for** | RBAC check, data access inside a tool | Personalize prompt, filter tools before LLM sees them |

```
agent.invoke(input, context=Context(...))
        │
        ▼
  middleware  ← ModelRequest[Context]   (before LLM)
        │
        ▼
       LLM    → decides to call a tool
        │
        ▼
     @tool    ← ToolRuntime[Context]    (at execution)
```

Same `Context` data flows through both — accessed at different stages.

- Need to block tool execution? → `ToolRuntime`
- Need to hide tools from LLM entirely? → `ModelRequest` middleware
- Best practice: **both** — middleware hides, `ToolRuntime` is safety net

## Tool metadata — declare permissions on the tool itself

Instead of hardcoding role/tool names in middleware, tag each tool with allowed roles:

```python
@tool
def cancel_order(order_id: str) -> str:
    """Cancel a Slipkart order."""
    ...

cancel_order.metadata = {"roles": ["admin"]}
get_order_status.metadata = {"roles": ["admin", "customer"]}
```

Then middleware becomes generic — no tool names hardcoded:

```python
@wrap_model_call
def filter_by_role(request: ModelRequest[Context], handler):
    role = request.runtime.context.role
    filtered = [t for t in request.tools if role in t.metadata.get("roles", [role])]
    request = request.override(tools=filtered)
    return handler(request)
```

New tool added? Just set its `.metadata` — middleware needs no changes.

## Security — Defense in Depth with Two Layers

Context engineering is not just about personalization — it is a core security mechanism for multi-role AI agents. Restricting what the LLM sees and what tools can execute must be treated as two independent enforcement points.

### Why one layer is not enough

**Middleware only (no ToolRuntime check):**
The LLM cannot see the tool, but if middleware is misconfigured, bypassed, or a future developer removes it, the tool executes without any restriction. There is no fallback.

**ToolRuntime only (no middleware filter):**
The tool blocks unauthorized execution, but the LLM still sees the tool in its schema. It will attempt to call it, receive a "Permission denied" response, and may retry or hallucinate — wasting tokens and producing poor responses.

### Two-layer model

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Middleware (LLM-level restriction)        │
│  What: filter tools before LLM schema is sent       │
│  How:  @wrap_model_call + tool.metadata             │
│  Result: LLM never knows the tool exists            │
├─────────────────────────────────────────────────────┤
│  Layer 2 — ToolRuntime (execution-level restriction) │
│  What: enforce role inside the tool function        │
│  How:  runtime.context.role check                   │
│  Result: even a direct/unexpected call is blocked   │
└─────────────────────────────────────────────────────┘
```

### Real example — ClaimSure Insurance Agent

A `staff` member should not be able to override claim rejections — only a `manager` can.

```python
# Tool declares who can access it
override_rejection.metadata = {"roles": ["manager"]}

# Layer 1 — middleware hides tool from LLM for non-managers
@wrap_model_call
def filter_by_role(request: ModelRequest[Context], handler):
    role = request.runtime.context.role
    filtered = [t for t in request.tools if role in t.metadata.get("roles", [role])]
    request = request.override(tools=filtered)
    return handler(request)

# Layer 2 — tool blocks execution even if called directly
@tool
def override_rejection(claim_id: str, reason: str, runtime: ToolRuntime[Context]) -> str:
    """Override a rejected claim. Managers only."""
    if runtime.context.role != "manager":
        return "Only managers can override rejections."
    ...
```

**What this guarantees:**
- A `staff` LLM session never sees `override_rejection` in its tool schema → cannot attempt the call
- If somehow called directly (API abuse, future bug) → blocked at execution
- Security is defined on the tool itself via `.metadata` → no scattered role checks across the codebase

### Summary

| Concern | Middleware | ToolRuntime |
|---|---|---|
| LLM attempts unauthorized tool call | Prevented | Blocked after attempt |
| Misconfigured middleware bypassed | Not covered | Still blocked |
| Poor LLM responses from denied tools | Prevented | Not covered |
| Defense in depth | — | Both together ✅ |

Always implement both layers for any tool that handles sensitive operations.

## Gotchas

- `@dynamic_prompt` and `@wrap_model_call` are in `langchain.agents.middleware` — not `langchain.agents`.
- Always use `request.override(field=...)` to modify a request — direct attribute assignment is deprecated.
- Middleware is passed in `middleware=[]` on `create_agent` — not at invoke time.
- Multiple middleware stack in order — first in `middleware=[]` runs first.
- `@dynamic_prompt` is a shorthand for `@wrap_model_call` that only changes the system message.
- **`tools=[]` causes an infinite loop.** `create_agent` always expects at least one tool. Add a real tool (or a dummy info tool) — the agent won't terminate otherwise.
