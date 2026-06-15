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

## Gotchas

- `@dynamic_prompt` and `@wrap_model_call` are in `langchain.agents.middleware` — not `langchain.agents`.
- Always use `request.override(field=...)` to modify a request — direct attribute assignment is deprecated.
- Middleware is passed in `middleware=[]` on `create_agent` — not at invoke time.
- Multiple middleware stack in order — first in `middleware=[]` runs first.
- `@dynamic_prompt` is a shorthand for `@wrap_model_call` that only changes the system message.
