# Runtime — Context, Store & Config

## Concept

**Runtime** is the set of things automatically injected into tools and middleware during agent execution — without you passing them as explicit arguments or the LLM seeing them.

Five components in the Runtime object:

| Component | What it is |
|-----------|-----------|
| `context` | Per-call static data (user_id, API keys, permissions) |
| `store` | Cross-thread long-term memory (BaseStore) |
| `stream_writer` | Write custom events to the stream |
| `execution_info` | thread_id, run_id, attempt number |
| `server_info` | LangGraph Server only — assistant ID, auth user |

## Flow

```
agent.invoke(input, config=..., context=Context(...))
                                        │
                    ┌───────────────────┘
                    │  injected automatically into tools + middleware
                    ▼
┌───────────────────────────────────────────────────┐
│  ToolRuntime                                      │
│    .context       → user_id, API keys, etc.       │
│    .store         → long-term memory store        │
│    .state         → current graph state           │
│    .stream_writer → emit custom stream events     │
│    .execution_info→ thread_id, run_id             │
└───────────────────────────────────────────────────┘
```

## Accessing runtime in tools

Declare a `runtime` parameter typed as `ToolRuntime[Context]` — it is injected automatically. The LLM never sees it as a tool argument.

```python
from pydantic import BaseModel
from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import os

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

class Context(BaseModel):
    user_id: str
    role: str   # "admin" | "customer"

def cancel_order(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Cancel a Slipkart order."""
    if runtime.context.role != "admin":
        return "Permission denied. Only admins can cancel orders."
    return f"Order {order_id} cancelled by {runtime.context.user_id}."

agent = create_agent(
    model=llm,
    tools=[cancel_order],
    system_prompt="You are a Slipkart support agent.",
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Cancel order ORD123."}]},
    context=Context(user_id="U001", role="customer"),  # "customer" → denied
)
print(result["messages"][-1].content)
```

## RunnableConfig — tags, metadata, callbacks

Every `.invoke()` / `.stream()` also accepts `config=` — a standard dict for tracing and callbacks. This is separate from `context=`.

```python
result = agent.invoke(
    {"messages": [...]},
    config={
        "tags":          ["slipkart", "support"],   # label run in LangSmith
        "metadata":      {"user_id": "U001"},       # key/value on the trace
        "run_name":      "support-session",         # human-readable run name
        "callbacks":     [MyCallback()],            # hook into execution events
        "recursion_limit": 10,                      # cap tool loop iterations
        "configurable":  {"thread_id": "sess-1"},   # required for checkpointer (memory)
    },
    context=Context(user_id="U001", role="customer"),  # separate from config
)
```

## Callbacks

Callbacks fire at specific points — useful for logging, latency tracking, or custom monitoring.

```python
from langchain_core.callbacks import BaseCallbackHandler

class SupportLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(f"[LLM] {serialized['name']} called")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[TOOL] {serialized['name']} | args: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"[TOOL] result: {output.content}")
```

| Hook | Fires when |
|------|-----------|
| `on_chat_model_start` | LLM call begins |
| `on_llm_new_token` | each token (streaming) |
| `on_llm_end` | LLM call completes |
| `on_tool_start` | tool is about to run |
| `on_tool_end` | tool returned a result |
| `on_tool_error` | tool raised an exception |

## context= vs config= — when to use which

| Need | Use |
|------|-----|
| Pass user_id, permissions, API keys to tools | `context=Context(...)` |
| Label runs in LangSmith | `config["tags"]`, `config["metadata"]` |
| Log tool/LLM events in real time | `config["callbacks"]` |
| Enable conversation memory | `config["configurable"]["thread_id"]` |
| Cap runaway tool loops | `config["recursion_limit"]` |

## Gotchas

- `context=` and `config=` are two separate keyword args — don't put context inside config.
- `ToolRuntime` import is `from langgraph.prebuilt import ToolRuntime` — not from langchain.
- The `runtime` param must be typed exactly as `ToolRuntime` or `ToolRuntime[Context]` — injection is by type hint.
- `on_tool_end`'s `output` is a `ToolMessage` — use `.content` to get the text.
- `recursion_limit` default is 25 — lower it when testing to fail fast on loops.
