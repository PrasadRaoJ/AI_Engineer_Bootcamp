# Event Streaming

## Concept

**`.astream_events()`** gives you fine-grained events during a run — when each component starts, streams, and ends. Unlike `.stream()` which only gives you output tokens, event streaming tells you **what is happening and when** inside the agent.

Use it for: observability, debugging, building UIs that show "thinking..." / "calling tool..." indicators.

## Flow

```
.stream()          — you only see the final output tokens
──────────────────────────────────────────────────────
User ──► LLM ──► "Paris" "is" "the" "capital"...


.astream_events()  — you see every internal event
──────────────────────────────────────────────────────
User ──► LLM
          │
          ├── event: on_chat_model_start   (LLM started)
          ├── event: on_chat_model_stream  (token: "Paris")
          ├── event: on_chat_model_stream  (token: " is")
          ├── event: on_tool_start         (tool: get_order_status called)
          ├── event: on_tool_end           (tool: returned result)
          ├── event: on_chat_model_stream  (token: "Your order...")
          └── event: on_chat_model_end     (LLM done)
```

## Key event types

| Event | Fires when |
|-------|-----------|
| `on_chat_model_start` | LLM begins generating |
| `on_chat_model_stream` | each token is generated |
| `on_chat_model_end` | LLM finishes |
| `on_tool_start` | a tool is called |
| `on_tool_end` | a tool returns a result |
| `on_chain_start` | a chain/pipeline begins |
| `on_chain_end` | a chain/pipeline finishes |

## Event structure

Each event is a dict:

```python
{
    "event": "on_chat_model_stream",   # event type
    "name":  "ChatOllama",             # which component
    "data":  {"chunk": AIMessageChunk} # the payload
}
```

## Usage

```python
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2", temperature=0)

async def main():
    async for event in llm.astream_events(
        [SystemMessage("You are a helpful assistant."),
         HumanMessage("Capital of India?")],
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_start":
            print("LLM started...")
        elif kind == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif kind == "on_chat_model_end":
            print("\nLLM done.")

asyncio.run(main())
```

## Gotchas

- Always async — there is no sync version of `.astream_events()`.
- Always pass `version="v2"` — v1 is deprecated.
- Filter by `event["event"]` to handle only the events you care about — there can be many noisy events from chains and wrappers.
- `on_tool_start` / `on_tool_end` only fire when tools execute inside a full agent chain (Phase 3+). With plain `bind_tools`, the tool call decision is visible in `on_chat_model_end` via `output.tool_calls`.
- More useful when you have tools + chains — on a plain LLM call, `.stream()` is simpler.
