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
    "name":  "ChatOllama",             # which component fired it
    "data":  {"chunk": AIMessageChunk} # the payload
}
```

## Basic usage

```python
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2", temperature=0)

async def main():
    async for event in llm.astream_events(
        [SystemMessage("You are a helpful assistant."),
         HumanMessage("Capital of India?")],
        version="v2",          # v1 is deprecated, v2 is current stable
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

## Concurrent consumption with `asyncio.gather()`

Run multiple event streams at the same time — each processes independently:

```python
async def collect_tokens(question):
    tokens = []
    async for event in llm.astream_events([HumanMessage(question)], version="v2"):
        if event["event"] == "on_chat_model_stream":
            tokens.append(event["data"]["chunk"].content)
    return "".join(tokens)

async def main():
    # both run concurrently — not one after the other
    answer1, answer2 = await asyncio.gather(
        collect_tokens("Capital of India?"),
        collect_tokens("Capital of Japan?"),
    )
    print("India:", answer1)
    print("Japan:", answer2)

asyncio.run(main())
```

## Tool call visibility with `bind_tools`

With plain `bind_tools` (no full agent), tools do not execute — the model only decides to call them. Check `on_chat_model_end` to see the decision:

```python
async for event in llm_with_tools.astream_events(messages, version="v2"):
    if event["event"] == "on_chat_model_end":
        output = event["data"]["output"]
        if output.tool_calls:
            for tc in output.tool_calls:
                print(f"Tool call decided: {tc['name']}({tc['args']})")
```

`on_tool_start` / `on_tool_end` only fire when tools actually execute inside a full agent (Phase 2+).

## Gotchas

- Always async — there is no sync version of `.astream_events()`.
- Always pass `version="v2"` — `v1` is deprecated and shows a warning.
- Filter by `event["event"]` — there can be many noisy chain/wrapper events.
- `on_tool_start` / `on_tool_end` only fire inside a running agent, not with plain `bind_tools`.
- `asyncio.gather()` runs multiple event streams concurrently — don't use sequential `await` if you want parallelism.
