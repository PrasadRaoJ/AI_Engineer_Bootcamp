# Streaming

## Concept

Instead of waiting for the full response, streaming yields **chunks** as the model generates them — token by token. Useful for chat UIs and long responses where you want output to appear immediately.

## Flow

```
.invoke()
┌──────┐        ┌─────┐        ┌─────────────────────────────────┐
│ User │──msg──►│ LLM │  wait  │ "The capital of France is Paris"│
└──────┘        └─────┘ ......►└─────────────────────────────────┘
                               full response arrives all at once


.stream()  ← sync, blocks the thread while waiting for each chunk
┌──────┐        ┌─────┐        ┌───────┐ ┌──────────┐ ┌──────────────┐ ┌────────┐
│ User │──msg──►│ LLM │───────►│"The " │ │"capital "│ │"of France "  │ │"Paris" │
└──────┘        └─────┘        └───────┘ └──────────┘ └──────────────┘ └────────┘
                               chunks arrive one by one → print each immediately
                               ⚠ nothing else can run while waiting for next chunk


.astream()  ← async, frees the thread between chunks (use in web servers / FastAPI)
┌────────┐      ┌─────┐        ┌───────┐ ┌──────────┐ ┌──────────────┐ ┌────────┐
│ User A │─msg─►│ LLM │───────►│"The " │ │"capital "│ │"of France "  │ │"Paris" │
└────────┘      └─────┘        └───────┘ └──────────┘ └──────────────┘ └────────┘
                               between each chunk, server is FREE →
┌────────┐
│ User B │─msg─► gets handled while User A waits for next chunk ✅
└────────┘
┌────────┐
│ User C │─msg─► gets handled while User A waits for next chunk ✅
└────────┘
                ⚠ with .stream() User B and C would be stuck waiting in queue
```

## Methods

| Method | Type | Use when |
|--------|------|----------|
| `.stream()` | sync | scripts, CLI — single user only |
| `.astream()` | async | FastAPI, web servers — multiple users simultaneously |

## **`.stream()`** — sync streaming

```python
for chunk in llm.stream("Tell me about India in 3 sentences."):
    print(chunk.content, end="", flush=True)  # print each token immediately
print()  # newline at the end
```

## **`.astream()`** — async streaming

```python
import asyncio

async def main():
    async for chunk in llm.astream("Tell me about India in 3 sentences."):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(main())
```

## Streaming with messages

Works the same way with a message list:

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("You are a Slipkart customer support agent."),
    HumanMessage("What is your return policy?"),
]

for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

## AIMessageChunk — what each chunk looks like

Each chunk is an `AIMessageChunk`, not a full `AIMessage`:

```python
chunks = list(llm.stream("What is 2+2?"))

chunk = chunks[1]
chunk.content          # plain text of this token, e.g. " 4"
chunk.content_blocks   # normalized list — [{"type": "text", "text": " 4"}]
                       # useful for multimodal (text, tool_call_chunk, reasoning)
```

`content_blocks` normalizes provider-specific formats — use it when you need to distinguish text from tool call chunks in the same stream.

## Agent streaming (Phase 2)

When using `create_agent`, streaming gains two new dimensions:

```python
# stream_mode="updates" — yields each node's output (model or tools)
for chunk in agent.stream(input, stream_mode="updates"):
    ...

# stream_mode=["messages", "updates"] — yields tuples (mode, data)
for mode, data in agent.stream(input, stream_mode=["messages", "updates"]):
    if mode == "messages":
        ...  # AIMessageChunk tokens
    elif mode == "updates":
        ...  # full node output (model call or tool result)
```

These are only available on a compiled agent graph — not on raw `llm.stream()`.

## Gotchas

- Each chunk is an `AIMessageChunk`, not a full `AIMessage` — `.content` may be empty `""` on the first/last chunk.
- `flush=True` is required to print tokens immediately — without it Python buffers output and defeats the purpose.
- **`.stream()`** blocks the thread — use **`.astream()`** in async contexts (FastAPI, etc.).
- `version=` and `stream_mode=` are agent-level params — passing them to raw `llm.stream()` raises a `TypeError`.
- Avoid combining raw `llm.stream()` with `.with_structured_output()` — partial objects are hard to use. LangChain supports it via accumulation, but wait until you need it.
