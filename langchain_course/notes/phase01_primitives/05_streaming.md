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

## **`.invoke()`** — full response, blocking

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

## Gotchas

- Each chunk is an `AIMessageChunk`, not a full `AIMessage` — `.content` may be an empty string `""` on the first/last chunk.
- `flush=True` is required to print tokens immediately — without it, Python buffers the output and defeats the purpose.
- **`.stream()`** is blocking — use **`.astream()`** in async contexts (FastAPI, etc.).
- Don't use streaming with `.with_structured_output()` — you need the complete JSON before it can be parsed.
