# Models — Chat Models

## Concept

A **ChatModel** is LangChain's unified interface to any LLM provider (Ollama, OpenAI, Anthropic, etc.).
You call `.invoke()`, `.stream()`, or `.batch()` — the same API regardless of the provider underneath.

## Core import (Ollama)

```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")
```

## Key parameters

| Param | What it does | Default |
|-------|-------------|---------|
| `model` | Which model to run | required |
| `temperature` | Randomness 0=deterministic, 1=creative | 0.7 |
| `num_predict` | Max tokens to generate | -1 (unlimited) |

## Flow

```
┌─────────────────────┐
│    Your Python code │
└──────────┬──────────┘
           │  .invoke() / .stream() / .batch()
           ▼
┌─────────────────────┐
│   ChatOllama (LLM)  │
└──────────┬──────────┘
           │
     ┌─────┴──────────────────────┐
     │ .invoke()                  │ .stream()               .batch()
     ▼                            ▼                         ▼
┌───────────┐          ┌────────────────────┐    ┌─────────────────────┐
│ AIMessage │          │ chunks (generator) │    │ list of AIMessages  │
│ .content  │          │ .content per chunk │    │ one per input       │
└───────────┘          └────────────────────┘    └─────────────────────┘
```

## Three ways to call

```python
# 1. Single call — returns AIMessage
response = llm.invoke("What is 2+2?")
print(response.content)

# 2. Streaming — yields chunks as they arrive
for chunk in llm.stream("Tell me a joke"):
    print(chunk.content, end="", flush=True)

# 3. Batch — multiple inputs in one call
responses = llm.batch(["What is 2+2?", "What is 3+3?"])
```

## When to use

- `.invoke()` — single request, you need the full response before continuing
- `.stream()` — chat UIs, long responses, you want to show output as it arrives
- `.batch()` — processing many inputs, more efficient than looping `.invoke()`

## Gotchas

- Response is an `AIMessage` object, not a plain string — use `.content` to get the text.
- `temperature=0` makes the model deterministic — useful for tests and structured output.
- Different providers have different param names — LangChain normalizes the common ones but not all.
