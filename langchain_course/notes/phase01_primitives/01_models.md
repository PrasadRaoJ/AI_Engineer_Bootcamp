# Models — Chat Models

## Concept

A **ChatModel** is LangChain's unified interface to any LLM provider (Ollama, OpenAI, Anthropic, etc.).
You call `.invoke()`, `.stream()`, or `.batch()` — the same API regardless of the provider underneath.

Models are the reasoning engine of agents — they decide which tools to call, how to interpret results, and when to give a final answer.

## Core import

```python
from langchain.chat_models import init_chat_model
import os

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2
```

## Key parameters

| Param | What it does | Default |
|-------|-------------|---------|
| `model` | Which model to run | required |
| `temperature` | Randomness — 0=deterministic, 1=creative | 0.7 |
| `num_predict` | Max tokens (Ollama-specific) | -1 (unlimited) |
| `max_tokens` | Max tokens (standard LangChain param) | provider default |
| `timeout` | Request deadline in seconds | None |
| `max_retries` | Retry on rate limits / server errors | 6 |

## Flow

```
┌─────────────────────┐
│    Your Python code │
└──────────┬──────────┘
           │  .invoke() / .stream() / .batch()
           ▼
┌─────────────────────┐
│  init_chat_model    │
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
# 1. invoke — wait for full reply
response = llm.invoke("What is 2+2?")
print(response.content)

# 2. stream — yield tokens as they arrive
for chunk in llm.stream("Tell me a joke"):
    print(chunk.content, end="", flush=True)

# 3. batch — multiple inputs, returns list in same order
responses = llm.batch(["Capital of India?", "Capital of Japan?"])
```

## batch vs batch_as_completed

```python
# batch — returns list in INPUT order, waits for all
responses = llm.batch(["Q1", "Q2", "Q3"])

# batch_as_completed — returns (index, result) tuples as each finishes
# order is NOT guaranteed — use index to reconstruct
for idx, response in llm.batch_as_completed(["Q1", "Q2", "Q3"]):
    print(f"[{idx}]", response.content)
```

Use `batch_as_completed` when inputs vary in length and you want to process results as soon as they're ready.

## When to use

- `.invoke()` — single request, need the full response before continuing
- `.stream()` — chat UIs, long responses, show output as it arrives
- `.batch()` — many inputs, results needed in order
- `.batch_as_completed()` — many inputs, process each result immediately as it finishes

## Switching providers

### Option A — `init_chat_model` (preferred for provider switching)

```python
from langchain.chat_models import init_chat_model

# "provider:model" string — no different imports needed
llm = init_chat_model("openai:gpt-4o", temperature=0)
llm = init_chat_model("anthropic:claude-sonnet-4-6", temperature=0)
llm = init_chat_model("llama3.2", model_provider="ollama", temperature=0)
```

Same `.invoke()` / `.stream()` / `.batch()` API — only the string changes.

### Option B — provider-specific classes (explicit, used in this course)

#### Ollama (local — used in this course)
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)
# model: "llama3.2" / "gemma3" / "qwen3.5:2b"
# no API key needed
```

#### OpenAI
```python
# pip install langchain-openai
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
# needs: OPENAI_API_KEY in .env
```

#### Anthropic (Claude)
```python
# pip install langchain-anthropic
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
# needs: ANTHROPIC_API_KEY in .env
```

#### Google Gemini
```python
# pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
# needs: GOOGLE_API_KEY in .env
```

#### Groq (fast inference)
```python
# pip install langchain-groq
from langchain_groq import ChatGroq

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
# needs: GROQ_API_KEY in .env
```

### Provider comparison

| Provider | Speed | Cost | Best for |
|----------|-------|------|----------|
| Ollama | medium | free | local dev, learning, no API key |
| Groq | very fast | free tier | quick prototyping with strong models |
| OpenAI | fast | paid | production, best tool calling |
| Anthropic | fast | paid | long context, safety-focused |
| Gemini | fast | free tier | multimodal, Google ecosystem |

## Gotchas

- Response is an `AIMessage` object, not a plain string — use `.content` to get the text.
- `temperature=0` makes the model deterministic — useful for tests and structured output.
- `num_predict` is Ollama-specific; other providers use `max_tokens`.
- `max_retries` default is 6 — increase to 10–15 for long-running agents on unreliable networks.
- Client errors (401, 404) are never retried — only rate limits (429) and server errors (5xx).
- `batch_as_completed()` returns results out of order — always use the index to map back to inputs.
