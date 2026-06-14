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

## Switching providers

Same `.invoke()` / `.stream()` / `.batch()` API across all providers — only the import, class, and model name change.

### Ollama (local — used in this course)
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)
# change: model name — "llama3.2" / "gemma3" / "qwen3.5:2b"
# no API key needed
```

### OpenAI
```python
# pip install langchain-openai
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
# change: model name — "gpt-4o" / "gpt-4o-mini" / "gpt-3.5-turbo"
# needs: OPENAI_API_KEY in .env
```

### Anthropic (Claude)
```python
# pip install langchain-anthropic
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
# change: model name — "claude-sonnet-4-6" / "claude-haiku-4-5-20251001" / "claude-opus-4-8"
# needs: ANTHROPIC_API_KEY in .env
```

### Google Gemini
```python
# pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
# change: model name — "gemini-2.0-flash" / "gemini-1.5-pro"
# needs: GOOGLE_API_KEY in .env
```

### Groq (fast inference)
```python
# pip install langchain-groq
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# change: model name — "llama-3.3-70b-versatile" / "mixtral-8x7b-32768" / "gemma2-9b-it"
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
- Different providers have different param names — LangChain normalizes the common ones but not all.
