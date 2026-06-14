# Phase 0 — Environment Setup

## What we're setting up

| Tool | Purpose |
|------|---------|
| `.venv` | Isolated Python environment so packages don't collide with the system |
| `requirements.txt` | Pinned package list for reproducibility |
| Ollama | Local LLM server — runs models on-device, no API key needed for dev |
| `.env` | Secrets file (API keys) — never committed to git |
| LangSmith | Cloud tracing + eval platform — free tier is enough |

---

## Step 1 — Virtual environment

On Ubuntu/Debian, `python3-venv` is not included by default — install it first:

```bash
sudo apt install python3.14-venv -y
```

Then create and activate the venv:

```bash
cd /home/jp/Desktop/Learning/learn_langchain
python3 -m venv .venv
source .venv/bin/activate   # re-run this every new terminal session
```

**When to use:** Always. Every terminal session working on this project.

**Gotcha:** If you see `ModuleNotFoundError` for any package, the venv is probably not activated. Check with `which python` — it must point inside `.venv/`.

**Gotcha:** On Ubuntu/Debian, `python3 -m venv` silently fails if `python3.14-venv` isn't installed — install it with `apt` before creating the venv.

---

## Step 2 — Install packages

```bash
pip install -r requirements.txt
```

Core packages and why:

| Package | Why |
|---------|-----|
| `langchain` | Core primitives — ChatModel, Tools, LCEL chains |
| `langchain-community` | Community integrations (Ollama, etc.) |
| `langchain-ollama` | First-class Ollama integration for LangChain |
| `langgraph` | Graph-based agent runtime |
| `langsmith` | Tracing + eval SDK |
| `python-dotenv` | Loads `.env` into `os.environ` at runtime |
| `pydantic` | Structured output schemas |

---

## Step 3 — Ollama (local LLM)

Ollama is already installed at `/snap/bin/ollama`.

```bash
ollama serve                    # starts the local server on http://localhost:11434
ollama pull llama3.2            # download a model (do this once)
ollama list                     # confirm the model is available
```

**When to use:** Use Ollama during development/exercises so you don't burn API credits. Switch to Claude/OpenAI for final projects or when you need stronger reasoning.

**Gotcha:** `ollama serve` must be running in a background terminal before any LangChain code that calls Ollama. If you see a connection refused error, start it first.

In LangChain code:
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")
```

---

## Step 4 — .env file

Create `.env` in the project root (never commit this):

```
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=learn-langchain

# Optional — only needed for projects that use hosted models
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Load it in every Python file:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Gotcha:** `.env` must be in the directory you run the script from, or pass the path explicitly: `load_dotenv("/absolute/path/.env")`.

---

## Step 5 — LangSmith

1. Create a free account at https://smith.langchain.com
2. Go to **Settings → API Keys → Create API Key**
3. Paste it into `.env` as `LANGSMITH_API_KEY`
4. Set `LANGSMITH_PROJECT` to a project name (e.g. `learn-langchain`) — LangSmith creates it on first trace

Every run that calls an LLM will now appear in the LangSmith UI automatically when `LANGSMITH_TRACING=true`.

**When to use:** Always on during learning — seeing the trace of every call is the fastest way to understand what LangChain is actually doing under the hood.

**Gotcha:** If `LANGSMITH_TRACING` is not set, no traces are sent even if the API key is present.

---

## Verify everything works

```bash
source .venv/bin/activate
python examples/phase00_setup/hello_trace.py
```

Expected: prints `pong`.

## Default model

We use `llama3.2` via Ollama for all exercises (local, no API key needed).

```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")
```

Other available models: `gemma3`, `qwen3.5:2b`. Switch freely depending on the task.

LangSmith tracing is optional for now — add it in Phase 6 when we cover observability.
