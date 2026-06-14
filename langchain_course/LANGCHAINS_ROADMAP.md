# learnAgents — Master Roadmap

> **Goal:** Build production-grade agents with LangChain + LangGraph + LangSmith (Python).
> **Strategy:** Topics ordered by learning dependency, NOT by doc layout. LangGraph + LangSmith get deep coverage; classic LangChain pieces get just enough to support agent work. Optional / reference-only material is collapsed at the bottom of each phase.
>
> Per-topic loop: read official doc → I write `notes/<topic>.md` (concept + when-to-use + gotchas) → runnable `examples/<topic>.py` → 1–2 problems in `exercises/<topic>.md` → you solve → we resolve doubts → check off.

Legend: `[ ]` todo · `[x]` done · `[~]` in progress · 🔑 critical path · 🧪 hands-on project · 📖 reference only (skim) · 🧠 LangChain stack skill · 🛠️ productization skill · 🧰 ambient skill (always available)

**Ambient skills (used across all phases)** 🧰
- **brainstorming** — runs at the start of every 🧪 project (design before code, hard gate)
- **find-skills** — used when a gap appears and we need a new skill
- **improve-codebase-architecture** — runs at the end of every 🧪 project (refactor checkpoint)
- **python-error-handling**, **python-design-patterns** — cross-cutting code quality
- **obsidian-markdown** — note format
- **langgraph-docs** — doc fetcher for any LangGraph question

**LangChain stack skills** 🧠 (mapped per phase)
- **langchain-middleware** → Phases 2, 5 — HITL approve/edit/reject + custom middleware hooks
- **langgraph-persistence** → Phase 4 — checkpointers, threads, time travel, Store, subgraph scoping
- **langgraph-human-in-the-loop** → Phase 4 — `interrupt()`, `Command(resume=)`, idempotency
- **langchain-rag** → Phase 5 — full RAG pipeline
- **langsmith-evaluator** → Phase 7 — offline/online evaluators, LLM-as-judge, trajectories
- **deep-agents-memory** → Phase 9 — StateBackend / StoreBackend / CompositeBackend / FilesystemBackend

**Productization skills** 🛠️ (Phase 10 + capstone)
- **fastapi-templates** — production async FastAPI scaffold
- **python-backend** — JWT/OAuth, async SQLAlchemy, Redis/Upstash, rate limiting
- **fastapi-python** — FastAPI style guide (companion to templates)
- **python-mcp-server-generator** — expose your agent as an MCP server
- **architecture-patterns** — Clean / Hexagonal / DDD for multi-module agents

**Agent-side tool skills** 🤖 (the agents themselves will use these)
- **firecrawl-search** — real web search agents can call
- **agent-browser** — browser automation agents can drive
- **web-research** — multi-subagent research orchestration pattern

**Situational / on-demand**
- **ui-ux-pro-max** — only if an agent gets a frontend (Phase 5 Frontend or Studio companion)
- **remotion-best-practices** — only if a "video generation agent" becomes a product
- **python-performance-optimization** — late-stage tuning
- **langfuse** — alternative observability (we use LangSmith)

---

## Phase 0 — Setup (1 short session) ✅

- [x] 🔑 Python env, `requirements.txt`, `.env` keys (OpenAI/Anthropic + LANGSMITH_API_KEY) — uv .venv, langchain 1.3.4, langgraph 1.2.4, langsmith 0.8.9
- [x] 🔑 Verify LangSmith tracing works on a hello-world LLM call — [examples/00_setup/hello_trace.py](examples/00_setup/hello_trace.py) returns 'pong'
- [x] 🔑 [LangChain Install](https://docs.langchain.com/oss/python/langchain/install)

---

## Phase 1 — LangChain core primitives (refresher, fast) 🧰 python-error-handling, python-design-patterns

Skip-able if confident, but every later concept assumes these.

- [ ] 🔑 [Overview](https://docs.langchain.com/oss/python/langchain/overview)
- [ ] 🔑 [Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart) — first runnable agent
- [ ] 🔑 [Models](https://docs.langchain.com/oss/python/langchain/models) — chat models, params, providers
- [ ] 🔑 [Messages](https://docs.langchain.com/oss/python/langchain/messages) — Human/AI/System/Tool message types
- [ ] 🔑 [Tools](https://docs.langchain.com/oss/python/langchain/tools) — `@tool`, schemas, tool calling
- [ ] 🔑 [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) — Pydantic, `with_structured_output`
- [ ] [Streaming](https://docs.langchain.com/oss/python/langchain/streaming)
- [ ] [Event streaming](https://docs.langchain.com/oss/python/langchain/event-streaming)

🧪 **Project 1:** Build a tool-using assistant (3+ tools, structured output, streaming). Uses 🤖 **firecrawl-search** as one of the tools. Start with 🧰 **brainstorming**, end with 🧰 **improve-codebase-architecture**.

---

## Phase 2 — LangChain Agents layer 🧠 langchain-middleware

The high-level `create_agent` abstraction — the easiest path to a working agent before we go under the hood with LangGraph.

- [ ] 🔑 [Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [ ] 🔑 [Runtime](https://docs.langchain.com/oss/python/langchain/runtime) — config, callbacks
- [ ] 🔑 [Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)
- [ ] 🔑 [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [ ] [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [ ] 🔑 [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [ ] [Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails)

---

## Phase 3 — LangGraph fundamentals 🔑 (deep) 🧠 langgraph-docs

Where production agents really live. We slow down here.

- [ ] 🔑 [Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [ ] 🔑 [Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [ ] 🔑 [Choosing Graph API vs Functional API](https://docs.langchain.com/oss/python/langgraph/choosing-apis)
- [ ] 🔑 [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — when graph, when agent
- [ ] 🔑 [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) — nodes, edges, state, conditional edges
- [ ] [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api)
- [ ] 🔑 [Pregel runtime](https://docs.langchain.com/oss/python/langgraph/pregel) — how execution actually works
- [ ] 🔑 [Use the Graph API (how-to)](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [ ] [Use the Functional API (how-to)](https://docs.langchain.com/oss/python/langgraph/use-functional-api)

🧪 **Project 2:** Rebuild Project 1 as an explicit LangGraph graph. Start with 🧰 **brainstorming** (what state shape? what nodes?), end with 🧰 **improve-codebase-architecture**.

---

## Phase 4 — LangGraph production features 🔑 (deep) 🧠 langgraph-persistence, langgraph-human-in-the-loop

- [ ] 🔑 [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — checkpointers, threads
- [ ] 🔑 [Add memory (how-to)](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [ ] 🔑 [Interrupts / HITL](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [ ] 🔑 [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [ ] [Event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)
- [ ] 🔑 [Subgraphs (how-to)](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [ ] [Time travel (how-to)](https://docs.langchain.com/oss/python/langgraph/use-time-travel)
- [ ] 🔑 [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)
- [ ] [Testing graphs](https://docs.langchain.com/oss/python/langgraph/test)

🧪 **Project 3:** Stateful multi-step agent with checkpointing + a human approval interrupt. Start with 🧰 **brainstorming**; consider 🛠️ **architecture-patterns** if structure starts feeling tangled; close with 🧰 **improve-codebase-architecture**.

---

## Phase 5 — Advanced LangChain patterns (built atop LangGraph) 🧠 langchain-middleware, langchain-rag · 🛠️ architecture-patterns

- [ ] 🔑 [Middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [ ] [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [ ] [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [ ] 🔑 [Multi-agent overview](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [ ] 🔑 [Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) + [tutorial](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)
- [ ] [Skills](https://docs.langchain.com/oss/python/langchain/multi-agent/skills) + [SQL assistant](https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant)
- [ ] 🔑 [Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs) + [customer support](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support)
- [ ] [Router](https://docs.langchain.com/oss/python/langchain/multi-agent/router) + [KB router](https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base)
- [ ] [Custom workflow](https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow)
- [ ] 🔑 [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
- [ ] 🔑 [RAG](https://docs.langchain.com/oss/python/langchain/rag)
- [ ] [Knowledge base / semantic search](https://docs.langchain.com/oss/python/langchain/knowledge-base)

---

## Phase 6 — LangSmith Observability 🔑

Wire into the Phase 3/5 projects — debug real code, not toys.

- [ ] 🔑 [Observability concepts](https://docs.langchain.com/langsmith/observability-concepts)
- [ ] 🔑 [Tracing quickstart](https://docs.langchain.com/langsmith/observability-quickstart)
- [ ] 🔑 [Trace an LLM app tutorial](https://docs.langchain.com/langsmith/observability-llm-tutorial)
- [ ] [Log LLM calls](https://docs.langchain.com/langsmith/log-llm-trace) · [retriever traces](https://docs.langchain.com/langsmith/log-retriever-trace)
- [ ] 🔑 [Custom instrumentation](https://docs.langchain.com/langsmith/annotate-code)
- [ ] [Metadata & tags](https://docs.langchain.com/langsmith/add-metadata-tags)
- [ ] [Filter traces](https://docs.langchain.com/langsmith/filter-traces-in-application) · [sampling](https://docs.langchain.com/langsmith/sample-traces)
- [ ] [Distributed tracing](https://docs.langchain.com/langsmith/distributed-tracing)
- [ ] 🔑 [Runs (spans)](https://docs.langchain.com/langsmith/runs) · [data format](https://docs.langchain.com/langsmith/run-data-format)
- [ ] [Threads](https://docs.langchain.com/langsmith/query-threads)
- [ ] [Dashboards](https://docs.langchain.com/langsmith/dashboards) · [Alerts](https://docs.langchain.com/langsmith/alerts) · [Rules](https://docs.langchain.com/langsmith/rules)
- [ ] [Annotation queues](https://docs.langchain.com/langsmith/annotation-queues) · [user feedback](https://docs.langchain.com/langsmith/attach-user-feedback)
- [ ] [Insights](https://docs.langchain.com/langsmith/insights)

---

## Phase 7 — LangSmith Evaluation 🔑 🧠 langsmith-evaluator

- [ ] 🔑 [Evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts) + [types](https://docs.langchain.com/langsmith/evaluation-types)
- [ ] 🔑 [Evaluation quickstart](https://docs.langchain.com/langsmith/evaluation-quickstart)
- [ ] 🔑 [Evaluate an LLM application](https://docs.langchain.com/langsmith/evaluate-llm-application)
- [ ] 🔑 [Evaluate a graph](https://docs.langchain.com/langsmith/evaluate-graph)
- [ ] [Evaluate a chatbot](https://docs.langchain.com/langsmith/evaluate-chatbot-tutorial)
- [ ] 🔑 [Evaluate a RAG application](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)
- [ ] 🔑 [Evaluate a complex agent](https://docs.langchain.com/langsmith/evaluate-complex-agent)
- [ ] [Evaluate intermediate steps](https://docs.langchain.com/langsmith/evaluate-on-intermediate-steps)
- [ ] [Pairwise eval](https://docs.langchain.com/langsmith/evaluate-pairwise)
- [ ] 🔑 [Async evals](https://docs.langchain.com/langsmith/evaluation-async) · [Local](https://docs.langchain.com/langsmith/local) · [Pytest](https://docs.langchain.com/langsmith/pytest)
- [ ] [Multi-turn simulation](https://docs.langchain.com/langsmith/multi-turn-simulation)
- [ ] [Repetitions](https://docs.langchain.com/langsmith/repetition) · [openevals](https://docs.langchain.com/langsmith/openevals)
- [ ] [Analyze experiment](https://docs.langchain.com/langsmith/analyze-an-experiment) · [Compare](https://docs.langchain.com/langsmith/compare-experiment-results)
- [ ] 🔑 [Manage evaluators](https://docs.langchain.com/langsmith/evaluators) · [Code (SDK)](https://docs.langchain.com/langsmith/code-evaluator-sdk) · [LLM-as-judge (SDK)](https://docs.langchain.com/langsmith/llm-as-judge-sdk)
- [ ] [Composite evaluators](https://docs.langchain.com/langsmith/composite-evaluators-sdk)
- [ ] [Few-shot evaluators](https://docs.langchain.com/langsmith/create-few-shot-evaluators)
- [ ] 🔑 Online evaluators ([code](https://docs.langchain.com/langsmith/online-evaluations-code), [LLM-as-judge](https://docs.langchain.com/langsmith/online-evaluations-llm-as-judge))
- [ ] 🔑 [Manage datasets](https://docs.langchain.com/langsmith/manage-datasets) · [programmatically](https://docs.langchain.com/langsmith/manage-datasets-programmatically)
- [ ] [Dataset transformations](https://docs.langchain.com/langsmith/dataset-transformations)
- [ ] [Run backtests](https://docs.langchain.com/langsmith/run-backtests-new-agent)
- [ ] [CI/CD with LangSmith](https://docs.langchain.com/langsmith/cicd-pipeline-example)

🧪 **Project 4:** Build a real eval suite for one of your earlier projects — dataset + 3 evaluators + run from pytest. Start with 🧰 **brainstorming** (which metrics actually matter for this agent?).

---

## Phase 8 — LangSmith Prompt Engineering

- [ ] [Prompt engineering concepts](https://docs.langchain.com/langsmith/prompt-engineering-concepts)
- [ ] 🔑 [Prompt engineering quickstart](https://docs.langchain.com/langsmith/prompt-engineering-quickstart)
- [ ] 🔑 [Create a prompt](https://docs.langchain.com/langsmith/create-a-prompt) · [Manage](https://docs.langchain.com/langsmith/manage-prompts) · [Programmatic](https://docs.langchain.com/langsmith/manage-prompts-programmatically)
- [ ] [Prompt template format](https://docs.langchain.com/langsmith/prompt-template-format)
- [ ] [Multimodal in prompts](https://docs.langchain.com/langsmith/multimodal-content)
- [ ] [Sync prompts with GitHub](https://docs.langchain.com/langsmith/prompt-commit)
- [ ] [Model configurations](https://docs.langchain.com/langsmith/model-configurations)

---

## Phase 9 — Deep Agents (advanced agent framework) 🧠 deep-agents-memory · 🤖 firecrawl-search, agent-browser

Builds on everything above. Skip if you've shipped a production agent already.

- [ ] 🔑 [Overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [ ] 🔑 [Quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart)
- [ ] 🔑 [Customization](https://docs.langchain.com/oss/python/deepagents/customization)
- [ ] [Harness](https://docs.langchain.com/oss/python/deepagents/harness)
- [ ] [Profiles](https://docs.langchain.com/oss/python/deepagents/profiles)
- [ ] [Models](https://docs.langchain.com/oss/python/deepagents/models)
- [ ] 🔑 [Tools](https://docs.langchain.com/oss/python/deepagents/tools)
- [ ] [Skills](https://docs.langchain.com/oss/python/deepagents/skills)
- [ ] 🔑 [Subagents](https://docs.langchain.com/oss/python/deepagents/subagents) · [Async](https://docs.langchain.com/oss/python/deepagents/async-subagents)
- [ ] 🔑 [Memory](https://docs.langchain.com/oss/python/deepagents/memory)
- [ ] [Backends](https://docs.langchain.com/oss/python/deepagents/backends)
- [ ] 🔑 [Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes) · [Interpreters](https://docs.langchain.com/oss/python/deepagents/interpreters)
- [ ] [Permissions](https://docs.langchain.com/oss/python/deepagents/permissions)
- [ ] 🔑 [Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)
- [ ] [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering)
- [ ] [Streaming](https://docs.langchain.com/oss/python/deepagents/streaming)
- [ ] 🔑 [Going to production](https://docs.langchain.com/oss/python/deepagents/going-to-production)

🧪 **Project 5 (capstone):** Build a Deep Agent with subagents + sandbox + HITL + full LangSmith tracing + eval suite. Subagents use 🤖 **firecrawl-search** / 🤖 **agent-browser** as tools. Start with 🧰 **brainstorming**, structure with 🛠️ **architecture-patterns**, close with 🧰 **improve-codebase-architecture**.

---

## Phase 10 — Deployment & Productization 🛠️ fastapi-templates, python-backend, fastapi-python, python-mcp-server-generator

This phase has two tracks: **(A) LangGraph-native deployment** via Agent Server + Studio (managed path), and **(B) Custom productization** — wrap the agent in your own FastAPI service when you need auth, rate limits, billing, or non-graph endpoints.

### A. LangGraph-native deployment

- [ ] 🔑 [Deployment overview](https://docs.langchain.com/langsmith/deployment)
- [ ] 🔑 [Deploy to cloud](https://docs.langchain.com/langsmith/deployment-quickstart)
- [ ] [Deployment components](https://docs.langchain.com/langsmith/components)
- [ ] 🔑 [LangGraph Studio](https://docs.langchain.com/oss/python/langgraph/studio) · [Get started](https://docs.langchain.com/langsmith/quick-start-studio)
- [ ] 🔑 [Agent Server](https://docs.langchain.com/langsmith/agent-server) · [Scale](https://docs.langchain.com/langsmith/agent-server-scale)
- [ ] [Streaming API](https://docs.langchain.com/langsmith/streaming) · [Stateless runs](https://docs.langchain.com/langsmith/stateless-runs) · [Cron jobs](https://docs.langchain.com/langsmith/cron-jobs)
- [ ] [Double texting](https://docs.langchain.com/langsmith/double-texting)
- [ ] [HITL via server API](https://docs.langchain.com/langsmith/add-human-in-the-loop)
- [ ] [Custom auth](https://docs.langchain.com/langsmith/add-custom-auth) · [middleware](https://docs.langchain.com/langsmith/custom-middleware) · [routes](https://docs.langchain.com/langsmith/custom-routes)
- [ ] [Semantic search in deployment](https://docs.langchain.com/langsmith/semantic-search)
- [ ] [Managed Deep Agents quickstart](https://docs.langchain.com/langsmith/managed-deep-agents-quickstart) · [Deploy](https://docs.langchain.com/langsmith/managed-deep-agents-deploy)
- [ ] [Local dev & testing](https://docs.langchain.com/langsmith/local-dev-testing)

### B. Custom productization (wrap your own FastAPI around an agent)

- [ ] 🛠️ **fastapi-templates** — scaffold async FastAPI project structure
- [ ] 🛠️ **python-backend** — JWT/OAuth auth on agent endpoints
- [ ] 🛠️ **python-backend** — async SQLAlchemy for conversation/run history
- [ ] 🛠️ **python-backend** — Redis/Upstash rate limiting + caching
- [ ] 🛠️ **python-mcp-server-generator** — expose the agent as an MCP server (so other agents/Claude Desktop can call it)
- [ ] Containerize (Dockerfile) and ship

🧪 **Project 6 (productization):** Take the capstone Deep Agent from Project 5 and wrap it in a FastAPI service with auth, rate limiting, and an MCP endpoint.

---

## 📖 Reference / Optional (skim only)

You probably don't need these to ship production agents — read on demand.

- LangChain Frontend / Generative UI ([overview](https://docs.langchain.com/oss/python/langchain/frontend/overview), [generative-ui](https://docs.langchain.com/oss/python/langchain/frontend/generative-ui), [assistant-ui](https://docs.langchain.com/oss/python/langchain/frontend/integrations/assistant-ui), etc.)
- LangChain testing deep-dive ([unit](https://docs.langchain.com/oss/python/langchain/test/unit-testing), [integration](https://docs.langchain.com/oss/python/langchain/test/integration-testing), [evals](https://docs.langchain.com/oss/python/langchain/test/evals))
- Deep Agents Frontend ([overview](https://docs.langchain.com/oss/python/deepagents/frontend/overview), todo list, sandbox)
- LangSmith Self-hosted (entire section — only if self-hosting)
- LangSmith Fleet (separate product surface)
- LangSmith LLM Gateway (private beta)
- LangSmith Chat
- Auth/RBAC/ABAC, billing, audit logs, data export
- Engine ([overview](https://docs.langchain.com/langsmith/engine), webhooks) — separate failure-debugging product

---

## Tracking

I update this file's checkboxes after each completed topic. `notes/`, `examples/`, `exercises/` populate alongside.
