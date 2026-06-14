# learn_langchain

LangChain + LangGraph + LangSmith — production agent engineering.
Roadmap: `../Ai_Platform_Engr/LANGCHAINS_ROADMAP.md`

---

## Folder Structure

```
learn_langchain/
├── notes/          concept summaries, when-to-use, gotchas — one .md per topic
├── examples/       runnable .py files — one per topic
├── exercises/      problem statements per topic (.md) + your solutions (.py)
└── projects/       full hands-on projects (one folder per project)
```

Each folder is split by phase matching the roadmap:

| Folder suffix | Roadmap phase |
|---------------|--------------|
| phase01_primitives | Phase 1 — LangChain core primitives |
| phase02_agents | Phase 2 — create_agent + middleware |
| phase03_langgraph_fundamentals | Phase 3 — StateGraph, nodes, edges |
| phase04_langgraph_production | Phase 4 — checkpointers, HITL, streaming |
| phase05_advanced_patterns | Phase 5 — multi-agent, RAG, retrieval |
| phase06_langsmith_observability | Phase 6 — tracing, dashboards, alerts |
| phase07_langsmith_evaluation | Phase 7 — evals, LLM-as-judge, datasets |
| phase08_langsmith_prompts | Phase 8 — prompt management |
| phase09_deep_agents | Phase 9 — create_deep_agent, subagents |
| phase10_deployment | Phase 10 — Agent Server, FastAPI wrapping |

---

## Projects

| Folder | Phase | What you build |
|--------|-------|---------------|
| project01_tool_assistant | Phase 1 | Tool-using assistant (3+ tools, structured output, streaming) |
| project02_langgraph_rebuild | Phase 3 | Rebuild project01 as an explicit LangGraph graph |
| project03_stateful_hitl_agent | Phase 4 | Stateful multi-step agent + checkpointing + HITL approval |
| project04_eval_suite | Phase 7 | Eval suite for project03 — dataset + 3 evaluators + pytest |
| project05_deep_agent_capstone | Phase 9 | Deep Agent with subagents + sandbox + HITL + LangSmith |
| project06_fastapi_productization | Phase 10 | FastAPI wrapper + auth + rate limiting + MCP endpoint |

---

## Per-topic workflow

1. Read the official doc link from LANGCHAINS_ROADMAP.md
2. Write a note in `notes/<phase>/<topic>.md` (concept + when-to-use + gotchas)
3. Write a runnable example in `examples/<phase>/<topic>.py`
4. Solve the exercise in `exercises/<phase>/<topic>.md`
5. Check off `[x]` in LANGCHAINS_ROADMAP.md
