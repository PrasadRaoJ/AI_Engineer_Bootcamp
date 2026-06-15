# LangChain vs LangGraph — The Big Picture

---

## What we learned in LangChain (Phases 1–2)

| Topic | What it gave us |
|-------|----------------|
| Models, Messages, Tools | Talk to LLMs, define tools the model can call |
| Structured output | Force model to return typed Pydantic objects |
| Streaming | Stream tokens as they arrive |
| `create_agent` | One-line agent that auto-loops: LLM → tool → LLM → tool → done |
| Middleware (`@before_agent`, `@after_agent`, `@wrap_model_call`) | Hook into the agent loop without rewriting it |
| `@dynamic_prompt` | Per-request system prompt — inject user ID, role, session data before every LLM call |
| Role-based tool access | `@wrap_model_call` + `request.override(tools=filtered)` — show/hide tools per role at model-call time |
| User-wise memory (`InMemoryStore`) | Per-user long-term memory across threads — store preferences, history, profile |
| HITL, PII, guardrails | Built-in safety via middleware |
| `InMemorySaver` | Short-term memory — same thread remembers previous turns (checkpointing) |

**Mental model:** LangChain gives you a **black box agent**. You hand it tools and middleware, it figures out the loop.

---

## What we'll learn in LangGraph (Phases 3–4)

| Topic | What it gives us |
|-------|-----------------|
| `StateGraph` | Define the graph yourself — nodes, edges, conditions |
| Nodes | Each step is an explicit Python function |
| Edges | You decide what runs next (linear, conditional, parallel) |
| State | A typed dict that flows through every node |
| Checkpointers | Pause, resume, replay — built into the graph |
| Subgraphs | Nest one graph inside another |
| `interrupt()` | Stop mid-graph and wait for human input |
| Fault tolerance | Retry failed nodes without restarting the whole graph |

**Mental model:** LangGraph gives you the **circuit board**. You draw the wires yourself.

---

## Key Differences

| | LangChain `create_agent` | LangGraph `StateGraph` |
|--|--------------------------|------------------------|
| **Control** | Framework decides the loop | You decide every step |
| **Visibility** | Black box — hard to inspect mid-run | Every node is explicit code |
| **Flow** | Linear: think → act → think → act | Any shape: branches, loops, parallel |
| **State** | Hidden inside the agent | Explicit typed dict you own |
| **Complexity** | Simple to set up | More code, more control |
| **Best for** | Single-task chat agents | Multi-step workflows with decisions |

---

## LangChain Drawbacks → Fixed by LangGraph

**1. You can't control the loop**
LangChain's agent loop is: LLM decides → calls tool → LLM decides → ... until done.
You can't say "after step 2, always go to step 5" or "if this fails, retry step 3."
→ LangGraph: you wire the exact path.

**2. No branching**
LangChain can't do "if the model classifies this as fraud, go to fraud path, else go to payout path."
→ LangGraph: conditional edges do exactly this.

**3. Hard to pause mid-workflow**
LangChain HITL fires before a tool call but can't pause in the middle of a 10-step process and resume from step 6.
→ LangGraph: `interrupt()` pauses at any node, `Command(resume=)` continues from exactly there.

**4. Parallel steps aren't possible**
LangChain runs tools sequentially inside one agent loop.
→ LangGraph: fan-out edges run multiple nodes in parallel, fan back in with a join.

**5. No retry on partial failure**
If step 7 of 10 fails in a LangChain agent, you restart from step 1.
→ LangGraph: checkpoint after every node — retry from the failed node only.

---

## When to use which

### Use LangChain when:
- You need a **conversational assistant** that answers questions and calls tools
- The task is **open-ended** — you don't know which tools will be called in what order
- You want **fast setup** with guardrails (PII, HITL, role filtering via middleware)

**Example 1 — Customer support bot**
"Reset my password" → agent picks `reset_password` tool → done.
The order of tool calls doesn't matter, the agent decides. LangChain is perfect.

**Example 2 — Travel booking assistant (Voyago)**
User says "book me a flight to Delhi". Agent searches, then books, then confirms.
Loose sequence, open-ended conversation. LangChain handles it fine.

---

### Use LangGraph when:
- You have a **fixed workflow** with steps that must happen in order
- You need **branching** — different paths based on a condition
- You need **parallel processing** — run two things at the same time
- You need **pause and resume** mid-workflow (approval flows, long-running tasks)

**Example 1 — Insurance claim pipeline**
File claim → validate documents → auto-approve if < ₹10k, else → human review → payout.
This is a fixed graph with a branch. LangGraph owns this.

**Example 2 — Job application screener**
Receive CV → parse skills (node 1) → score against job (node 2) → if score > 80: shortlist (node 3a), else: reject (node 3b) → send email (node 4).
Branching + fixed steps = LangGraph.

**Example 3 — Multi-step research agent**
Search web (parallel: 3 sources at once) → summarise each → merge → generate report → human approves → publish.
Parallel nodes + human pause mid-graph = LangGraph.

---

## Where do dynamic prompts, role access, and user memory go in LangGraph?

These are **LangChain middleware concepts** — they exist because `create_agent` is a black box and middleware is the only way to inject behaviour. In LangGraph you don't need special hooks because you write the nodes yourself.

| LangChain (middleware) | LangGraph equivalent |
|------------------------|---------------------|
| `@dynamic_prompt` — inject user_id into system prompt before LLM call | A `build_prompt` node that reads state and sets `state["system"]` before the LLM node |
| `@wrap_model_call` — filter tool list per role | The LLM node itself receives `tools=` filtered by `state["role"]` — plain Python |
| `InMemoryStore` — per-user long-term memory | Same `InMemoryStore`, but your nodes read/write it directly — no middleware needed |
| `InMemorySaver` — short-term thread memory | Same `InMemorySaver` passed to the graph as `checkpointer=` — works identically |

**Key insight:** The patterns (dynamic prompts, role filtering, user memory) carry over to LangGraph. Only the *mechanism* changes — from middleware hooks to explicit nodes.

---

## One-line summary

> **LangChain** — give the model tools and let it figure out the plan.
> **LangGraph** — you draw the plan, the model executes each step.
