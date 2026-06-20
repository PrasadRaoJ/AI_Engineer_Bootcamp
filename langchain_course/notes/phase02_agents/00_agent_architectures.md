# Agent Architectures in LangChain
---

## The Big Picture

```
┌──────────────────────────────────────────────────────────────┐
│                    AGENT ARCHITECTURES                        │
│                                                               │
│   SINGLE-AGENT              MULTI-AGENT                       │
│   ────────────              ────────────                      │
│   1. ReAct                  4. Subagents (Supervisor)         │
│   2. Plan-and-Execute       5. Router                         │
│   3. Reflection             6. Handoffs                       │
│                             7. Skills                         │
└──────────────────────────────────────────────────────────────┘
```

Rule: **Start single. Add tools before adding agents. Graduate to multi-agent only when you hit clear limits.**

---

## SINGLE-AGENT PATTERNS

---

### 1. ReAct — Reasoning + Acting

**The most common pattern. What `create_agent` automates.**

#### How it works

The key property: **each step's action is decided after seeing the previous result**. A script can't do this because you don't know the branches upfront.

```
Human: "My order ORD456 shows delivered but I never received it."
              │
              ▼
     ┌─────────────────┐
     │   LLM REASONS   │  → "Delivered but not received — could be stolen,
     └────────┬────────┘       wrong address, or scan error. Check status first."
              │ tool_calls: [get_order_status(ORD456)]
              ▼
     ┌─────────────────┐
     │      ACT        │  → get_order_status() executes
     └────────┬────────┘
              │ ToolMessage: "Delivered 2 days ago. GPS: doorstep unit 12B."
              ▼
     ┌─────────────────┐
     │    OBSERVE      │  → "GPS shows unit 12B. User might be 12A. Check delivery photo."
     └────────┬────────┘
              │ tool_calls: [get_delivery_proof(ORD456)]
              ▼
     ┌─────────────────┐
     │      ACT        │  → get_delivery_proof() executes
     └────────┬────────┘
              │ ToolMessage: "Photo: package left at door labelled 12B."
              ▼
     ┌─────────────────┐
     │    OBSERVE      │  → "Wrong door, not stolen. File misdelivery, not lost package."
     └────────┬────────┘
              │ tool_calls: [file_misdelivery_report(ORD456)]
              ▼
     ┌─────────────────┐
     │      ACT        │  → file_misdelivery_report() executes
     └────────┬────────┘
              │ ToolMessage: "Report filed. Replacement dispatched."
              ▼
     ┌─────────────────┐
     │    OBSERVE      │  → tool_calls empty → DONE
     └────────┬────────┘
              ▼
     "Your package went to a neighbouring unit by mistake.
      A replacement has been dispatched — arrives in 1-2 days."
```

A script can't write this upfront — "delivered but not received" has multiple causes (stolen, wrong door, scan error). The agent reads each result and decides the next step dynamically. Loop repeats until `tool_calls` is empty.

#### LangChain code

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[get_order_status, get_delivery_proof, file_misdelivery_report, file_lost_package],
    system_prompt="You are a Slipkart support agent. Diagnose delivery issues step by step before taking action.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "ORD456 shows delivered but I never got it."}]})
```

#### When to use

- ✅ Path to answer is unknown upfront (need tool results to decide next step)
- ✅ Interactive / conversational tasks
- ✅ Customer support, coding agents, research assistants
- ✅ Default choice — start here

#### Pros / Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Handles branching naturally | Variable latency (unknown loops) |
| Cheap per step | Token cost grows with each loop |
| Simple to implement | Risk: infinite loop without `max_steps` |

#### Real-world examples

- **SWE-agent**: reads error → finds file → edits code → runs tests → repeats until green
- **E-commerce support bot**: ambiguous complaint → diagnose with tools → pick correct resolution (script can't enumerate all cases)
- **Research agent**: search topic → read results → decide if more searches needed → summarize

#### Key safeguard

```python
# always set recursion limit — one incident saw 4 agents run 11 days = $47,000 bill
agent = create_agent(..., recursion_limit=10)
```

---

### 2. Plan-and-Execute

**Planner writes the full task list upfront. Executor runs each step.**

#### How it works

```
Human: "Research AI trends, write a 5-section report, save as PDF"
              │
              ▼
     ┌─────────────────────────────────────────┐
     │   PLANNER LLM (strong model)            │
     │   1. Search "AI trends 2026"            │
     │   2. Search "LLM benchmarks 2026"       │
     │   3. Summarize search results           │
     │   4. Write report sections              │
     │   5. Format and save PDF                │
     └────────────────┬────────────────────────┘
                      │ plan (list of steps)
                      ▼
     ┌─────────────────────────────────────────┐
     │   EXECUTOR (cheap model, step by step)  │
     │   Step 1: search_tool("AI trends 2026") │
     │   Step 2: search_tool("LLM benchmarks") │
     │   Step 3: summarize(results)            │
     │   Step 4: write_sections(summary)       │
     │   Step 5: save_pdf(report)              │
     └─────────────────────────────────────────┘
                      │
                      ▼
              Final output (PDF)
```

#### LangChain code

```python
from langchain.agents import create_agent
import os

# strong model for planning
planner_llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
# cheap model for execution
executor_llm = init_chat_model("llama3.2", model_provider="ollama")

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query} ..."

@tool
def save_file(filename: str, content: str) -> str:
    """Save content to a file."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Saved {filename}"

# planner: generates the task list
planner = create_agent(
    model=planner_llm,
    tools=[],                   # planner only thinks, doesn't act
    system_prompt="""You are a task planner. Given a goal, output a numbered
    list of concrete steps for an executor to follow. Be specific.""",
)

# executor: runs each step
executor = create_agent(
    model=executor_llm,
    tools=[web_search, save_file],
    system_prompt="You are an executor. Run the exact step given to you.",
)

# orchestrate
goal = "Research AI trends 2026 and write a summary report"
plan_result = planner.invoke({"messages": [{"role": "user", "content": goal}]})
plan = plan_result["messages"][-1].content

# run each step
for step in plan.split("\n"):
    if step.strip():
        executor.invoke({"messages": [{"role": "user", "content": step}]})
```

#### When to use

- ✅ Multi-step workflows with predictable shape (ETL, research pipelines)
- ✅ Cost matters — use cheap model for execution (1 strong + N cheap calls)
- ✅ Task steps don't depend heavily on previous step outputs
- ❌ Avoid when later steps need unpredictable results from earlier steps

#### Pros / Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Bounded latency after planning | Brittle if step outputs surprise the plan |
| Cost efficient (strong + cheap) | Planner commits before seeing tool results |
| Easy to observe / debug | Needs re-planning gates for complex tasks |

#### Cost profile

```
ReAct:             N × strong_model_call
Plan-and-Execute:  1 × strong_model + N × cheap_model   ← wins when N > 3
```

---

### 3. Reflection

**Agent critiques its own output and improves it in a loop.**

#### How it works

```
Human: "Write a Python function to merge two sorted lists"
              │
              ▼
     ┌─────────────────┐
     │  GENERATOR LLM  │  → writes initial code
     └────────┬────────┘
              │ draft answer
              ▼
     ┌─────────────────┐
     │  CRITIC LLM     │  → "Missing edge case: empty list. O(n) but could be cleaner."
     └────────┬────────┘
              │ critique
              ▼
     ┌─────────────────┐
     │  GENERATOR LLM  │  → rewrites with edge cases fixed
     └────────┬────────┘
              │ improved answer
              ▼
     ┌─────────────────┐
     │  CRITIC LLM     │  → "Looks good. Approve."
     └────────┬────────┘
              │ approved
              ▼
         Final answer
```

#### LangChain code

```python
from langchain.agents import create_agent

generator = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a Python expert. Write clean, correct code.",
)

critic = create_agent(
    model=llm,          # ideally a different model to avoid self-bias
    tools=[],
    system_prompt="""You are a strict code reviewer. Find bugs, edge cases,
    and inefficiencies. If the code is good, respond with APPROVED.""",
)

def reflection_loop(task: str, max_rounds: int = 3) -> str:
    draft = generator.invoke({"messages": [{"role": "user", "content": task}]})
    answer = draft["messages"][-1].content

    for round_num in range(max_rounds):
        review = critic.invoke({
            "messages": [{"role": "user", "content": f"Review this code:\n{answer}"}]
        })
        critique = review["messages"][-1].content

        if "APPROVED" in critique:
            print(f"Approved after {round_num + 1} round(s)")
            break

        # regenerate with critique
        improved = generator.invoke({
            "messages": [{"role": "user", "content": f"Fix this code based on feedback:\n{answer}\n\nFeedback:\n{critique}"}]
        })
        answer = improved["messages"][-1].content

    return answer

result = reflection_loop("Write a function to merge two sorted lists")
```

#### When to use

- ✅ High-stakes output: code, legal docs, financial analysis
- ✅ Accuracy > speed
- ✅ Pair with external signals: run tests, lint, type-check as critic
- ❌ Avoid for simple conversational tasks (overkill)

#### Pros / Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Significantly improves output quality | Multiplies token cost by rounds |
| External test signals make it very powerful | Same model as critic → self-bias (approves own output) |
| AlphaCodium: GPT-4 jumped 19% → 44% on CodeContests | Slow for real-time tasks |

#### Key safeguard — avoid self-bias

```python
# BAD: same model critiques itself — will approve its own output
critic = create_agent(model=same_llm, ...)

# GOOD: different model or external signal as critic
critic = create_agent(model=different_llm, ...)

# BEST: run the actual code as critic
@tool
def run_tests(code: str) -> str:
    """Execute the code and return test results."""
    # run pytest, return pass/fail
```

---

## MULTI-AGENT PATTERNS

---

### 4. Subagents — Supervisor / Centralized

**One supervisor agent coordinates specialized subagents by calling them as tools.**

#### How it works

```
Human: "Book a flight to Delhi and add it to my calendar"
              │
              ▼
     ┌──────────────────────────────────┐
     │         SUPERVISOR AGENT         │
     │   sees: [flight_agent_tool,      │
     │           calendar_agent_tool]   │
     └──────┬───────────────┬───────────┘
            │               │  (parallel)
            ▼               ▼
   ┌──────────────┐  ┌──────────────────┐
   │ FLIGHT AGENT │  │  CALENDAR AGENT  │
   │ search_flight│  │  create_event()  │
   │ book_flight  │  │  check_conflicts │
   └──────┬───────┘  └────────┬─────────┘
          │ "Booked AI123"    │ "Event added"
          └────────┬──────────┘
                   │
                   ▼
          SUPERVISOR synthesizes
          "Flight AI123 booked and added to your calendar for June 20."
```

#### LangChain code

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

# subagent as a tool
def make_subagent_tool(agent, name, description):
    @tool(name=name, description=description)
    def subagent_tool(task: str) -> str:
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})
        return result["messages"][-1].content
    return subagent_tool

flight_agent = create_agent(model=llm, tools=[search_flights, book_flight],
                             system_prompt="You are a flight booking specialist.")

calendar_agent = create_agent(model=llm, tools=[create_event, check_calendar],
                               system_prompt="You are a calendar management specialist.")

flight_tool = make_subagent_tool(flight_agent, "flight_agent", "Books flights and checks availability")
calendar_tool = make_subagent_tool(calendar_agent, "calendar_agent", "Manages calendar events")

supervisor = create_agent(
    model=llm,
    tools=[flight_tool, calendar_tool],
    system_prompt="You are a personal assistant. Delegate tasks to specialists.",
)

result = supervisor.invoke({
    "messages": [{"role": "user", "content": "Book a flight to Delhi and add it to my calendar"}]
})
```

#### When to use

- ✅ Multiple distinct domains (booking + calendar + CRM + email)
- ✅ Subagents can run in parallel
- ✅ Different teams own different subagents
- ✅ Strong context isolation needed between domains

#### Pros / Cons

| ✅ Pros | ❌ Cons |
|---------|---------|
| Parallel execution | +1 model call overhead per subagent |
| Domain isolation | Context passed through supervisor (bottleneck) |
| Distributed development (separate teams) | Harder to debug end-to-end |

---

### 5. Router — Parallel Dispatch

**Classifies input, dispatches to specialists in parallel, synthesizes results.**

#### How it works

```
Human: "What's trending in AI and what's the INR/USD rate today?"
              │
              ▼
     ┌──────────────────────┐
     │       ROUTER         │  classifies: needs [news, finance]
     └──────────┬───────────┘
         ┌──────┴──────┐
         │ (parallel)  │
         ▼             ▼
  ┌────────────┐  ┌─────────────┐
  │ NEWS AGENT │  │FINANCE AGENT│
  │ web_search │  │ forex_api() │
  └────────────┘  └─────────────┘
         │             │
         └──────┬──────┘
                ▼
         ROUTER synthesizes
```

#### LangChain code

```python
from langchain_core.tools import tool

@tool
def route_to_news(query: str) -> str:
    """Search for news and current events."""
    result = news_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

@tool
def route_to_finance(query: str) -> str:
    """Answer finance, stock market, and currency questions."""
    result = finance_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

router = create_agent(
    model=llm,
    tools=[route_to_news, route_to_finance],
    system_prompt="You are a router. Dispatch queries to the right specialist and synthesize results.",
)
```

#### When to use

- ✅ Distinct verticals with no shared state (news vs finance vs weather)
- ✅ Parallel execution is the priority
- ✅ Stateless per-request design
- ❌ Avoid for multi-turn conversations (state doesn't carry over)

---

### 6. Handoffs — State-Driven Transitions

**Active agent changes dynamically. Agents hand off to the next via tool calls.**

#### How it works

```
Human: "I need support" → TRIAGE AGENT
              │
              │ (user: "My flight was cancelled")
              ▼
     ┌──────────────────┐
     │   TRIAGE AGENT   │  → "This is a booking issue"
     └────────┬─────────┘
              │ handoff_to(booking_agent)
              ▼
     ┌──────────────────┐
     │  BOOKING AGENT   │  → handles rebooking
     └────────┬─────────┘
              │ handoff_to(payment_agent) if refund needed
              ▼
     ┌──────────────────┐
     │  PAYMENT AGENT   │  → processes refund
     └──────────────────┘
```

#### LangChain code

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def handoff_to_booking() -> str:
    """Transfer conversation to booking specialist."""
    return "HANDOFF:booking"

@tool
def handoff_to_payment() -> str:
    """Transfer conversation to payment specialist."""
    return "HANDOFF:payment"

triage_agent = create_agent(
    model=llm,
    tools=[handoff_to_booking, handoff_to_payment],
    system_prompt="You are a triage agent. Understand the issue and hand off to the right specialist.",
)
```

#### When to use

- ✅ Multi-stage conversations: triage → specialist → resolution
- ✅ State must carry through (user context preserved across agents)
- ✅ Customer support flows
- ❌ Cannot parallelize (sequential by design)

---

### 7. Skills — Progressive Disclosure

**Single agent loads specialized knowledge and prompts on-demand.**

#### How it works

```
Human: "Help me with Python code"
              │
              ▼
     ┌──────────────────────────────────┐
     │          SINGLE AGENT            │
     │  loads: python_skill prompt      │  → responds as Python expert
     └──────────────────────────────────┘
              │
Human: "Now help me with a SQL query"
              │
              ▼
     ┌──────────────────────────────────┐
     │          SAME AGENT              │
     │  loads: sql_skill prompt         │  → responds as SQL expert
     └──────────────────────────────────┘
```

Not truly multi-agent — one agent with switchable personas/knowledge. Covered in Phase 2 `@dynamic_prompt`.

#### When to use

- ✅ One agent needs multiple specializations
- ✅ Simplicity over isolation
- ❌ Context bloat over long conversations
- ❌ Avoid when domains truly need isolation

---

## COMPARISON TABLE

| Architecture | Latency | Token Cost | Parallelism | State | Best for |
|-------------|---------|-----------|-------------|-------|----------|
| **ReAct** | Variable | Medium | ❌ | ✅ | General purpose, start here |
| **Plan-and-Execute** | Bounded | Low (strong+cheap) | Partial | ✅ | Structured pipelines, ETL |
| **Reflection** | Slow | High (N rounds) | ❌ | ❌ | High-stakes accuracy |
| **Subagents** | Fast | Medium+overhead | ✅ | Partial | Multi-domain, team separation |
| **Router** | Fast | Low | ✅ | ❌ | Stateless parallel dispatch |
| **Handoffs** | Medium | Medium | ❌ | ✅ | Sequential conversation flows |
| **Skills** | Fast | Low | ❌ | ✅ | Single agent, multiple personas |

---

## DECISION TREE

```
Start here ──► Is task interactive / conversational?
                   │
           YES ────┘                  NO
           │                          │
           ▼                          ▼
      Use ReAct              Does it have predictable steps?
                                  │
                          YES ────┘              NO
                          │                      │
                          ▼                      ▼
                  Plan-and-Execute          Still ReAct
                          │
                    Needs quality check?
                          │
                     YES ──┘
                     │
                     ▼
               Add Reflection wrapper

Multi-agent? Add only when:
  - Truly distinct domains           → Subagents or Router
  - Multi-stage sequential flow      → Handoffs
  - Multiple personas in one agent   → Skills
```

---

## COMPOSITION — How They Stack in Production

Patterns compose freely:

```
Plan-and-Execute              ← top level orchestration
    └─ each step = ReAct agent      ← handles tool loops
           └─ final output = Reflection pass  ← quality gate

Supervisor (Subagents)
    └─ each subagent = Router       ← classifies domain
           └─ routed agent = ReAct  ← executes with tools
```

Real production example (travel booking assistant):
- **Handoff**: triage → booking → payment
- **ReAct**: each agent loops on tools until done
- **HITL**: Reflection-like gate before any payment action

---

## Coverage by Phase

| Architecture | Phase |
|-------------|-------|
| ReAct | Phase 1–2 (tool calling loop, `create_agent`) |
| Skills | Phase 2 (`@dynamic_prompt`) |
| Handoffs | Phase 2 (HITL, `Command(resume=...)`) |
| Plan-and-Execute | LangGraph Phase 3 |
| Reflection | LangGraph Phase 3 |
| Subagents / Supervisor | LangGraph Phase 4 |
| Router | LangGraph Phase 4 |

---

*Sources: [LangChain Blog — Choosing the Right Multi-Agent Architecture](https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture) · [DEV — Three Agent Patterns 2026](https://dev.to/gabrielanhaia/react-plan-and-execute-or-reflection-the-three-agent-patterns-every-engineer-needs-in-2026-355p) · [LangChain Multi-Agent Docs](https://docs.langchain.com/oss/python/langchain/multi-agent)*
