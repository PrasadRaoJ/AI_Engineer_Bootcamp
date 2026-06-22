from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

"""
Phase 2 — Topic 6: Human-in-the-loop
Pause on risky tool calls — approve, edit, or reject before execution.
"""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from a database table matching a condition."""
    return f"Deleted records from '{table}' where {condition}."


@tool
def read_table(table: str) -> str:
    """Read all records from a database table (safe, read-only)."""
    return f"Records in '{table}': [order_001, order_002, order_003]"


# ── Example 1: approve — execute the call as-is ───────────────────────────────

print("=== Example 1: approve ===")

agent = create_agent(
    model=llm,
    tools=[delete_records, read_table],
    system_prompt="You are a database assistant. Be concise.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_records": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_table": False,       # safe read — never pause
            },
        )
    ],
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "hitl-001"}}

# invoke — agent will call delete_records and PAUSE
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all records from orders where status='cancelled'."}]},
    config=cfg,
)
print("Interrupted. Pending tool calls:", result["__interrupt__"])

# human reviews, decides to approve
result2 = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=cfg,
)
print("After approve:", result2["messages"][-1].content)

print()

# ── Example 2: reject — tool is NOT called, feedback goes to agent ─────────────

print("=== Example 2: reject ===")

agent2 = create_agent(
    model=llm,
    tools=[delete_records, read_table],
    system_prompt="You are a database assistant. Be concise.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_records": {"allowed_decisions": ["approve", "reject"]},
                "read_table": False,
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

cfg2 = {"configurable": {"thread_id": "hitl-002"}}

result = agent2.invoke(
    {"messages": [{"role": "user", "content": "Delete all records from orders where status='cancelled'."}]},
    config=cfg2,
)
print("Interrupted:", result["__interrupt__"])

# human rejects — agent gets rejection feedback and responds accordingly
result2 = agent2.invoke(
    Command(resume={
        "decisions": [{"type": "reject", "feedback": "Rejected. Do not retry — this table is read-only in production."}]
    }),
    config=cfg2,
)
print("After reject:", result2["messages"][-1].content)

print()

# ── Example 3: conditional interrupt — pause only for dangerous conditions ─────

print("=== Example 3: conditional interrupt (when=) ===")

def is_delete_all(request: ToolCallRequest) -> bool:
    """Interrupt only when condition contains no WHERE filter (mass delete risk)."""
    condition = request.tool_call["args"].get("condition", "")
    return condition.strip().lower() in ("", "1=1", "true")

agent3 = create_agent(
    model=llm,
    tools=[delete_records, read_table],
    system_prompt="You are a database assistant. Be concise.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_records": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Mass-delete detected — approval required.",
                    "when": is_delete_all,    # only interrupt for unfiltered deletes
                },
                "read_table": False,
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

# filtered delete → condition is specific → when() returns False → no interrupt
cfg3a = {"configurable": {"thread_id": "hitl-003a"}}
result = agent3.invoke(
    {"messages": [{"role": "user", "content": "Delete records from orders where status='cancelled'."}]},
    config=cfg3a,
)
# when() returned False → ran without interrupt → result is a normal dict
if "__interrupt__" in result:
    print("Interrupted (unexpected):", result["__interrupt__"])
else:
    print("No interrupt (filtered delete passed through):", result["messages"][-1].content[:100])
