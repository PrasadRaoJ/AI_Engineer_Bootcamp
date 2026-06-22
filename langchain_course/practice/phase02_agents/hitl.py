from dotenv import load_dotenv
load_dotenv()
import os

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent,AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool


llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from a database table matching a condition."""
    return f"Deleted records from '{table}' where {condition}."


@tool
def read_table(table: str) -> str:
    """Read all records from a database table (safe, read-only)."""
    return f"Records in '{table}': [order_001, order_002, order_003]"


# ── Example 1: approve — execute the call as-is ───────────────────────────────

agent = create_agent(
    model=llm,
    tools = [delete_records,read_table],
    system_prompt="You are a database assistant. Be concise.",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_records": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_table": False,
            },
        )
    ],
    checkpointer=InMemorySaver(),
)


cfg = {"configurable": {"thread_id": "hitl-001"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all records from orders where status='cancelled'."}]},
    config=cfg,
)

print()
print("Interrupted. Pending tool calls:", result["__interrupt__"])

result2 = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=cfg,
)
print("After approve:", result2["messages"][-1].content)

print()



# ── Example 2: reject — tool is NOT called, feedback goes to agent ─────────────

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

