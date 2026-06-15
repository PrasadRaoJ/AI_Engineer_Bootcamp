"""
ClaimSure Insurance Claims Agent
middleware.py:
  @before_agent  — off-topic filter (insurance-only agent)
  @after_agent   — PII output stripper (policy IDs + bank accounts)
  @wrap_model_call — hide approve_claim and flag_for_fraud_review from customers
  @dynamic_prompt — inject user_id + role into system prompt per call
  AdjusterAuditLogger — log approve/flag calls for adjuster sessions
"""
import re
from langchain.agents.middleware import (
    before_agent,
    after_agent,
    wrap_model_call,
    dynamic_prompt,
    AgentState,
)
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.runtime import Runtime
from schemas import Context


# ── @before_agent — block non-insurance queries ───────────────────────────────
# Keyword allow-list: at least one must appear in the user's message.

INSURANCE_KEYWORDS = [
    "claim", "claims", "policy", "policies", "premium", "accident", "damage",
    "document", "payout", "coverage", "health", "motor", "theft", "vehicle",
    "incident", "repair", "insurance", "adjuster", "approve", "reject",
    "status", "file", "submit", "flood", "fire", "injury", "hospital",
]

@before_agent(can_jump_to=["end"])
def off_topic_filter(state: AgentState, runtime: Runtime):
    msgs = state.get("messages", [])
    latest = next((m for m in reversed(msgs) if m.type == "human"), None)
    if latest is None:
        return None
    text = latest.content.lower()
    if any(kw in text for kw in INSURANCE_KEYWORDS):
        return None
    return {
        "messages": [{
            "role": "assistant",
            "content": (
                "I can only assist with ClaimSure insurance claims and policies. "
                "Please describe your insurance-related request."
            ),
        }],
        "jump_to": "end",
    }


# ── @after_agent — strip PII from the final AI response ──────────────────────
# Patterns stripped:
#   POL-<digits>  → [POLICY]    (policy numbers like POL-5521)
#   9–18 digits   → [ACCOUNT]   (bank account numbers)

@after_agent(can_jump_to=["end"])
def pii_output_stripper(state: AgentState, runtime: Runtime):
    msgs = state.get("messages", [])
    last_ai = next((m for m in reversed(msgs) if m.type == "ai" and m.content), None)
    if last_ai is None:
        return None

    clean = re.sub(r"\bPOL-\d+\b", "[POLICY]", last_ai.content)
    clean = re.sub(r"\b\d{9,18}\b", "[ACCOUNT]", clean)

    if clean == last_ai.content:
        return None
    return {"messages": [AIMessage(content=clean)]}


# ── @wrap_model_call — hide adjuster-only tools from customers ────────────────
# Fires before each LLM call; filters tool list based on context.role.
# Guard: context is None on Command(resume=...) — fall through with all tools.

CUSTOMER_HIDDEN_TOOLS = {"approve_claim", "flag_for_fraud_review"}

@wrap_model_call
def role_based_tool_filter(request: ModelRequest[Context], handler):
    ctx = request.runtime.context
    if ctx and ctx.role == "customer":
        filtered = [t for t in request.tools if t.name not in CUSTOMER_HIDDEN_TOOLS]
        request = request.override(tools=filtered)
    return handler(request)


# ── @dynamic_prompt — inject user_id + role into system prompt ───────────────

@dynamic_prompt
def claims_prompt(request: ModelRequest[Context]) -> str:
    ctx     = request.runtime.context
    user_id = ctx.user_id if ctx else "unknown"
    role    = ctx.role    if ctx else "customer"

    base = (
        f"You are the ClaimSure insurance assistant. Today is 2026-06-15. "
        f"Session user ID: {user_id}. "
        f"When calling any tool that takes policyholder_id, always pass '{user_id}'. "
        f"Be concise and professional."
    )
    if role == "adjuster":
        return base + (
            " Adjuster mode: you can use approve_claim and flag_for_fraud_review. "
            "Claims above ₹10,000 will pause for human adjuster approval."
        )
    return base + (
        " Customer mode: help with filing claims, checking status, and submitting documents. "
        "Do not mention or use approve_claim or flag_for_fraud_review."
    )


# ── AdjusterAuditLogger — log approve/flag tool calls ────────────────────────

class AdjusterAuditLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?")
        if name in ("approve_claim", "flag_for_fraud_review"):
            print(f"  [AUDIT →] {name}  args: {str(input_str)[:140]}")

    def on_tool_end(self, output, **kwargs):
        print(f"  [AUDIT ←] result: {str(output)[:140]}")
