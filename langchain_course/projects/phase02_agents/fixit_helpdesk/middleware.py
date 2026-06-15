"""
FixIt IT Helpdesk Agent
middleware.py: @before_agent content filter, @wrap_model_call role-based tool filter,
               AuditLogger callback for audit trail.
"""
from langchain.agents.middleware import (
    before_agent,
    AgentState,
    wrap_model_call,
)
from langchain.agents.middleware.types import ModelRequest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.runtime import Runtime
from schemas import Context


# ── @before_agent — block social engineering / prompt injection ────────────────

INJECTION_PATTERNS = [
    "ignore your instructions",
    "ignore previous instructions",
    "pretend you are",
    "act as",
    "you are now",
    "bypass",
    "override",
    "jailbreak",
    "as an it admin",
    "forget your instructions",
    "disregard your",
    "new instructions",
]

@before_agent(can_jump_to=["end"])
def social_engineering_filter(state: AgentState, runtime: Runtime):
    msgs = state.get("messages", [])
    if not msgs:
        return None
    # Check only the most recent human message — prior turns already passed the filter
    latest = next((m for m in reversed(msgs) if m.type == "human"), None)
    if latest is None:
        return None
    text = latest.content.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in text:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": (
                        "Request blocked by security policy. "
                        "Prompt manipulation is not permitted. "
                        "Please submit a genuine IT support request."
                    ),
                }],
                "jump_to": "end",
            }
    return None


# ── @wrap_model_call — filter tool list by role ───────────────────────────────

EMPLOYEE_TOOL_NAMES = {
    "reset_password",
    "check_vpn_config",
    "install_software",
    "check_ticket_status",
    "create_ticket",
}

@wrap_model_call
def role_based_tool_filter(request: ModelRequest[Context], handler):
    ctx = request.runtime.context
    # On Command(resume=...) context is None — keep all tools (admin already approved)
    if ctx and ctx.role == "employee":
        filtered = [t for t in request.tools if t.name in EMPLOYEE_TOOL_NAMES]
        request = request.override(tools=filtered)
    return handler(request)


# ── AuditLogger — callback for IT audit trail ─────────────────────────────────

class AuditLogger(BaseCallbackHandler):
    """Logs every tool call name, args, and result for the IT audit trail."""

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown_tool")
        print(f"  [AUDIT →] {name}  args: {str(input_str)[:120]}")

    def on_tool_end(self, output, **kwargs):
        print(f"  [AUDIT ←] result: {str(output)[:120]}")
