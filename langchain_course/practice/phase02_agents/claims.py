from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

llm = init_chat_model("llama3.2", model_provider="ollama", temperature=0)


CLAIMS = {
    "CLM001": {"status": "under review",       "amount": 25000,  "owner": "U001", "rejected": False},
    "CLM002": {"status": "pending documents",  "amount": 45000,  "owner": "U002", "rejected": False},
    "CLM003": {"status": "rejected",           "amount": 120000, "owner": "U001", "rejected": True },
    "CLM004": {"status": "approved",           "amount": 18000,  "owner": "U003", "rejected": False},
}


# Roles:
#   customer  — filed the claim, can only see their own claims
#   staff     — reviews claims, can approve up to ₹50,000
#   manager   — can approve any amount, override rejections

class Context(BaseModel):
    user_id: str
    role: str  # "customer" | "staff" | "manager"


def print_flow(label, result):
    print(f"=== {label} ===")
    for i, msg in enumerate(result["messages"]):
        name = type(msg).__name__
        if name == "HumanMessage":
            print(f"[{i}] USER        : {msg.content}")
        elif name == "AIMessage":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[{i}] LLM DECIDES : call {tc['name']}({tc['args']})")
            else:
                print(f"[{i}] LLM REPLY   : {msg.content[:100]}")
        elif name == "ToolMessage":
            print(f"[{i}] TOOL RESULT : {msg.content}")
    print()


@tool
def check_claim_status(claim_id: str, runtime: ToolRuntime[Context]) -> str:
    """Check the current status of a ClaimSure insurance claim."""
    claim = CLAIMS.get(claim_id)
    if not claim:
        return f"No claim found with ID {claim_id}."
    if runtime.context.role == "customer" and claim["owner"] != runtime.context.user_id:
        return "Access denied. You can only view your own claims."
    return f"Claim {claim_id} | Status: {claim['status']} | Amount: ₹{claim['amount']}"


@tool
def request_documents(claim_id: str, doc_type: str, runtime: ToolRuntime[Context]) -> str:
    """Request additional documents from a customer for a claim. Staff and managers only."""
    if runtime.context.role == "customer":
        return "Only staff can request documents."
    claim = CLAIMS.get(claim_id)
    if not claim:
        return f"No claim found with ID {claim_id}."
    return f"Document request sent for {claim_id}: '{doc_type}' required from customer."


@tool
def approve_claim(claim_id: str, runtime: ToolRuntime[Context]) -> str:
    """Approve a ClaimSure insurance claim. Staff can approve up to ₹50,000. Manager can approve any amount."""
    if runtime.context.role == "customer":
        return "Customers cannot approve claims."
    claim = CLAIMS.get(claim_id)
    if not claim:
        return f"No claim found with ID {claim_id}."
    if runtime.context.role == "staff" and claim["amount"] > 50000:
        return f"Claim {claim_id} is ₹{claim['amount']} — exceeds ₹50,000 staff limit. Escalate to a manager."
    claim["status"] = "approved"
    return f"Claim {claim_id} approved by {runtime.context.user_id}. ₹{claim['amount']} will be paid in 3-5 days."


@tool
def override_rejection(claim_id: str, reason: str, runtime: ToolRuntime[Context]) -> str:
    """Override a rejected claim and approve it. Managers only."""
    if runtime.context.role != "manager":
        return "Only managers can override rejections."
    claim = CLAIMS.get(claim_id)
    if not claim:
        return f"No claim found with ID {claim_id}."
    if not claim["rejected"]:
        return f"Claim {claim_id} is not rejected — nothing to override."
    claim["status"] = "approved"
    claim["rejected"] = False
    return f"Rejection overridden for {claim_id} by manager {runtime.context.user_id}. Reason: {reason}. Claim approved."


agent = create_agent(
    model=llm,
    tools=[check_claim_status, request_documents, approve_claim, override_rejection],
    system_prompt="You are a ClaimSure insurance support agent. Help customers and staff manage insurance claims clearly and concisely. Always reply in plain text. Never output JSON or tool calls in your final response.",
    context_schema=Context,
    
)


# Test 1 — customer checks someone else's claim (should be denied)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the status of claim CLM003?"}]},
    context=Context(user_id="U002", role="customer"),
)
print_flow("customer checks another person's claim", result)


# Test 2 — staff tries to approve a ₹1,20,000 claim (exceeds ₹50k limit)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Approve claim CLM003."}]},
    context=Context(user_id="S001", role="staff"),
)
print_flow("staff approves large claim", result)


# Test 3 — manager overrides rejected claim
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Override the rejection on CLM003. Customer submitted all required documents."}]},
    context=Context(user_id="M001", role="manager"),
    debug=True
)
print_flow("manager overrides rejection", result)
