"""
ClaimSure Insurance Claims Agent
main.py: One agent, customer + adjuster contexts, 6 scenarios.

New vs fixit_helpdesk:
  @after_agent    — PII output stripper (policy IDs, bank account numbers)
  PIIMiddleware   — credit card redaction from customer input
  HITL when=      — conditional interrupt: only fires when claim amount > ₹10,000
"""
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    ToolCallRequest,
)
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from schemas import Context
from tools import (
    create_claim, get_claim_status, submit_document,
    list_my_claims, approve_claim, flag_for_fraud_review,
    CLAIMS, _claim_counter,
)
from middleware import (
    off_topic_filter, claims_prompt, pii_output_stripper,
    role_based_tool_filter, AdjusterAuditLogger,
)

llm = ChatOllama(model="qwen3.5:2b", temperature=0)
store        = InMemoryStore()
saver        = InMemorySaver()

ALL_TOOLS = [
    create_claim, get_claim_status, submit_document,
    list_my_claims, approve_claim, flag_for_fraud_review,
]

# ── Seed policyholders (stored with full sensitive data) ──────────────────────

store.put(("policyholders", "POL001"), "profile", {
    "name":         "Arjun Mehta",
    "policy_number":"POL-5521",
    "bank_account": "998877665544",
    "vehicle":      "Honda City MH-12-AB-1234",
})
store.put(("policyholders", "POL002"), "profile", {
    "name":         "Sunita Rao",
    "policy_number":"POL-8821",
    "bank_account": "112233445566",
    "vehicle":      "Maruti Swift KA-01-CD-5678",
})

# ── Seed pre-existing claims ──────────────────────────────────────────────────

CLAIMS["CLM-001"] = {
    "claim_id":           "CLM-001",
    "policyholder_id":    "POL001",
    "incident_type":      "motor_accident",
    "incident_date":      "2026-06-10",
    "amount":             4500.0,
    "description":        "Minor dent on front bumper after parking lot collision",
    "status":             "under_review",
    "documents_received": ["incident_photo", "repair_estimate"],
}
store.put(("claims", "CLM-001"), "report", CLAIMS["CLM-001"])
store.put(("policyholders", "POL001"), "claims", ["CLM-001"])

CLAIMS["CLM-002"] = {
    "claim_id":           "CLM-002",
    "policyholder_id":    "POL002",
    "incident_type":      "motor_accident",
    "incident_date":      "2026-06-05",
    "amount":             85000.0,
    "description":        "Total loss — engine and body damage after flash flooding",
    "status":             "under_review",
    "documents_received": ["police_report", "flood_photos", "mechanic_assessment"],
}
store.put(("claims", "CLM-002"), "report", CLAIMS["CLM-002"])
store.put(("policyholders", "POL002"), "claims", ["CLM-002"])

_claim_counter[0] = 3  # next new claim will be CLM-003

# ── HITL when= predicate ──────────────────────────────────────────────────────
# Fires ONLY when approve_claim is called on a claim with amount > ₹10,000.
# Reads CLAIMS dict (module-level mirror of store) — predicate can't access store.

def large_claim(request: ToolCallRequest) -> bool:
    claim_id = request.tool_call["args"].get("claim_id", "").upper()
    return float(CLAIMS.get(claim_id, {}).get("amount", 0)) > 10_000

# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    context_schema=Context,
    middleware=[
        off_topic_filter,            # @before_agent  — blocks non-insurance queries
        claims_prompt,               # @dynamic_prompt — injects user_id per call
        pii_output_stripper,         # @after_agent   — strips POL-XXXX + bank accounts
        role_based_tool_filter,      # @wrap_model_call — hides approve/flag from customers
        PIIMiddleware("credit_card"),  # redact card numbers from customer input
        HumanInTheLoopMiddleware(
            interrupt_on={
                "approve_claim": {
                    "allowed_decisions": ["approve", "reject"],
                    "when": large_claim,
                },
            }
        ),
    ],
    checkpointer=saver,
    store=store,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def last_msg(result) -> str:
    msgs = result.value["messages"] if hasattr(result, "value") else result["messages"]
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 1 — Customer files a new claim
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Scenario 1 — Customer files a new claim")
print("=" * 60)

cfg1 = {"configurable": {"thread_id": "cs-s1"}}
ctx1 = Context(user_id="POL001", role="customer")

r1 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "I had an accident near MG Road on June 13th. My Honda City was hit from behind "
        "at a traffic light. Rear bumper and boot are damaged. "
        "Repair estimate is ₹7,200."}]},
    config=cfg1, context=ctx1, version="v2",
)
print(f"Agent: {last_msg(r1)[:300]}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 2 — Small claim auto-approved (when= returns False → no HITL)
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 2 — Small claim (₹4,500) auto-approved — no HITL")
print("=" * 60)

cfg2 = {
    "configurable": {"thread_id": "cs-s2"},
    "callbacks": [AdjusterAuditLogger()],
}
ctx2 = Context(user_id="ADJ001", role="adjuster")

r2 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Approve claim CLM-001 — documents look good, straightforward parking damage."}]},
    config=cfg2, context=ctx2, version="v2",
)
print(f"HITL triggered: {bool(r2.interrupts)}  (expected: False)")
print(f"Agent: {last_msg(r2)[:300]}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 3 — Large claim pauses for HITL (when= returns True)
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 3 — Large claim (₹85,000) pauses for adjuster HITL")
print("=" * 60)

cfg3 = {
    "configurable": {"thread_id": "cs-s3"},
    "callbacks": [AdjusterAuditLogger()],
}
ctx3 = Context(user_id="ADJ001", role="adjuster")

r3 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Approve claim CLM-002 — all three documents verified, flood damage is genuine."}]},
    config=cfg3, context=ctx3, version="v2",
)

while r3.interrupts:
    req = r3.interrupts[0].value["action_requests"][0]
    print(f"HITL interrupt → {req['name']} | args: {req['args']}")
    r3 = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=cfg3, version="v2",
    )
print(f"Agent: {last_msg(r3)[:300]}")
print(f"No more interrupts: {not r3.interrupts}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 4 — Customer checks claim status
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 4 — Customer checks claim status")
print("=" * 60)

cfg4 = {"configurable": {"thread_id": "cs-s4"}}
ctx4 = Context(user_id="POL001", role="customer")

r4 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "What's the status of my claim CLM-001? Did you receive my documents?"}]},
    config=cfg4, context=ctx4, version="v2",
)
print(f"Agent: {last_msg(r4)[:400]}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 5 — Off-topic blocked by @before_agent
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 5 — Off-topic queries blocked")
print("=" * 60)

off_topic_queries = [
    "Can you help me write a complaint letter to my employer?",
    "What's the best restaurant near Bandra?",
    "Book me a flight to Chennai next Friday.",
]

ctx5 = Context(user_id="POL001", role="customer")
for i, query in enumerate(off_topic_queries, 1):
    cfg5 = {"configurable": {"thread_id": f"cs-s5-{i}"}}
    r5 = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=cfg5, context=ctx5, version="v2",
    )
    reply = last_msg(r5)
    blocked = "only" in reply.lower() or "insurance" in reply.lower()
    print(f"  Query {i}: {'[BLOCKED]' if blocked else '[PASSED]'} {query[:55]}...")
    if blocked:
        print(f"    Reply: {reply[:100]}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 6 — Policy ID and bank account stripped from customer output
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 6 — @after_agent strips PII from customer response")
print("=" * 60)

cfg6 = {"configurable": {"thread_id": "cs-s6"}}
ctx6 = Context(user_id="POL002", role="customer")

r6 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Can you give me a full update on my claim CLM-002 including all details?"}]},
    config=cfg6, context=ctx6, version="v2",
)
reply6 = last_msg(r6)
print(f"Agent reply:\n{reply6[:500]}")
print()
policy_id_leaked  = "POL-8821" in reply6
bank_acc_leaked   = "112233445566" in reply6
print(f"Policy number hidden (POL-8821 absent): {not policy_id_leaked}")
print(f"Bank account hidden (112233445566 absent): {not bank_acc_leaked}")
print(f"[POLICY] placeholder present: {'[POLICY]' in reply6}")
