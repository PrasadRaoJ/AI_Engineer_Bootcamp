"""
ClaimSure Insurance Claims Agent
tools.py: 6 tools + module-level CLAIMS dict (also used by HITL when= predicate).

Store access pattern: runtime.store.get() returns an Item object.
Always unwrap with item.value — never use the Item directly.
"""
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from schemas import Context

# ── Module-level state ────────────────────────────────────────────────────────
# CLAIMS mirrors what's in the store so the HITL when= predicate can read it
# (predicates only receive ToolCallRequest — no store access).
CLAIMS: dict = {}
_claim_counter = [1]   # mutable list so main.py can reset it after import

# ── Internal helpers ──────────────────────────────────────────────────────────

def _next_step(status: str) -> str:
    return {
        "filed":        "awaiting initial review (1–2 business days)",
        "under_review": "adjuster assessment in progress",
        "approved":     "payout being processed to your account",
        "rejected":     "decision letter sent — contact support to appeal",
        "fraud_flagged":"referred to Special Investigations Unit",
    }.get(status, "contact support for details")


# ── Customer tools ────────────────────────────────────────────────────────────

@tool
def create_claim(
    policyholder_id: str,
    incident_type: str,
    incident_date: str,
    amount: float,
    description: str,
    runtime: ToolRuntime[Context],
) -> str:
    """File a new insurance claim. Returns a claim ID. Saves claim to long-term store."""
    item = runtime.store.get(("policyholders", policyholder_id), "profile")
    if not item:
        return f"Policyholder {policyholder_id} not found. Check the ID and try again."

    claim_id = f"CLM-{_claim_counter[0]:03d}"
    _claim_counter[0] += 1

    report = {
        "claim_id":           claim_id,
        "policyholder_id":    policyholder_id,
        "incident_type":      incident_type,
        "incident_date":      incident_date,
        "amount":             float(amount),
        "description":        description,
        "status":             "filed",
        "documents_received": [],
    }
    CLAIMS[claim_id] = report
    runtime.store.put(("claims", claim_id), "report", report)

    existing_item = runtime.store.get(("policyholders", policyholder_id), "claims")
    existing = existing_item.value if existing_item else []
    runtime.store.put(("policyholders", policyholder_id), "claims", existing + [claim_id])

    return (
        f"Claim filed — ID: {claim_id}\n"
        f"  Incident: {incident_type} on {incident_date}\n"
        f"  Estimated amount: ₹{float(amount):,.0f}\n"
        f"  Status: filed — you will hear back within 48 hours."
    )


@tool
def get_claim_status(claim_id: str, runtime: ToolRuntime[Context]) -> str:
    """Get the current status of a claim: stage, documents received, next step."""
    item = runtime.store.get(("claims", claim_id.upper()), "report")
    if not item:
        return f"Claim {claim_id} not found. Check the claim ID and try again."
    report = item.value

    profile_item = runtime.store.get(("policyholders", report["policyholder_id"]), "profile")
    profile = profile_item.value if profile_item else {}

    return (
        f"Claim {report['claim_id']} — {report['status'].upper()}\n"
        f"  Policy: {profile.get('policy_number', 'N/A')}\n"
        f"  Incident: {report['incident_type']} on {report['incident_date']}\n"
        f"  Claimed amount: ₹{report['amount']:,.0f}\n"
        f"  Documents on file: {', '.join(report['documents_received']) or 'none'}\n"
        f"  Next step: {_next_step(report['status'])}"
    )


@tool
def submit_document(
    claim_id: str,
    doc_type: str,
    content: str,
    runtime: ToolRuntime[Context],
) -> str:
    """Submit a supporting document for an existing claim (e.g. photo, police report)."""
    item = runtime.store.get(("claims", claim_id.upper()), "report")
    if not item:
        return f"Claim {claim_id} not found."
    report = item.value

    report["documents_received"].append(doc_type)
    CLAIMS[claim_id.upper()] = report
    runtime.store.put(("claims", claim_id.upper()), "report", report)

    return (
        f"Document received for {claim_id}.\n"
        f"  Type: {doc_type}\n"
        f"  Total documents on file: {len(report['documents_received'])}"
    )


@tool
def list_my_claims(runtime: ToolRuntime[Context]) -> str:
    """List all claims filed by the current policyholder (reads from long-term store)."""
    user_id = runtime.context.user_id
    item = runtime.store.get(("policyholders", user_id), "claims")
    claim_ids = item.value if item else []

    if not claim_ids:
        return f"No claims on file for {user_id}."

    lines = [f"Claims for {user_id}:"]
    for cid in claim_ids:
        r_item = runtime.store.get(("claims", cid), "report")
        if r_item:
            r = r_item.value
            lines.append(
                f"  {cid}: {r['incident_type']} | ₹{r['amount']:,.0f} | {r['status']}"
            )
    return "\n".join(lines)


# ── Adjuster-only tools (hidden from customers via @wrap_model_call) ──────────

@tool
def approve_claim(
    claim_id: str,
    adjuster_id: str,
    decision: str,
    notes: str,
    runtime: ToolRuntime[Context],
) -> str:
    """Approve or reject a claim. Adjuster role only. Large claims pause for HITL."""
    item = runtime.store.get(("claims", claim_id.upper()), "report")
    if not item:
        return f"Claim {claim_id} not found."
    report = item.value

    decision = {"approve": "approved", "reject": "rejected"}.get(decision.lower(), decision.lower())
    if decision not in ("approved", "rejected"):
        return f"Invalid decision '{decision}'. Use 'approved' or 'rejected'."

    report["status"] = decision
    CLAIMS[claim_id.upper()] = report
    runtime.store.put(("claims", claim_id.upper()), "report", report)

    profile_item = runtime.store.get(("policyholders", report["policyholder_id"]), "profile")
    profile = profile_item.value if profile_item else {}
    bank = profile.get("bank_account", "N/A")

    if decision == "approved":
        return (
            f"Claim {claim_id} APPROVED by {adjuster_id}.\n"
            f"  Payout: ₹{report['amount']:,.0f} → account {bank}\n"
            f"  Notes: {notes}\n"
            f"  Policyholder notified by SMS and email."
        )
    return (
        f"Claim {claim_id} REJECTED by {adjuster_id}.\n"
        f"  Reason: {notes}\n"
        f"  Decision letter dispatched within 24 hours."
    )


@tool
def flag_for_fraud_review(
    claim_id: str,
    reason: str,
    runtime: ToolRuntime[Context],
) -> str:
    """Escalate a suspicious claim for fraud investigation. Adjuster role only."""
    item = runtime.store.get(("claims", claim_id.upper()), "report")
    if not item:
        return f"Claim {claim_id} not found."
    report = item.value

    report["status"] = "fraud_flagged"
    CLAIMS[claim_id.upper()] = report
    runtime.store.put(("claims", claim_id.upper()), "report", report)

    return (
        f"Claim {claim_id} flagged for fraud review.\n"
        f"  Reason: {reason}\n"
        f"  Referred to Special Investigations Unit — Ref: SIU-{claim_id}"
    )
