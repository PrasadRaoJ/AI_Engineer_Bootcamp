# ClaimSure Insurance Claims Agent

## Problem Statement

ClaimSure handles motor and health insurance claims for 50,000 policyholders. Today, claims go through a 3-day manual process: customer calls in, fills a form, an agent reviews it, a manager approves. ClaimSure wants to automate Tier-1 (document submission, status checks, small claims under ₹10,000) and gate Tier-2 (large payouts, fraud-flagged claims) behind a human adjuster.

The risk: customers must never see the `approve_claim` tool — that's internal only. Responses must never leak bank account numbers or policy IDs back to the customer raw. And the agent must stay strictly on insurance topics — it is not a general assistant.

---

## What the agent must handle

1. Customer files a new claim with details (incident type, date, amount)
2. Agent creates a structured `ClaimReport` and saves it to the store
3. Customer asks for status on an existing claim
4. Customer submits a supporting document (simulated as text description)
5. Adjuster (internal role) reviews and approves/rejects — `approve_claim` is hidden from customer view entirely
6. Claims above ₹10,000 pause for human adjuster approval even in adjuster view
7. Off-topic queries ("can you help me book a flight?") are blocked at entry
8. Bank account numbers and policy IDs in any response are stripped before the customer sees them

---

## Scenarios to implement

```
Scenario 1 — File a new claim (customer)
  "I had a minor accident on June 12th. My car has a dent. Repair estimate is ₹4,500."
  → create_claim() → ClaimReport generated → saved to store → customer gets claim ID

Scenario 2 — Small claim auto-approved (no HITL)
  Amount is ₹4,500 → below ₹10,000 threshold → approve_claim runs without pause

Scenario 3 — Large claim requires adjuster HITL
  Amount is ₹85,000 → above ₹10,000 → approve_claim PAUSES → adjuster reviews → approves

Scenario 4 — Customer asks for status
  "What is the status of claim CLM-002?"
  → get_claim_status(claim_id) → returns current stage + documents received

Scenario 5 — Off-topic blocked
  "Can you help me write a complaint letter to my employer?"
  → @before_agent catches it → "I can only help with ClaimSure insurance claims."

Scenario 6 — Policy ID stripped from output
  Agent would reply: "Your claim on policy POL-8821 has been approved."
  → @after_agent strips POL-8821 → "Your claim on [POLICY] has been approved."
```

---

## Tools to build

| Tool | Args | Returns |
|------|------|---------|
| `create_claim` | `policyholder_id, incident_type, incident_date, amount, description` | claim ID + ClaimReport |
| `get_claim_status` | `claim_id` | current stage, documents received, estimated resolution date |
| `submit_document` | `claim_id, doc_type, content` | confirmation that document was received |
| `list_my_claims` | *(reads from store via ToolRuntime)* | all claims for this policyholder |
| `approve_claim` | `claim_id, adjuster_id, decision, notes` | approval/rejection confirmation + payout amount |
| `flag_for_fraud_review` | `claim_id, reason` | escalation confirmation |

`approve_claim` and `flag_for_fraud_review` must **never appear in the customer-facing tool list** — only adjuster context gets them.

---

## Schemas to define

```python
class Context(BaseModel):
    user_id: str
    role: Literal["customer", "adjuster"]

class ClaimReport(BaseModel):
    claim_id: str
    policyholder_id: str
    incident_type: Literal["motor_accident", "theft", "health", "property", "travel"]
    incident_date: str          # YYYY-MM-DD
    amount: float
    description: str
    status: Literal["filed", "under_review", "approved", "rejected", "fraud_flagged"]
    documents_received: list[str]
```

---

## Phase 1 + 2 features and where they belong

| Feature | Where to use it |
|---------|----------------|
| `@tool` with `args_schema=` | All 6 tools — `approve_claim` especially needs typed, validated inputs |
| `with_structured_output(ClaimReport)` | Generate a `ClaimReport` on every `create_claim` call |
| Streaming | Stream the adjuster's review notes as they type the decision |
| `create_agent` | Main agent loop for both customer and adjuster flows |
| `context_schema=` + `ToolRuntime[Context]` | Pass `user_id` and `role`; tools scope to the right policyholder via `runtime.context.user_id` |
| `@wrap_model_call` | Hide `approve_claim` and `flag_for_fraud_review` from customer role entirely |
| `@before_agent` | Block off-topic queries — insurance-only agent |
| `@after_agent` | Strip raw policy IDs and bank account numbers from the final response |
| `PIIMiddleware("credit_card")` | Redact any card numbers from customer input |
| `InMemoryStore` + `runtime.store` | Save ClaimReport under `("claims", claim_id)` — tools read it back via `runtime.store.get` |
| `HumanInTheLoopMiddleware` | Interrupt `approve_claim` when `amount > 10_000` using `when=` predicate |
| `config=` with callbacks | Adjuster audit log — every `approve_claim` call logged with adjuster_id and timestamp |

---

## HITL `when=` predicate hint

```python
def requires_adjuster_approval(request: ToolCallRequest) -> bool:
    amount = request.tool_call["args"].get("amount", 0)  # you'll need to look this up
    return amount > 10_000
```

The catch: `approve_claim` doesn't directly receive the amount — you need to look it up from the store using the `claim_id`. Think about how to do this cleanly inside the predicate.

---

## `@before_agent` — off-topic detection

The agent should only respond to insurance topics. Block anything unrelated:

```python
INSURANCE_KEYWORDS = ["claim", "policy", "premium", "accident", "damage",
                      "document", "payout", "coverage", "health", "motor", "theft"]

# if none of the keywords appear in the message → block it
```

---

## Long-term store structure

```
namespace: ("claims", claim_id)
  key: "report"  → ClaimReport dict

namespace: ("policyholders", user_id)
  key: "claims"  → list of claim_ids for this user
  key: "profile" → {"name": ..., "policy_number": ..., "bank_account": ...}
```

Profile is stored in full internally but must be stripped from external responses by `@after_agent`.

---

## Suggested file structure

```
projects/phase02_agents/projectC_insurance_claims/
├── main.py          # two agents (customer + adjuster), 6 scenarios
├── tools.py         # all 6 tools + mock claim/policy data
├── schemas.py       # Context, ClaimReport
├── middleware.py    # @before_agent off-topic filter, @after_agent PII stripper, @wrap_model_call role filter
└── README.md        # your own explanation after building it
```

---

## Hints

- Build **two separate agents** in `main.py`: one for customer (restricted tools), one for adjuster (full tools including approve_claim). Both share the same `store` so the adjuster sees what the customer filed.
- `@after_agent` output stripping: use `re.sub(r"POL-\d+", "[POLICY]", last.content)` and similar patterns for bank accounts.
- For the `when=` predicate on HITL, the cleanest approach is to look up the claim from the store inside the predicate using the `claim_id` arg, then check its amount.
- `list_my_claims` should use `runtime.store.search(("policyholders", runtime.context.user_id))` — return a formatted list of all claim IDs and statuses for this user.
- Scenario 2 vs 3: the only difference is the amount. Make sure your mock data has one claim under and one over ₹10,000 to clearly demonstrate the HITL threshold.
- `ClaimReport` structured output is generated by the agent (not the tool) — after `create_claim` returns the raw dict, pass it through `llm.with_structured_output(ClaimReport)` to produce the typed object.
