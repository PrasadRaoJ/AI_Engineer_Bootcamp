# ClaimSure Insurance Claims Agent

An AI insurance claims assistant that handles customer self-service (filing, status, documents),
gates adjuster actions (approve, flag for fraud) behind human approval for large amounts,
blocks off-topic queries before the LLM runs, and strips sensitive identifiers (policy numbers,
bank accounts) from every customer-facing response.

---

## File structure

```
claimsure_insurance/
├── schemas.py    — Context (user_id, role) and ClaimReport (structured output schema)
├── tools.py      — 6 tools + module-level CLAIMS dict (used by HITL when= predicate)
├── middleware.py — @before_agent, @after_agent, @wrap_model_call, @dynamic_prompt, AuditLogger
├── main.py       — agent setup, seeded data, HITL predicate, 6 scenarios
└── README.md     — this file
```

---

## schemas.py

```python
class Context(BaseModel):
    user_id: str
    role: Literal["customer", "adjuster"]

class ClaimReport(BaseModel):
    claim_id: str
    policyholder_id: str
    incident_type: Literal["motor_accident", "theft", "health", "property", "travel"]
    incident_date: str
    amount: float
    description: str
    status: Literal["filed", "under_review", "approved", "rejected", "fraud_flagged"]
    documents_received: List[str]
```

`Context` is injected per-call via `context=Context(...)` on `agent.invoke()`.
`ClaimReport` is the canonical data shape stored per claim in `InMemoryStore`.

---

## tools.py

### Module-level state

```python
CLAIMS: dict = {}       # mirrors store — readable by HITL when= predicate
_claim_counter = [1]    # mutable list so main.py can reset after seeding
```

`CLAIMS` is the key design decision in this project. The `when=` predicate for HITL
receives only a `ToolCallRequest` — it cannot access `InMemoryStore`. By mirroring
every claim into this module-level dict, the predicate can look up `amount` from
`claim_id` without touching the store.

### Store access pattern

`runtime.store.get(namespace, key)` returns an `Item` object — NOT the raw value.
Always unwrap:

```python
item = runtime.store.get(("claims", claim_id), "report")
if not item:
    return "not found"
report = item.value   # ← the actual dict
```

Failing to call `.value` causes `TypeError: unsupported operand type(s) for +: 'Item' and 'list'`
when you try to combine the result with other Python types.

### Customer tools (4)

| Tool | What it does |
|------|-------------|
| `create_claim` | Validates policyholder exists, generates `CLM-XXX` ID, saves report to store AND `CLAIMS` dict, tracks claim IDs under policyholder namespace |
| `get_claim_status` | Reads claim report from store, looks up policyholder profile for policy number (intentionally included so `@after_agent` can strip it) |
| `submit_document` | Appends `doc_type` to `documents_received`, saves back to store |
| `list_my_claims` | Reads claim ID list from `("policyholders", user_id)`, fetches each report |

### Adjuster-only tools (2)

| Tool | What it does |
|------|-------------|
| `approve_claim` | Updates claim status, returns payout with bank account number (stripped by `@after_agent` in customer context), accepts `'approve'`/`'approved'` both |
| `flag_for_fraud_review` | Sets status to `fraud_flagged`, returns SIU reference |

The `approve_claim` tool normalises the `decision` arg:
```python
decision = {"approve": "approved", "reject": "rejected"}.get(decision.lower(), decision.lower())
```
This handles the common model mistake of passing `"approve"` instead of `"approved"`.

---

## middleware.py

### `@before_agent` — off-topic filter

```python
INSURANCE_KEYWORDS = ["claim", "policy", "accident", "damage", ...]

@before_agent(can_jump_to=["end"])
def off_topic_filter(state: AgentState, runtime: Runtime):
    latest = next((m for m in reversed(msgs) if m.type == "human"), None)
    text = latest.content.lower()
    if any(kw in text for kw in INSURANCE_KEYWORDS):
        return None
    return {"messages": [...blocked reply...], "jump_to": "end"}
```

Uses an allow-list (any keyword must be present) rather than a block-list. This inverts the
fixit_helpdesk pattern — instead of blocking known bad patterns, we block everything that
doesn't look like insurance. More robust for a narrow-domain agent.

### `@after_agent` — PII output stripper ← **new in this project**

```python
@after_agent(can_jump_to=["end"])
def pii_output_stripper(state: AgentState, runtime: Runtime):
    last_ai = next((m for m in reversed(msgs) if m.type == "ai" and m.content), None)
    clean = re.sub(r"\bPOL-\d+\b", "[POLICY]", last_ai.content)
    clean = re.sub(r"\b\d{9,18}\b", "[ACCOUNT]", clean)
    if clean == last_ai.content:
        return None
    return {"messages": [AIMessage(content=clean)]}
```

Fires after the agent finishes. Appends a clean copy of the last AI message.
`last_msg()` iterates in reverse, so it picks up the clean version — the original
dirty message stays in state but is never the final reply.

**Why `@after_agent` not `PIIMiddleware`:** `PIIMiddleware(apply_to_output=True)` works for
standard PII types (email, credit card). For custom patterns like `POL-\d+`, `@after_agent`
with `re.sub` is more explicit and auditable.

### `PIIMiddleware` — credit card redaction from input ← **new in this project**

```python
PIIMiddleware("credit_card")   # in middleware= list on create_agent
```

Intercepts customer messages before they reach the model. If a customer accidentally
types a card number in their claim description, it is redacted to `[CREDIT_CARD]`
before the LLM ever sees it.

### `@wrap_model_call` — hide adjuster tools from customers

```python
CUSTOMER_HIDDEN_TOOLS = {"approve_claim", "flag_for_fraud_review"}

@wrap_model_call
def role_based_tool_filter(request: ModelRequest[Context], handler):
    ctx = request.runtime.context
    if ctx and ctx.role == "customer":
        filtered = [t for t in request.tools if t.name not in CUSTOMER_HIDDEN_TOOLS]
        request = request.override(tools=filtered)
    return handler(request)
```

Same pattern as fixit_helpdesk. Guard `if ctx and ...` — context is `None` on
`Command(resume=...)` calls; fall through with all tools.

### `@dynamic_prompt` — inject user_id per call

Injects `"When calling any tool that takes policyholder_id, always pass '{user_id}'"`.
Without this, the model treats `policyholder_id` as a template placeholder and
passes the literal string `"<policyholder_id>"` to tools.

---

## main.py — HITL `when=` predicate

```python
def large_claim(request: ToolCallRequest) -> bool:
    claim_id = request.tool_call["args"].get("claim_id", "").upper()
    return float(CLAIMS.get(claim_id, {}).get("amount", 0)) > 10_000
```

`ToolCallRequest` only has `request.tool_call["args"]` — no store access.
`CLAIMS` (module-level dict in tools.py) is imported into main.py and used here.
Returns `True` → HITL interrupt. Returns `False` → auto-execute.

### `interrupt_on` dict format with `when=`

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "approve_claim": {
            "allowed_decisions": ["approve", "reject"],
            "when": large_claim,
        },
    }
)
```

When `when=` is present, the interrupt only fires if the predicate returns `True`.
Requires `langchain>=1.3.3`.

---

## Scenario walkthroughs

### Scenario 1 — Customer files a claim

```
POL001 (customer): "I had an accident near MG Road... repair estimate ₹7,200."
  → @before_agent: "accident", "repair" in text → pass
  → @wrap_model_call: role=customer → approve_claim, flag_for_fraud_review hidden
  → agent calls create_claim("POL001", "motor_accident", "2026-06-13", 7200.0, ...)
  → CLAIMS["CLM-003"] saved, store updated
  → reply: "Claim filed — ID: CLM-003"
```

### Scenario 2 — Small claim auto-approved (no HITL)

```
ADJ001 (adjuster): "Approve claim CLM-001 — straightforward parking damage."
  → agent calls approve_claim("CLM-001", "ADJ001", "approve", "...")
  → when=(CLM-001.amount=4500) → 4500 > 10000? False → NO interrupt
  → tool executes immediately
  → AuditLogger logs: [AUDIT →] approve_claim ...
  → HITL triggered: False ✓
```

### Scenario 3 — Large claim pauses for HITL

```
ADJ001 (adjuster): "Approve claim CLM-002 — flood damage genuine."
  → agent calls approve_claim("CLM-002", "ADJ001", "approve", "...")
  → when=(CLM-002.amount=85000) → 85000 > 10000? True → INTERRUPT
  → result.interrupts truthy → human reviews and approves
  → Command(resume={"decisions": [{"type": "approve"}]}) sent
  → tool executes: "Claim CLM-002 APPROVED — Payout ₹85,000 → account [ACCOUNT]"
  → No more interrupts ✓
```

The `while r3.interrupts:` loop in main.py handles the edge case where the model
retries `approve_claim` after a validation error — each retry re-triggers HITL.

### Scenario 4 — Customer status check

```
POL001 (customer): "What's the status of CLM-001?"
  → get_claim_status("CLM-001") → returns status APPROVED, no PII to strip
  → reply: "Your claim CLM-001 is APPROVED. Payout being processed."
```

### Scenario 5 — Off-topic blocked

```
"Can you help me write a complaint letter to my employer?"
  → @before_agent: none of INSURANCE_KEYWORDS found → blocked
  → "I can only assist with ClaimSure insurance claims." ← zero LLM calls

"What's the best restaurant near Bandra?" → blocked
"Book me a flight to Chennai next Friday." → blocked
```

### Scenario 6 — PII stripped from customer output

```
POL002 (customer): "Full update on claim CLM-002?"
  → get_claim_status returns: "Policy: POL-8821\n...account 112233445566..."
  → agent summarises and includes "Policy: POL-8821"
  → @after_agent fires:
      re.sub(r"\bPOL-\d+\b", "[POLICY]")  → "Policy: [POLICY]"
      re.sub(r"\b\d{9,18}\b", "[ACCOUNT]") → no match (amount has ₹ prefix)
  → appends clean AIMessage to state
  → last_msg() picks up clean version
  → POL-8821 absent in reply ✓   [POLICY] present ✓
```

---

## Phase 1 + 2 features used

| Feature | Where |
|---------|-------|
| `@tool` with typed args | All 6 tools in tools.py |
| `create_agent` | main.py — single agent, both roles via context |
| `context_schema=Context` + `ToolRuntime[Context]` | user_id injected per call; tools scope to right policyholder |
| `@before_agent(can_jump_to=["end"])` | Off-topic allow-list filter |
| `@after_agent(can_jump_to=["end"])` | PII output stripper — policy IDs + bank accounts |
| `@wrap_model_call` + `request.override(tools=...)` | Hides approve_claim / flag_for_fraud_review from customers |
| `@dynamic_prompt` | Injects concrete user_id into system prompt |
| `PIIMiddleware("credit_card")` | Redacts card numbers from customer input |
| `HumanInTheLoopMiddleware` with `when=` predicate | Interrupts approve_claim only when amount > ₹10,000 |
| `InMemoryStore` + `item.value` | All claim data stored per-claim; policyholder tracks claim list |
| `config={"callbacks": [AdjusterAuditLogger()]}` | Audit log for every approve/flag call |

---

## Key gotchas

**1. `runtime.store.get()` returns an Item, not the raw value**

Always unwrap: `item = store.get(...); value = item.value if item else default`.
Using the Item directly causes `TypeError` on any comparison or concatenation.

**2. HITL `when=` predicate can't access InMemoryStore**

The predicate only receives `ToolCallRequest`. To check a claim's amount from a `claim_id`
argument, you must mirror store data into a module-level dict (`CLAIMS` in tools.py)
that the predicate can import.

**3. `@after_agent` appends — it does not replace**

Returning `{"messages": [AIMessage(content=clean)]}` adds the clean message to state.
The original message (with raw PII) remains in history. `last_msg()` iterates in reverse,
so the clean version is returned. This is intentional: the store keeps the full audit trail.

**4. Model normalisation for enum-like tool args**

Models (especially smaller ones) pass `"approve"` when the tool expects `"approved"`.
Normalise in the tool with a dict map rather than trusting the model to be precise.

**5. `@dynamic_prompt` is mandatory when tools take an ID from context**

Without `"always pass '{user_id}' as policyholder_id"` in the system prompt, the model
passes the literal string `"<policyholder_id>"` to tools — causing "not found" errors.

---

## How to run

```bash
cd langchain_course/projects/phase02_agents/claimsure_insurance
python main.py
```

Requires Ollama running locally with `qwen3.5:2b`:
```bash
ollama pull qwen3.5:2b
ollama serve
```
