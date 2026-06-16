# Project 01 — Slipkart Customer Support Agent

## What it does

A customer support agent for Slipkart. Takes a customer's complaint, classifies the issue, calls the right tool with the right arguments, and streams a professional reply.

---

## Primitives used

| Primitive | Where |
|-----------|-------|
| `init_chat_model` (Models) | powers all LLM calls |
| `SystemMessage`, `HumanMessage`, `ToolMessage` (Messages) | conversation history |
| `@tool`, `bind_tools` (Tools) | 4 support actions |
| `with_structured_output` + Pydantic (Structured Output) | ticket classification |
| `.stream()` (Streaming) | final agent reply |

---

## Flow

```
┌─────────────────────────┐
│     Customer message    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  STEP 1 — Classify      │  SupportTicket: order_id, issue, priority, action_needed
│  (structured output)    │  extracts structured metadata for logging/observability
│                         │  Note: action_needed is logged only; Step 2 decides tool independently
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  STEP 2 — Tool call     │  LLM picks the right tool + args from the message
│  (bind_tools)           │  we run it and add result to history as ToolMessage
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  STEP 3 — Stream reply  │  LLM reads full history + tool result
│  (.stream())            │  streams final reply token by token
└─────────────────────────┘
```

---

## File breakdown

### `schemas.py` — Pydantic classification schema

Used in Step 1 to extract structured intent from the customer's message.

```python
class SupportTicket(BaseModel):
    order_id: str                                              # e.g. "ORD123"
    issue: str                                                 # short description
    priority: Literal["low", "medium", "high"]                 # urgency level
    action_needed: Literal["status", "cancel", "refund", "track"]  # what to do
```

**Why classify first:** Lets us log structured data, route to the right team, or set SLAs — before the tool even runs.

**Why `Literal` for priority and action:** Constrains the model to a fixed set of values so downstream logic can branch on them safely.

---

### `tools.py` — Tool definitions + mock data

#### Mock data

```python
ORDERS = {
    "ORD123": {"status": "Out for delivery. Expected by 6 PM today.",
               "tracking": "Left Mumbai warehouse at 9 AM. Currently in transit to Delhi.",
               "amount": 1299},
    "ORD456": {"status": "Delivered on 12 Jun 2026.",
               "tracking": "Delivered to front door at 2:34 PM.",
               "amount": 3499},
    "ORD789": {"status": "Delayed. New expected date: 16 Jun 2026.",
               "tracking": "Stuck at Pune sorting center due to weather.",
               "amount": 899},
}
```

In production these would be database queries.

#### Tools

**`get_order_status(order_id)`**
- Input: order ID like `ORD123`
- Logic: looks up `ORDERS[order_id]["status"]`
- Output: `"Out for delivery. Expected by 6 PM today."`

**`cancel_order(order_id)`**
- Input: order ID
- Logic: looks up the order amount from `ORDERS` dict
- Output: `"Order ORD123 has been cancelled. Refund of ₹1299 will be credited in 3-5 business days."`

**`raise_refund(order_id, reason)`**
- Input: order ID + reason string (model extracts this from customer's message)
- Logic: looks up the order amount from `ORDERS` dict
- Output: `"Refund request of ₹3499 raised for order ORD456. Reason: Damaged item received. You will hear from us in 24-48 hours."`

**`track_delivery(order_id)`**
- Input: order ID
- Logic: looks up `ORDERS[order_id]["tracking"]`
- Output: `"Stuck at Pune sorting center due to weather."`

```python
ALL_TOOLS = [get_order_status, cancel_order, raise_refund, track_delivery]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}  # name → function lookup for step 2
```

---

### `main.py` — Agent logic

#### System message
```python
SYSTEM = SystemMessage(
    "You are a formal customer support agent for Slipkart. "
    "Be professional, concise, and empathetic."
)
```

#### `run(user_input)` function — 3 steps

**Step 1 — Classify**
```python
ticket = llm.with_structured_output(SupportTicket).invoke([SYSTEM, HumanMessage(user_input)])
# returns SupportTicket object: order_id, issue, priority, action_needed
```

**Step 2 — Tool call**
```python
response = llm.bind_tools(ALL_TOOLS).invoke(history)
if response.tool_calls:
    result = TOOL_MAP[tc["name"]].invoke(tc["args"])    # run the Python function
    history.append(response)                             # AIMessage with tool_call
    history.append(ToolMessage(result, tool_call_id=tc["id"]))  # tool result
```

**Step 3 — Stream**
```python
for chunk in llm_with_tools.stream(history):
    print(chunk.content, end="", flush=True)
```

---

## Orders (mock data)

| Order | Status | Amount |
|-------|--------|--------|
| ORD123 | Out for delivery | ₹1299 |
| ORD456 | Delivered | ₹3499 |
| ORD789 | Delayed | ₹899 |

---

## How to run

```bash
cd projects/phase01_primitives/project01_slipkart_support
python main.py
```

---

## Sample interactions

| Customer message | Classified as | Tool called | Result |
|-----------------|--------------|-------------|--------|
| "My order ORD789 is delayed. Where is it?" | action=status, priority=low | `get_order_status("ORD789")` | delay info + new date |
| "Please cancel my order ORD123." | action=cancel, priority=low | `cancel_order("ORD123")` | cancelled + ₹1299 refund |
| "I got a damaged item in ORD456. Refund please." | action=refund, priority=high | `raise_refund("ORD456", "Damaged item")` | ₹3499 refund ticket raised |
