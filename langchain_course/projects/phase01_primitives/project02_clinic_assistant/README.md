# Project 02 — Yapollo Clinic Appointment Assistant

## What it does

A clinic receptionist agent for Yapollo Clinic, Nellore. Takes a patient's natural language message, classifies the intent, calls the right tool with the right arguments, and streams a warm professional reply.

---

## Primitives used

| Primitive | Where |
|-----------|-------|
| `init_chat_model` (Models) | powers all LLM calls |
| `SystemMessage`, `HumanMessage`, `ToolMessage` (Messages) | conversation history |
| `@tool`, `bind_tools` (Tools) | 4 clinic actions |
| `with_structured_output` + Pydantic (Structured Output) | request classification |
| `.stream()` (Streaming) | final receptionist reply |

---

## Flow

```
┌─────────────────────────┐
│     Patient message     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  STEP 1 — Classify      │  AppointmentRequest: patient_name, reason, urgency, action_needed
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

Used in Step 1 to extract structured intent from the patient's message.

```python
class AppointmentRequest(BaseModel):
    patient_name: Optional[str]   # None if not mentioned (e.g. "tell me about Dr. Anusha")
    reason: str                   # why they're contacting
    urgency: Literal["routine", "urgent", "emergency"]
    action_needed: Literal["check", "book", "cancel", "info"]
```

**Why `patient_name` is Optional:** Some queries (e.g. "tell me about Dr. Anusha") don't involve a patient — forcing a name would cause the model to hallucinate one.

**Why `action_needed`:** Tells us at a glance what the patient wants before Step 2 runs. Useful for logging, routing, priority queues in production.

---

### `tools.py` — Tool definitions + mock data

#### Mock data

```python
SLOTS = {
    "2026-06-15": {
        "Dr. Prasad (General Physician)": ["10:00 AM", "11:30 AM", "3:00 PM"],
        "Dr. Anusha (Cardiologist)":      ["9:00 AM", "2:00 PM"],
    },
    ...
}

DOCTORS = {
    "Dr. Prasad": {"specialization": "General Physician", "experience": "15 years", "fee": "₹500"},
    "Dr. Anusha": {"specialization": "Cardiologist",      "experience": "20 years", "fee": "₹1200"},
}

APPOINTMENTS = {
    "APT001": {"patient": "Ravi Kumar", "doctor": "Dr. Prasad", "date": "2026-06-15", "time": "10:00 AM"},
}
```

In production these would be database queries.

#### Tools

**`check_availability(date, specialization)`**
- Input: `date` as `YYYY-MM-DD`, `specialization` as free text (e.g. "General Physician")
- Logic: searches `SLOTS[date]` for any doctor whose name contains the specialization
- Output: `"Dr. Prasad (General Physician): 10:00 AM, 11:30 AM, 3:00 PM"`

**`book_appointment(patient_name, doctor, date, time)`**
- Input: all four fields from the patient's message
- Logic: generates a new APT ID, adds to `APPOINTMENTS` dict
- Output: `"Appointment confirmed! ID: APT002 | Dr. Prasad on 2026-06-15 at 10:00 AM. Please arrive 10 mins early."`

**`cancel_appointment(appointment_id)`**
- Input: appointment ID like `APT001`
- Logic: pops entry from `APPOINTMENTS`
- Output: `"Appointment APT001 for Ravi Kumar with Dr. Prasad on 2026-06-15 has been cancelled."`

**`get_doctor_info(doctor_name)`**
- Input: doctor name (partial match supported — "Anusha" matches "Dr. Anusha")
- Logic: loops `DOCTORS` dict with `.lower()` partial match
- Output: `"Dr. Anusha — Cardiologist | 20 years experience | Consultation fee: ₹1200"`

---

### `main.py` — Agent logic

#### System message
```python
SYSTEM = SystemMessage(
    "You are a formal clinic receptionist at Yapollo Clinic, Nellore. "
    "Be professional, warm, and helpful. Always confirm appointment details clearly. "
    "Today's date is 2026-06-14. Always use year 2026 for any dates mentioned."
)
```
- Why today's date: without it the LLM sends dates in 2023 (its training bias)

#### `run(user_input)` function — 3 steps

**Step 1 — Classify**
```python
ticket = llm.with_structured_output(AppointmentRequest).invoke([SYSTEM, HumanMessage(user_input)])
# returns AppointmentRequest object with typed fields
```

**Step 2 — Tool call**
```python
response = llm.bind_tools(ALL_TOOLS).invoke(history)
if response.tool_calls:
    result = TOOL_MAP[tc["name"]].invoke(tc["args"])   # run the Python function
    history.append(response)                            # AIMessage with tool_call
    history.append(ToolMessage(result, tool_call_id=tc["id"]))  # tool result
```

**Step 3 — Stream**
```python
for chunk in llm_with_tools.stream(history):
    print(chunk.content, end="", flush=True)
```

---

## Doctors

| Doctor | Specialization | Experience | Fee |
|--------|---------------|------------|-----|
| Dr. Prasad | General Physician | 15 years | ₹500 |
| Dr. Anusha | Cardiologist | 20 years | ₹1200 |

---

## How to run

```bash
cd projects/phase01_primitives/project02_clinic_assistant
python main.py
```

---

## Sample interactions

| Patient message | Classified as | Tool called | Result |
|----------------|--------------|-------------|--------|
| "I am Priya Mehta. I need a general physician on 15th June." | action=check, urgency=routine | `check_availability("2026-06-15", "General Physician")` | Shows Dr. Prasad slots |
| "Can you tell me about Dr. Anusha?" | action=info, urgency=routine | `get_doctor_info("Dr. Anusha")` | Cardiologist profile + fee |
| "Please cancel my appointment APT001." | action=cancel, urgency=urgent | `cancel_appointment("APT001")` | Cancels Ravi Kumar's booking |
