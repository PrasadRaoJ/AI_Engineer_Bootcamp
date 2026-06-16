# Voyago Travel Booking Assistant

A production-style AI travel agent built with LangChain agents and LangGraph primitives.
Covers every Phase 1 + 2 feature in one coherent project: tools, structured output, streaming,
agents, context injection, dynamic prompts, PII redaction, human-in-the-loop, short-term memory,
and long-term memory.

---

## What it does

A customer sends natural language — "find me a flight to Delhi, book the cheap one, and a hotel near the airport" — and the agent:

1. Calls the right tools in sequence (search → interrupt for approval → book)
2. Remembers the whole conversation within a session (short-term memory)
3. Loads the customer's saved preferences from a persistent store (long-term memory)
4. Strips passport numbers and credit cards before the model ever sees them (PII middleware)
5. Pauses before any payment action and waits for a human to approve or reject (HITL)
6. Gives VIP customers a different system prompt with lounge access info (@dynamic_prompt)
7. Parses the final booking reply into a typed `BookingConfirmation` object (structured output)

---

## File structure

```
voyago_travel/
├── schemas.py   — Pydantic models: Context (per-call injection) and BookingConfirmation
├── tools.py     — 7 tools + mock FLIGHTS/HOTELS data + booking counter
├── main.py      — agent setup, middleware stack, 5 scenarios + async event streaming
└── README.md    — this file
```

---

## schemas.py — Pydantic models

```python
class Context(BaseModel):
    user_id: str
    loyalty_tier: Literal["basic", "silver", "gold", "vip"]
```

`Context` is the **per-call injection payload**. It is not a message — it never reaches the LLM
directly. You pass it as `context=ctx` on every `agent.invoke(...)` call. LangChain routes it
through the agent's `context_schema=Context` declaration and makes it available in two places:

- `request.runtime.context` inside `@dynamic_prompt` — to personalise the system prompt
- `runtime.context` inside any tool declared as `def my_tool(runtime: ToolRuntime[Context])` —
  to read `runtime.context.user_id` or `runtime.context.loyalty_tier`

```python
class BookingConfirmation(BaseModel):
    booking_id: str
    type: Literal["flight", "hotel"]
    passenger_name: str
    details: str
    amount: float
    status: Literal["confirmed", "pending", "cancelled"]
```

`BookingConfirmation` is used with `llm.with_structured_output(BookingConfirmation)` to parse
the agent's natural-language booking reply into a typed Python object. This is the Phase 1
structured output pattern applied to agent output rather than direct LLM output.

---

## tools.py — Tools and mock data

### Mock data

```python
FLIGHTS = {
    "FL001": {"origin": "HYD", "destination": "BLR", "date": "2026-06-20", "time": "08:30", "price": 4200, "airline": "IndiGo"},
    ...
}

HOTELS = {
    "HT001": {"city": "bangalore", "name": "Voyago Grand Bangalore", "near": "airport", "price_per_night": 4500},
    ...
}

BOOKINGS = {}   # populated at runtime by book_flight / book_hotel
_counter = [1]  # mutable list so the closure can mutate it
```

All data lives in plain dicts. `BOOKINGS` is populated at runtime when a booking tool executes.
`_counter` uses a mutable list rather than a bare `int` so the `_next_bid()` function can
increment it without needing `global` (Python closures can mutate a list but not rebind an int).

### City normaliser

```python
_CITY_CODES = {
    "hyderabad": "HYD", "bangalore": "BLR", "bengaluru": "BLR",
    "delhi": "DEL", "new delhi": "DEL",
    "mumbai": "BOM", "bombay": "BOM",
    "chennai": "MAA", "madras": "MAA",
}

def _code(city: str) -> str:
    return _CITY_CODES.get(city.lower().strip(), city.upper().strip())
```

`_code` maps any user-facing city name ("Bengaluru", "Bombay") to its IATA code. The fallback
is `.upper().strip()` so if the LLM already passes a code like `"BOM"` it passes through
unchanged. `search_flights` uses `_code` for both origin and destination. `search_hotels` uses
the same lookup table in reverse (code → city name) for its hotel dict lookup.

### Tools declared with `@tool`

Every tool is decorated with `@tool`, which turns it into a `StructuredTool` that:
- generates a JSON schema from the type annotations
- exposes a `.name` and `.description` that the LLM reads to decide when to call it
- makes it passable as an element of the `tools=` list on `create_agent`

**`search_flights`** — read-only, no interrupt

```python
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    orig, dest = _code(origin), _code(destination)
    matches = [
        f"  {fid}: {f['airline']} | {f['time']} | ₹{f['price']}"
        for fid, f in FLIGHTS.items()
        if f["origin"] == orig and f["destination"] == dest and f["date"] == date
    ]
    if not matches:
        return f"No flights found from {orig} to {dest} on {date}."
    return f"Flights from {orig} to {dest} on {date}:\n" + "\n".join(matches)
```

Returns a formatted string the LLM reads and then repeats to the user. Because it is read-only,
`HumanInTheLoopMiddleware` does NOT interrupt on it — only on the three mutating tools below.

**`book_flight`** — mutating, triggers HITL

```python
@tool
def book_flight(flight_id: str, passenger_name: str, seat_pref: str) -> str:
    fl = FLIGHTS.get(flight_id.upper())
    if not fl:
        return f"Flight {flight_id} not found."
    bid = _next_bid()
    BOOKINGS[bid] = { ... }
    return f"Flight booked! Booking ID: {bid} | ..."
```

`flight_id.upper()` normalises whatever the LLM passes ("FL002", "fl002") to the dict key format.
The HITL middleware intercepts this tool call before it executes and pauses the agent.

**`cancel_booking`** — mutating, triggers HITL

```python
@tool
def cancel_booking(booking_id: str) -> str:
    b = BOOKINGS.pop(booking_id.upper(), None)
    if not b:
        return f"Booking {booking_id} not found."
    refund = b["amount"] * 0.9
    return f"Booking {booking_id} cancelled. Refund of ₹{refund:.0f} in 3-5 business days."
```

`dict.pop(key, None)` atomically removes and returns the booking in one step. Refund is 90% of
the original amount. Double-cancelling or cancelling a non-existent ID returns a safe error string
rather than raising an exception (tools should always return strings, never raise).

**`search_hotels`** — read-only, no interrupt

```python
@tool
def search_hotels(city: str, check_in: str, check_out: str, near: str = "") -> str:
    city_norm = _CITY_CODES.get(city.lower().strip(), city.upper().strip())
    code_to_city = {"BLR": "bangalore", "BOM": "mumbai", "DEL": "delhi", ...}
    city_norm = code_to_city.get(city_norm, city_norm.lower())
    ...
```

The two-step normalisation: first map city name → IATA code (or keep the code if already given),
then reverse-map code → hotel dict key ("mumbai"). The `near` filter does a substring match
against `h["near"]` so "bkc" matches "BKC Business Suites".

**`book_hotel`** — mutating, triggers HITL

Mirrors `book_flight`. `hotel_id.upper()` normalises input; total = `price_per_night * nights`.

**`get_my_preferences`** — LTM read, uses `ToolRuntime[Context]`

```python
@tool
def get_my_preferences(runtime: ToolRuntime[Context]) -> str:
    item = runtime.store.get(("users", runtime.context.user_id), "preferences")
    if item is None:
        return "No preferences saved yet."
    lines = [f"  {k}: {v}" for k, v in item.value.items()]
    return "Your saved preferences:\n" + "\n".join(lines)
```

`ToolRuntime[Context]` is the signal to LangChain to auto-inject the runtime (store, context,
config) as the `runtime` parameter when calling this tool. The agent never passes it explicitly —
the framework fills it in. `runtime.store` is the `InMemoryStore` bound to the agent.
`runtime.context.user_id` is the `user_id` from the `Context` object you passed on `agent.invoke`.
The store key is a tuple `("users", uid)` — a namespace path.

**`save_preference`** — LTM write, uses `ToolRuntime[Context]`

```python
@tool
def save_preference(key: str, value: str, runtime: ToolRuntime[Context]) -> str:
    uid = runtime.context.user_id
    item = runtime.store.get(("users", uid), "preferences")
    prefs = dict(item.value) if item else {}
    prefs[key] = value
    runtime.store.put(("users", uid), "preferences", prefs)
    return f"Saved preference — {key}: {value}."
```

Reads the existing preferences dict (if any), merges the new key-value pair in, and writes the
whole dict back. This pattern avoids losing existing preferences on every save — read-merge-write
rather than overwrite. The LLM never sees the store internals; it just gets a confirmation string.

---

## main.py — Agent setup and scenarios

### Imports and singletons

```python
llm    = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
store  = InMemoryStore()
saver  = InMemorySaver()
```

- `init_chat_model` — provider-agnostic LLM via env vars, `temperature=0` for deterministic tool calls
- `InMemoryStore` — long-term store that persists across threads and sessions (within a process)
- `InMemorySaver` — checkpoint backend that persists conversation history per `thread_id`

### Pre-populating LTM for the returning customer

```python
store.put(("users", "U002"), "preferences", {
    "seat":         "window",
    "meal":         "vegetarian",
    "home_city":    "Hyderabad",
    "destinations": "Delhi, Mumbai",
})
```

This simulates a returning customer (U002) who has used Voyago before. The `store.put` call
writes directly to the store before any agent invoke. In a real app this data would come from a
database. Scenario 2 demonstrates the agent reading these preferences automatically.

### @dynamic_prompt — tier-based system prompt

```python
@dynamic_prompt
def voyago_prompt(request: ModelRequest[Context]) -> str:
    ctx  = request.runtime.context
    tier = ctx.loyalty_tier if ctx else "basic"
    base = (
        "You are a Voyago travel assistant. Today is 2026-06-15. "
        "Help customers search and book flights and hotels. Be concise. "
        "Always pass dates as YYYY-MM-DD when calling tools."
    )
    if tier == "vip":
        return base + " This is a VIP customer: mention complimentary lounge access ..."
    if tier in ("gold", "silver"):
        return base + f" This is a {tier.capitalize()} member — mention upgrade eligibility."
    return base
```

`@dynamic_prompt` is a middleware decorator that runs before every model call. It receives the
full `ModelRequest` and must return a string that becomes the system prompt for that call.

**Critical guard:** `ctx = request.runtime.context` is `None` on `Command(resume=...)` calls.
When the HITL middleware resumes after a human decision, the agent re-invokes the model without
a new `context=` argument — so `request.runtime.context` is `None`. Without the `if ctx else "basic"`
guard, this raises `AttributeError: 'NoneType' object has no attribute 'loyalty_tier'`. The guard
falls back to the base prompt safely.

### Middleware stack

```python
agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    context_schema=Context,
    middleware=[
        voyago_prompt,                                             # 1. system prompt
        PIIMiddleware("credit_card", strategy="redact",           # 2. strip cards
                      apply_to_input=True),
        PIIMiddleware("passport", strategy="redact",              # 3. strip passports
                      apply_to_input=True, detector=r"[A-Z]\d{7}"),
        HumanInTheLoopMiddleware(interrupt_on={                   # 4. HITL gate
            "book_flight": True,
            "book_hotel":  True,
            "cancel_booking": True,
        }),
    ],
    checkpointer=saver,
    store=store,
)
```

Middleware runs **in order, top to bottom**, on every model call:

1. `voyago_prompt` — injects the personalised system prompt first, before any PII is even checked
2. `PIIMiddleware("credit_card")` — scans the human message for credit card patterns and replaces
   them with `[REDACTED]` before the string reaches the model. Uses the built-in Luhn-format
   pattern.
3. `PIIMiddleware("passport", detector=r"[A-Z]\d{7}")` — same strategy but with a custom regex
   for Indian passport format (one uppercase letter + 7 digits). The built-in `"passport"` type
   uses a different pattern, so a custom `detector=` regex is needed.
4. `HumanInTheLoopMiddleware` — intercepts tool calls. When the agent selects `book_flight`,
   `book_hotel`, or `cancel_booking`, the agent pauses and returns a result with `.interrupts`
   set. No tool actually executes until a `Command(resume=...)` is sent.

`checkpointer=saver` wires up short-term memory. Every message in the conversation is
checkpointed under the `thread_id` in the config. The next `agent.invoke` on the same thread_id
automatically restores the prior conversation.

`store=store` wires up long-term memory. The same `InMemoryStore` instance is bound to
`runtime.store` inside every tool that declares `ToolRuntime[Context]`.

### Helper: `last_msg`

```python
def last_msg(result) -> str:
    msgs = result.value["messages"] if hasattr(result, "value") else result["messages"]
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return ""
```

`result.value["messages"]` is the correct accessor (not the deprecated `result["messages"]`).
Iterating in reverse finds the most recent non-empty `AIMessage`. Empty-content `AIMessage`s
appear when the model issues a tool call — `content=""` with `tool_calls=[...]`. Skipping those
ensures you get the human-readable reply, not the internal tool-dispatch step.

### Helper: `approve_all`

```python
def approve_all(result, cfg):
    while result.interrupts:
        n = len(result.interrupts[0].value["action_requests"])
        decisions = [{"type": "approve"}] * n
        for req in result.interrupts[0].value["action_requests"]:
            print(f"    Approving → {req['name']}({list(req['args'].values())})")
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=cfg, version="v2",
        )
    return result
```

`result.interrupts` is a tuple — truthy when the agent is paused, empty when it has finished.
The loop handles the case where a single agent response triggered multiple simultaneous tool calls
(e.g. Scenario 3 where the model may batch `book_flight` + `book_hotel` in one step). When that
happens, `len(result.interrupts[0].value["action_requests"])` is 2, and you must send exactly 2
decisions — one per pending tool call. Sending only 1 decision raises:
`ValueError: Number of human decisions (1) does not match number of hanging tool calls (2)`.

---

## Scenario walkthroughs

### Scenario 1 — New customer, HITL approve → BookingConfirmation

```
Turn 1: "Find flights from Hyderabad to Bangalore on 2026-06-20."
        → agent calls search_flights("HYD", "BLR", "2026-06-20")
        → no interrupt (read-only tool)
        → agent replies with FL001/FL002 options

Turn 2: "Book FL002 for Priya Sharma, aisle seat."
        → agent calls book_flight("FL002", "Priya Sharma", "aisle")
        → HumanInTheLoopMiddleware fires → result.interrupts is truthy
        → approve_all sends {"type": "approve"}
        → tool executes → agent replies with confirmation

Parse reply → llm.with_structured_output(BookingConfirmation).invoke(...)
           → returns typed BookingConfirmation(booking_id="BK001", amount=3800.0, ...)
```

Two turns are required. On Turn 1 the agent presents options — it does not book unprompted.
On Turn 2 the explicit flight ID and passenger name give the model everything it needs to call
`book_flight` without ambiguity.

### Scenario 2 — Returning customer: LTM → search → HITL

```
Turn 1: "Check my saved preferences. Then find a flight to Delhi on 2026-06-22 that matches them."
        → agent calls get_my_preferences()
          (runtime.context.user_id = "U002", reads from InMemoryStore)
        → store returns: seat=window, meal=vegetarian, home_city=Hyderabad, destinations=Delhi/Mumbai
        → agent calls search_flights("HYD", "DEL", "2026-06-22")
        → replies with FL003 details mentioning Gold tier upgrade eligibility

Turn 2: "Book FL003 for Ravi Kumar with window seat."
        → HumanInTheLoopMiddleware fires → approve
        → booking confirmed
```

The `@dynamic_prompt` sees `tier="gold"` in the context and appends the upgrade eligibility
line to the system prompt. The LTM preferences loaded by `get_my_preferences` give the model
context to recommend window-seat flights.

### Scenario 3 — Flight + hotel: two HITL pauses

```
Turn 1: search flights HYD→BOM + hotels in Mumbai near BKC
        → two parallel tool calls (search_flights + search_hotels) — no interrupt on either

Turn 2: "Book FL004 for Arjun Mehta aisle. Also book HT003 for 2 nights."
        → model may batch book_flight + book_hotel in one step (2 simultaneous tool calls)
        → approve_all reads n=2, sends [approve, approve]
        → both tools execute, agent replies with both confirmations
```

This is the scenario that revealed the `n` decisions requirement. `approve_all` handles it
by always reading `len(action_requests)` dynamically rather than hardcoding 1.

### Scenario 4 — STM: mid-conversation date edit

```
Turn 1 (thread_id="v-s4"): "Find flights to Chennai on 2026-06-18."
        → agent searches, InMemorySaver checkpoints this turn

Turn 2 (same thread_id):   "Actually, change the date to 2026-06-20 instead."
        → agent loads checkpoint: full prior conversation is in context
        → model knows "Chennai" from Turn 1, only the date changes
        → calls search_flights("HYD", "MAA", "2026-06-20")
```

Without `thread_id` being the same across both turns, Turn 2 would have no context and the agent
would ask "change what to 20th?". The checkpoint replays the full message history automatically —
no extra code needed beyond matching `thread_id`.

### Scenario 5 — PII redaction

```
Input: "My card is 4111111111111111 and passport is Z1234567. Find me a flight to Bangalore."

PIIMiddleware("credit_card") runs first:
  "4111111111111111" → "[REDACTED]" (matches Luhn credit card pattern)

PIIMiddleware("passport", detector=r"[A-Z]\d{7}") runs next:
  "Z1234567" → "[REDACTED]" (matches custom Indian passport regex)

Model sees: "My card is [REDACTED] and passport is [REDACTED]. Find me a flight to Bangalore."

Agent reply: never contains the original card or passport values.
```

The `apply_to_input=True` flag tells each `PIIMiddleware` to scan the human message before it
reaches the model. Default (`apply_to_input=False`) would only scan the model's output — useful
for preventing leaks in replies, but not for stripping sensitive input.

### Bonus — astream_events: real-time tool events

```python
async def stream_search():
    async for event in agent.astream_events(..., version="v2"):
        if event["event"] == "on_tool_start":
            print(f"  [→ {event['name']}]")
        elif event["event"] == "on_tool_end":
            print(f"  [← {event['data']['output'][:100]}]")
        elif event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
```

`astream_events` is the async version of the agent invoke that emits events at each step:
- `on_chat_model_stream` — one chunk per token as the LLM generates output
- `on_tool_start` — emitted before a tool executes (useful for "Searching flights..." UI updates)
- `on_tool_end` — emitted after the tool returns with its result

`version="v2"` is required — it matches the event schema that LangGraph uses for tool
call events. The VIP customer `(tier="vip")` gets the lounge access system prompt, verifiable
by reading the first `on_chat_model_stream` event's accumulated content.

---

## Phase 1 + 2 features — where each lives

| Feature | Phase | File | What it does here |
|---------|-------|------|--------------------|
| `@tool` | 1 | tools.py | Wraps all 7 functions into structured LangChain tools |
| `with_structured_output` | 1 | main.py S1 | Parses agent booking reply into `BookingConfirmation` |
| `.stream()` / `astream_events` | 1 | main.py Bonus | Real-time token + tool event streaming |
| `create_agent` | 2 | main.py | Replaces the manual tool loop from Phase 1 examples |
| `context_schema=Context` | 2 | main.py | Declares the per-call injection type |
| `ToolRuntime[Context]` | 2 | tools.py | Auto-injects runtime into `get_my_preferences` / `save_preference` |
| `@dynamic_prompt` | 2 | main.py | Per-call system prompt based on loyalty tier |
| `PIIMiddleware` (built-in) | 2 | main.py | Redacts credit cards from input |
| `PIIMiddleware` (custom detector) | 2 | main.py | Custom regex for passport format |
| `HumanInTheLoopMiddleware` | 2 | main.py | Pauses before any booking or cancellation |
| `Command(resume=...)` | 2 | main.py | Sends human decision back to paused agent |
| `InMemorySaver` + `thread_id` | 2 | main.py | Persists conversation across turns (STM) |
| `InMemoryStore` | 2 | main.py | Persists user preferences across threads (LTM) |

---

## Key gotchas discovered during build

**1. `@dynamic_prompt` context is `None` on resume**

When `Command(resume=...)` resumes a paused agent, the framework re-runs the middleware stack
to generate the model call that processes the tool result. At that point there is no `context=`
argument — it was only on the original `agent.invoke(...)`. So `request.runtime.context` is
`None`. The fix is to always guard:

```python
ctx  = request.runtime.context
tier = ctx.loyalty_tier if ctx else "basic"
```

**2. Multiple simultaneous tool calls need N decisions**

When the model batches two tool calls in one response (e.g. `book_flight` + `book_hotel`), the
HITL interrupt holds both. You must send exactly as many decisions as there are pending calls:

```python
n = len(result.interrupts[0].value["action_requests"])
decisions = [{"type": "approve"}] * n
```

Sending 1 decision for 2 tool calls raises `ValueError` immediately.

**3. `result.interrupts` vs `hasattr(result, "interrupts")`**

`hasattr(result, "interrupts")` is always `True` — the attribute exists even when the tuple is
empty. Use truthiness: `if result.interrupts:` checks whether the tuple is non-empty.

**4. `result.value["messages"]` not `result["messages"]`**

The `GraphRecursionError` or `KeyError` pattern. `result` is an `AddableValuesDict` (or
similar), not a plain dict. Subscript access on the result object is deprecated in LangGraph 1.x.
Always use `result.value["messages"]`.

**5. Pass YYYY-MM-DD dates in prompts**

Telling the model "find flights for June 22nd" causes inconsistent tool call args — sometimes
`"2026-06-22"`, sometimes `"June 22, 2026"`, sometimes `"22 June"`. The system prompt instructs
`"Always pass dates as YYYY-MM-DD when calling tools"` and scenario prompts pass explicit dates
like `"2026-06-22"`. This alone eliminates most tool call failures.

---

## How to run

```bash
cd langchain_course/projects/phase02_agents/voyago_travel
python main.py
```

Requires Ollama running locally with `llama3.2` pulled:

```bash
ollama pull llama3.2
ollama serve
```
