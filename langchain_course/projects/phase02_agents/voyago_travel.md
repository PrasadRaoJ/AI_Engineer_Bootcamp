# Voyago Travel Booking Assistant

## Problem Statement

Voyago is a travel startup that wants to replace its call-centre booking flow with an AI agent. Customers call in (simulated as text input) to search flights, book hotels, and manage their itinerary — all in one conversation. The agent must remember returning customers' preferences, stream results as it searches, pause for payment confirmation before committing a booking, and protect passport/card data from leaking into logs.

The challenge: a single customer might say "find me a flight to Mumbai on Friday, book a window seat, and a hotel near the airport" — one message, three actions, one coherent reply.

---

## What the agent must handle

1. A customer searches for available flights (date, origin, destination)
2. Agent streams results back token by token as it "searches"
3. Customer picks a flight — agent pauses for approval before booking (payment step)
4. Customer also wants a hotel — agent searches, pauses again before booking
5. Customer asks "what are my travel preferences?" — agent pulls from long-term store
6. Customer says "change the return date to Sunday" — agent remembers the whole conversation context (same thread)
7. A VIP customer gets a different system prompt — priority lanes, lounge access info

---

## Scenarios to implement

```
Scenario 1 — New customer, one-way trip
  "I need a flight from Hyderabad to Bangalore on June 20th. Aisle seat preferred."
  → search flights → pause (HITL) → book on approve → confirm

Scenario 2 — Returning customer, preferences remembered
  "Book me the usual — Delhi trip, window seat, vegetarian meal."
  → pull preference from store → search → pause → book

Scenario 3 — Multi-step in one turn
  "Flight to Mumbai on 22nd plus a hotel near BKC for 2 nights."
  → two tool calls → two HITL pauses → two approvals

Scenario 4 — Mid-conversation edit (short-term memory)
  Turn 1: "Find flights to Chennai on 18th June."
  Turn 2: "Actually make it 20th June instead."
  → agent remembers the destination, only updates the date

Scenario 5 — PII in input
  "Book using my card 4111-1111-1111-1111 and passport Z1234567."
  → credit_card and passport numbers must be redacted before model sees them
```

---

## Tools to build

| Tool | Args | Returns |
|------|------|---------|
| `search_flights` | `origin, destination, date` | list of flight options with price + timing |
| `book_flight` | `flight_id, passenger_name, seat_pref` | booking confirmation + PNR |
| `cancel_booking` | `booking_id` | cancellation confirmation + refund amount |
| `search_hotels` | `city, check_in, check_out, near` | list of hotel options |
| `book_hotel` | `hotel_id, guest_name, nights` | booking confirmation + room number |
| `get_my_preferences` | *(no args — reads from store via ToolRuntime)* | saved seat pref, meal, past destinations |
| `save_preference` | `key, value` | saves a preference to the store for this user |

Use mock data (dicts) — no real API calls needed.

---

## Schemas to define

```python
# Structured output — attach to every confirmed booking for audit
class BookingConfirmation(BaseModel):
    booking_id: str
    type: Literal["flight", "hotel"]
    passenger_name: str
    details: str          # "HYD → BLR, 20 Jun, 08:30"
    amount: float
    status: Literal["confirmed", "pending", "cancelled"]
```

---

## Phase 1 + 2 features and where they belong

| Feature | Where to use it |
|---------|----------------|
| `@tool` with `args_schema=` | All 7 tools — enforce typed inputs |
| `with_structured_output(BookingConfirmation)` | After every successful booking |
| `.stream()` | Stream the flight/hotel search results to the user |
| `astream_events` | Emit `on_tool_start` / `on_tool_end` events — show "Searching flights..." before result appears |
| `create_agent` | Main agent loop — replaces the manual Phase 1 tool loop |
| `context_schema=` + `ToolRuntime[Context]` | Pass `user_id` and `loyalty_tier` per call; tools read them from runtime |
| `@dynamic_prompt` | VIP customers get an upgraded system prompt mentioning lounge access and priority lanes |
| `InMemorySaver` + `thread_id` | Multi-turn edits ("change the date") within one booking session |
| `InMemoryStore` + `runtime.store` | Save and retrieve seat preference, meal preference, past destinations |
| `PIIMiddleware("credit_card")` | Redact card numbers from input before model sees them |
| `PIIMiddleware("ip", detector=r"[A-Z]\d{7}")` | Custom detector for passport numbers |
| `HumanInTheLoopMiddleware` | Interrupt on `book_flight` and `book_hotel` — never commit payment without approval |

---

## Context schema hint

```python
class Context(BaseModel):
    user_id: str
    loyalty_tier: Literal["basic", "silver", "gold", "vip"]
```

---

## Suggested file structure

```
projects/phase02_agents/projectA_travel_booking/
├── main.py          # agent setup, scenarios, invoke calls
├── tools.py         # all 7 tools + mock flight/hotel data
├── schemas.py       # BookingConfirmation, Context
└── README.md        # your own explanation after building it
```

---

## Hints

- `search_flights` doesn't need HITL — it's read-only. Only `book_flight` and `book_hotel` should interrupt.
- Use `when=` on `HumanInTheLoopMiddleware` to only pause bookings above a certain amount (e.g. skip HITL for flights under ₹1000).
- In `get_my_preferences`, use `runtime.store.get(("users", runtime.context.user_id), "preferences")` — return a friendly message if no preferences saved yet.
- For Scenario 4 (mid-conversation edit), the key insight is that `thread_id` must be the **same** across both turns — the agent will automatically remember the destination from turn 1.
- `@dynamic_prompt` receives `request.runtime.context.loyalty_tier` — use it to conditionally add VIP-specific lines to the system prompt.
- Event streaming hint: use `async for event in agent.astream_events(...)` and filter `event["event"] == "on_tool_start"` to show "Searching flights…" before the tool result arrives.
