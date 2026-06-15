"""
Voyago Travel Booking Assistant
main.py: Agent setup and 5 scenarios + async event streaming.

Features: create_agent · context_schema · @dynamic_prompt · PIIMiddleware
          HumanInTheLoopMiddleware · InMemorySaver (STM) · InMemoryStore (LTM)
          .stream() · astream_events · with_structured_output(BookingConfirmation)
"""
import asyncio
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, dynamic_prompt, HumanInTheLoopMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from schemas import Context, BookingConfirmation
from tools import (
    search_flights, book_flight, cancel_booking,
    search_hotels, book_hotel, get_my_preferences, save_preference,
)

llm    = ChatOllama(model="llama3.2", temperature=0)
store  = InMemoryStore()
saver  = InMemorySaver()

ALL_TOOLS = [
    search_flights, book_flight, cancel_booking,
    search_hotels, book_hotel, get_my_preferences, save_preference,
]

# ── Pre-populate returning customer (U002) ─────────────────────────────────────

store.put(("users", "U002"), "preferences", {
    "seat":         "window",
    "meal":         "vegetarian",
    "home_city":    "Hyderabad",
    "destinations": "Delhi, Mumbai",
})

# ── @dynamic_prompt — VIP gets lounge access mention ──────────────────────────

@dynamic_prompt
def voyago_prompt(request: ModelRequest[Context]) -> str:
    # context is None on resume calls — fall back to base prompt
    ctx  = request.runtime.context
    tier = ctx.loyalty_tier if ctx else "basic"
    base = (
        "You are a Voyago travel assistant. Today is 2026-06-15. "
        "Help customers search and book flights and hotels. Be concise. "
        "Always pass dates as YYYY-MM-DD when calling tools."
    )
    if tier == "vip":
        return (
            base + " This is a VIP customer: mention complimentary lounge access "
            "at major airports and priority check-in."
        )
    if tier in ("gold", "silver"):
        return base + f" This is a {tier.capitalize()} member — mention upgrade eligibility."
    return base


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    context_schema=Context,
    middleware=[
        voyago_prompt,
        PIIMiddleware("credit_card", strategy="redact", apply_to_input=True),
        PIIMiddleware("passport",    strategy="redact", apply_to_input=True,
                      detector=r"[A-Z]\d{7}"),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_flight":    True,
                "book_hotel":     True,
                "cancel_booking": True,
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

def resume(cfg, decision_type="approve", message=""):
    """Resume after interrupt — handles N simultaneous tool calls."""
    # peek at how many tool calls are pending
    pending = agent.get_state(cfg)
    interrupts = pending.tasks
    n = max(1, sum(1 for t in interrupts if t))  # at least 1

    decision = {"type": decision_type}
    if message:
        decision["message"] = message
    return agent.invoke(
        Command(resume={"decisions": [decision] * n}),
        config=cfg, version="v2",
    )

def approve_all(result, cfg):
    """Keep approving until no more interrupts."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 1 — New customer · search → pick → HITL approve → BookingConfirmation
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Scenario 1 — New customer, one-way trip")
print("=" * 60)

cfg1 = {"configurable": {"thread_id": "v-s1"}}
ctx1 = Context(user_id="U001", loyalty_tier="basic")

# Turn 1: search (no booking — no interrupt)
r = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Find flights from Hyderabad to Bangalore on 2026-06-20."}]},
    config=cfg1, context=ctx1, version="v2",
)
print(f"Search results:\n{last_msg(r)[:300]}")

# Turn 2: pick and book → HITL fires
r = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Book FL002 for Priya Sharma, aisle seat."}]},
    config=cfg1, context=ctx1, version="v2",
)
print(f"\nInterrupt → {r.interrupts[0].value['action_requests'][0]['name']} "
      f"| args: {r.interrupts[0].value['action_requests'][0]['args']}")

r = approve_all(r, cfg1)
reply = last_msg(r)
print(f"Agent: {reply[:200]}")

# with_structured_output → parse agent reply into typed BookingConfirmation
conf = llm.with_structured_output(BookingConfirmation).invoke([{
    "role": "user",
    "content": (
        f"Extract the flight booking details from this message and return a BookingConfirmation. "
        f"Set status='confirmed'. Message: \"{reply}\""
    )
}])
print(f"BookingConfirmation → {conf.booking_id} | ₹{conf.amount} | {conf.status}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 2 — Returning customer · LTM preferences → search → HITL
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 2 — Returning customer, saved preferences (Gold tier)")
print("=" * 60)

cfg2 = {"configurable": {"thread_id": "v-s2"}}
ctx2 = Context(user_id="U002", loyalty_tier="gold")

r = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Check my saved preferences. Then find a flight to Delhi on 2026-06-22 that matches them."}]},
    config=cfg2, context=ctx2, version="v2",
)
print(f"Agent (after reading prefs):\n{last_msg(r)[:300]}")

r = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Book FL003 for Ravi Kumar with window seat."}]},
    config=cfg2, context=ctx2, version="v2",
)
if r.interrupts:
    print(f"\nInterrupt → {r.interrupts[0].value['action_requests'][0]['name']}")
    r = approve_all(r, cfg2)

print(f"Agent: {last_msg(r)[:200]}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 3 — Multi-step: flight + hotel · two HITL pauses
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 3 — Flight + hotel in one session (two approvals)")
print("=" * 60)

cfg3 = {"configurable": {"thread_id": "v-s3"}}
ctx3 = Context(user_id="U003", loyalty_tier="silver")

# Turn 1: search both
r = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Search flights from Hyderabad to Mumbai on 2026-06-22, and hotels in Mumbai near BKC."}]},
    config=cfg3, context=ctx3, version="v2",
)
print(f"Search results:\n{last_msg(r)[:300]}")

# Turn 2: book both — model may batch or sequence the tool calls
r = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Book FL004 for Arjun Mehta aisle seat. Also book HT003 for 2 nights for Arjun Mehta."}]},
    config=cfg3, context=ctx3, version="v2",
)
r = approve_all(r, cfg3)
print(f"\nAgent: {last_msg(r)[:300]}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 4 — STM: mid-conversation edit (same thread remembers destination)
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 4 — Mid-conversation edit (short-term memory)")
print("=" * 60)

cfg4 = {"configurable": {"thread_id": "v-s4"}}
ctx4 = Context(user_id="U004", loyalty_tier="basic")

# Turn 1: search Chennai 18th
r = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Find flights to Chennai on 2026-06-18."}]},
    config=cfg4, context=ctx4, version="v2",
)
print(f"Turn 1:\n{last_msg(r)[:200]}")

# Turn 2: change date — agent must remember Chennai
r = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Actually, change the date to 2026-06-20 instead."}]},
    config=cfg4, context=ctx4, version="v2",
)
print(f"\nTurn 2 (date changed to 20th — agent remembers Chennai):\n{last_msg(r)[:200]}")


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 5 — PII: credit card + passport redacted before model sees them
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Scenario 5 — PII redaction (credit card + passport)")
print("=" * 60)

cfg5 = {"configurable": {"thread_id": "v-s5"}}
ctx5 = Context(user_id="U005", loyalty_tier="basic")

raw = "My card is 4111111111111111 and passport is Z1234567. Find me a flight to Bangalore."

r = agent.invoke(
    {"messages": [{"role": "user", "content": raw}]},
    config=cfg5, context=ctx5, version="v2",
)
# reject if HITL fires (no need to actually book here)
if r.interrupts:
    r = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "Test only — do not book."}]}),
        config=cfg5, version="v2",
    )

reply = last_msg(r)
print(f"Agent: {reply[:300]}\n")
print(f"Credit card (4111111111111111) in reply : {'❌ LEAKED' if '4111111111111111' in reply else '✅ redacted'}")
print(f"Passport   (Z1234567)          in reply : {'❌ LEAKED' if 'Z1234567' in reply else '✅ redacted'}")


# ══════════════════════════════════════════════════════════════════════════════
# Bonus — astream_events: real-time tool events while searching
# ══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Bonus — astream_events (VIP customer, real-time tool events)")
print("=" * 60)

async def stream_search():
    cfg = {"configurable": {"thread_id": "v-stream"}}
    ctx = Context(user_id="U006", loyalty_tier="vip")
    query = "What flights are available from Hyderabad to Bangalore on 2026-06-20?"
    print(f"User: {query}")
    print("Agent: ", end="")
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": query}]},
        config=cfg,
        context=ctx,
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_tool_start":
            print(f"\n  [→ {event['name']}]", end=" ")
        elif kind == "on_tool_end":
            out = str(event["data"].get("output", ""))[:100]
            print(f"\n  [← {out}]\nAgent: ", end="")
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_search())
