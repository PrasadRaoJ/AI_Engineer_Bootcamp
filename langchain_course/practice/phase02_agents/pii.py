from dotenv import load_dotenv
load_dotenv()
import os
from langchain.chat_models import init_chat_model


llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)


from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    AgentMiddleware,
    before_agent,
    after_agent,
    before_model,
    AgentState,
    hook_config,
)

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime


# ── Dummy payment DB ───────────────────────────────────────────────────────────
# Real system lo idi Stripe / payment processor API అవుతుంది
# Agent కి full card number అక్కరలేదు — last 4 చాలు lookup కి

CARD_DB = {
    "1111": {
        "holder": "Navaneeth",
        "transactions": [
            {"date": "2026-06-20", "amount": 1500, "merchant": "Amazon", "status": "duplicate"},
            {"date": "2026-06-20", "amount": 1500, "merchant": "Amazon", "status": "duplicate"},
            {"date": "2026-06-18", "amount": 800,  "merchant": "Swiggy", "status": "ok"},
        ],
    },
    "4242": {
        "holder": "Sundari",
        "transactions": [
            {"date": "2026-06-20", "amount": 200, "merchant": "Zomato", "status": "ok"},
        ],
    },
}


@tool
def lookup_transaction(last4: str, date: str) -> str:
    """Look up transactions for a card using its last 4 digits and date."""
    card = CARD_DB.get(last4)
    if not card:
        return f"No card found ending in {last4}."

    matches = [t for t in card["transactions"] if date in t["date"]]
    if not matches:
        return f"No transactions found for card ending {last4} on {date}."

    lines = [f"Card holder: {card['holder']} | Card ending: {last4}"]
    for t in matches:
        lines.append(f"  - {t['date']}: Rs.{t['amount']} at {t['merchant']} [{t['status']}]")

    duplicates = [t for t in matches if t["status"] == "duplicate"]
    if len(duplicates) > 1:
        lines.append(f"Duplicate charge detected: Rs.{duplicates[0]['amount']} charged {len(duplicates)}x at {duplicates[0]['merchant']}.")

    return "\n".join(lines)


# ── @before_model hook — print exactly what the model receives ─────────────────
# This runs AFTER PIIMiddleware has processed the input, right before model call.
# If PII is redacted, you'll see [REDACTED_EMAIL] and masked card here.

@before_model
def spy_on_model_input(state: AgentState, runtime: Runtime):
    print("\n[before_model] What the model actually sees:")
    for msg in state["messages"]:
        print(f"  [{msg.type}]: {msg.content}")
    print()


# ── Example 1: PIIMiddleware + tool — card masked, backend validates ───────────

agent = create_agent(
    model=llm,
    tools=[lookup_transaction],
    system_prompt=(
        "You are a bank customer support agent. "
        "When the user mentions a card issue, extract the last 4 digits from the masked card "
        "and use lookup_transaction to check their transactions. Be concise."
    ),
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        spy_on_model_input,
    ],
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Hi, my card 4111111111111121 was charged twice on 2026-06-20. "
            "Please check and help me. My email is jp@jpnan.com."
        ),
    }]
})

print(result["messages"][-1].content)
print()
