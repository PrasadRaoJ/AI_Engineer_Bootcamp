from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

"""
Phase 2 — Topic 7: Guardrails
Real use case: Bank customer support chatbot.

PIIMiddleware   — strip card/email before model ever sees them (PCI-DSS / GDPR safe)
@before_agent   — block off-topic questions before wasting a model call
class middleware — reusable topic filter, configurable per deployment
@after_agent    — prevent model from leaking internal system details in response
"""
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
from langgraph.runtime import Runtime

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2


# ── Example 1: PIIMiddleware — card number masked before model sees it ─────────
#
# User reports a billing issue and pastes their card number in the message.
# PIIMiddleware strips card + email BEFORE sending to the model.
# AI response and logs never contain real card numbers — PCI-DSS / GDPR safe.

print("=== Example 1: PIIMiddleware — billing complaint ===")

# @before_model runs AFTER PIIMiddleware has processed the input.
# Prints exactly what the model receives — confirms PII is stripped.
@before_model
def spy_on_model_input(state: AgentState, runtime: Runtime):
    print("[before_model] What the model actually sees:")
    for msg in state["messages"]:
        print(f"  [{msg.type}]: {msg.content}")
    print()

support_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "You are a bank customer support agent. "
        "Help the user with their billing issue. Be concise."
    ),
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        spy_on_model_input,   # confirm PII is stripped before model call
    ],
)

r = support_agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Hi, my card 4111111111111111 was charged twice on June 20. "
            "Please help. My email is john@example.com."
        ),
    }]
})
print(r["messages"][-1].content[:300])

print()


# ── Example 2: @before_agent — block off-topic questions ──────────────────────
#
# Support bots should only answer banking questions.
# @before_agent catches off-topic input and short-circuits — model is never called.
# Saves tokens and prevents the bot being used as a general chatbot.

print("=== Example 2: @before_agent — block off-topic questions ===")

BANKING_TOPICS = ["card", "account", "transaction", "charge", "transfer", "balance", "loan"]

@before_agent(can_jump_to=["end"])
def topic_guard(state: AgentState, runtime: Runtime):
    if not state["messages"]:
        return None
    first = state["messages"][0]
    if first.type != "human":
        return None
    if not any(topic in first.content.lower() for topic in BANKING_TOPICS):
        return {
            "messages": [{"role": "assistant", "content": "I can only help with banking queries. Please contact the right department."}],
            "jump_to": "end",
        }
    return None   # continue normally

support_agent2 = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a bank customer support agent. Be concise.",
    middleware=[topic_guard],
)

# on-topic → passes through to model
r = support_agent2.invoke({"messages": [{"role": "user", "content": "What is my account balance?"}]})
print("Banking question:", r["messages"][-1].content[:120])

# off-topic → blocked before model is called
r = support_agent2.invoke({"messages": [{"role": "user", "content": "Can you write me a poem?"}]})
print("Off-topic blocked:", r["messages"][-1].content)

print()


# ── Example 3: class-based TopicFilterMiddleware ───────────────────────────────
#
# Same as Example 2 but as a reusable class — configurable per deployment.
# Retail bank uses ["card", "loan"], crypto exchange uses ["wallet", "trade"].

print("=== Example 3: class-based TopicFilterMiddleware ===")

class TopicFilterMiddleware(AgentMiddleware):
    def __init__(self, allowed_topics: list, fallback_message: str):
        super().__init__()
        self.allowed_topics = [t.lower() for t in allowed_topics]
        self.fallback_message = fallback_message

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        if not state["messages"]:
            return None
        first = state["messages"][0]
        if first.type != "human":
            return None
        if not any(topic in first.content.lower() for topic in self.allowed_topics):
            return {
                "messages": [{"role": "assistant", "content": self.fallback_message}],
                "jump_to": "end",
            }
        return None

support_agent3 = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a bank customer support agent. Be concise.",
    middleware=[TopicFilterMiddleware(
        allowed_topics=["card", "account", "transaction", "charge", "transfer", "balance", "loan"],
        fallback_message="I'm a banking assistant. I can only help with account, card, and transaction queries.",
    )],
)

r = support_agent3.invoke({"messages": [{"role": "user", "content": "Tell me about cooking recipes."}]})
print("Off-topic:", r["messages"][-1].content)

r = support_agent3.invoke({"messages": [{"role": "user", "content": "I need to transfer money to another account."}]})
print("On-topic:", r["messages"][-1].content[:120])

print()


# ── Example 4: @after_agent — prevent leaking internal system details ──────────
#
# The model might accidentally say "our backend team will check the database".
# @after_agent scans the final response and replaces it if forbidden words appear.
# Users never see internal architecture details.

print("=== Example 4: @after_agent — block internal details in response ===")

@after_agent(can_jump_to=["end"])
def output_guard(state: AgentState, runtime: Runtime):
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None
    forbidden = ["internal system", "backend", "database", "sql", "server error"]
    if any(word in last.content.lower() for word in forbidden):
        last.content = "[Response withheld — internal details cannot be shared with customers.]"
    return None

support_agent4 = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "You are a bank support agent. If you cannot resolve an issue, "
        "mention that it's an internal system problem and the backend team will fix it."
    ),
    middleware=[output_guard],
)

r = support_agent4.invoke({"messages": [{"role": "user", "content": "Why is my transfer failing?"}]})
print("Output:", r["messages"][-1].content[:200])
