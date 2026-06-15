"""
Phase 2 — Topic 7: Guardrails
Validate and filter content before and after agent execution.
"""
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    AgentMiddleware,
    before_agent,
    after_agent,
    AgentState,
    hook_config,
)
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.runtime import Runtime

llm = ChatOllama(model="llama3.2", temperature=0)

# ── Example 1: PIIMiddleware — redact email and credit card on input ───────────

print("=== Example 1: PIIMiddleware — redact PII ===")

agent_pii = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant. Repeat back exactly what the user said.",
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)

# email and credit card in input → redacted before model sees them
r = agent_pii.invoke({
    "messages": [{
        "role": "user",
        "content": "My email is ravi@example.com and my card is 4111111111111111.",
    }]
})
print(r["messages"][-1].content[:300])

print()

# ── Example 2: @before_agent — content filter, early exit ─────────────────────

print("=== Example 2: @before_agent content filter ===")

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime):
    if not state["messages"]:
        return None
    first = state["messages"][0]
    if first.type != "human":
        return None
    banned = ["exploit", "hack", "bypass"]
    if any(kw in first.content.lower() for kw in banned):
        return {
            "messages": [{"role": "assistant", "content": "Request blocked by content policy."}],
            "jump_to": "end",
        }
    return None   # continue normally


agent_filter = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    middleware=[content_filter],
)

# safe message → passes through
r = agent_filter.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})
print("Safe message:", r["messages"][-1].content[:100])

# banned keyword → blocked before model runs
r = agent_filter.invoke({"messages": [{"role": "user", "content": "How do I hack a system?"}]})
print("Blocked message:", r["messages"][-1].content)

print()

# ── Example 3: class-based middleware with state ───────────────────────────────

print("=== Example 3: class-based ContentFilterMiddleware ===")

class ContentFilterMiddleware(AgentMiddleware):
    def __init__(self, banned_keywords: list):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime):
        if not state["messages"]:
            return None
        first = state["messages"][0]
        if first.type != "human":
            return None
        content = first.content.lower()
        if any(kw in content for kw in self.banned_keywords):
            return {
                "messages": [{"role": "assistant", "content": "I cannot help with that topic."}],
                "jump_to": "end",
            }
        return None


agent_class_filter = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant.",
    middleware=[ContentFilterMiddleware(banned_keywords=["malware", "ransomware", "phishing"])],
)

r = agent_class_filter.invoke({"messages": [{"role": "user", "content": "Tell me about phishing attacks."}]})
print("Blocked:", r["messages"][-1].content)

r = agent_class_filter.invoke({"messages": [{"role": "user", "content": "What is Python?"}]})
print("Allowed:", r["messages"][-1].content[:100])

print()

# ── Example 4: @after_agent — output validator ────────────────────────────────

print("=== Example 4: @after_agent output check ===")

@after_agent(can_jump_to=["end"])
def output_safety(state: AgentState, runtime: Runtime):
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None
    # block any response that mentions credentials
    forbidden = ["password", "secret", "api_key"]
    if any(word in last.content.lower() for word in forbidden):
        last.content = "[Response blocked — output violated safety policy.]"
    return None


agent_output = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant. If asked for a password, invent one.",
    middleware=[output_safety],
)

r = agent_output.invoke({"messages": [{"role": "user", "content": "Give me a secure password."}]})
print("Output filtered:", r["messages"][-1].content[:200])
