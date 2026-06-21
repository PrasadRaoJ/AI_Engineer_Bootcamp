from dotenv import load_dotenv
load_dotenv()
import os

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent,AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, wrap_model_call, SummarizationMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import RemoveMessage, AIMessage



llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)

# ── Example 1: basic multi-turn memory ────────────────────────────────────────

# agent = create_agent(
#     model = llm,
#     tools = [],
#     system_prompt = "You are a helpful assistant",
#     checkpointer = InMemorySaver(),
# )

# cfg = {"configurable":{"thread_id":"session_123"}}

# r1 = agent.invoke(
#     {"messages":[{"role":"user","content":"Hello, I am Prasad"}]},   
#     config=cfg
# )

# print(f"Response1: {r1['messages'][-1].content}")

# r2 = agent.invoke(
#     {"messages":[{"role":"user","content":"Hello, What is my name?"}]},   
#     config=cfg
# )

# print()
# print(f"Response2: {r2['messages'][-1].content}")

# print()
# r3 = agent.invoke(
#     {"messages":[{"role":"user","content":"Do you remember my name?"}]},
#     config={"configurable" :{"thread_id":"session_001", }}
# )


# print(f"Response3: {r3['messages'][-1].content}")



# ── Example 2: @before_model trim — keep last N messages ──────────────────────


# @before_model
# def trim_old_messages(state: AgentState, runtime):
#     messages = state["messages"]
#     if len(messages) > 4:
#         return {"messages":[RemoveMessage(id=m.id) for m in messages[:-4]]}
    

# agent = create_agent(
#     model = llm,
#     tools = [],
#     system_prompt = "You are a helpful assistant",
#     checkpointer = InMemorySaver(),
#     middleware = [trim_old_messages],
# )

# cfg = {"configurable":{"thread_id":"session_123"}}

# agent.invoke({"messages": [{"role": "user", "content": "My name is Prasad."}]}, config=cfg)
# agent.invoke({"messages": [{"role": "user", "content": "I work at Azkashine."}]}, config=cfg)
# agent.invoke({"messages": [{"role": "user", "content": "I live in Nellore."}]}, config=cfg)


# r = agent.invoke({"messages": [{"role": "user", "content": "What do you know about me?"}]}, config=cfg)
# print(f"Response: {r['messages'][-1].content}")

# ── Example 3: SummarizationMiddleware — compress history ─────────────────────

# summarizer = SummarizationMiddleware(
#     model=llm,
#     trigger = ("messages",5),
#     keep=("messages",2),
# )

# agent = create_agent(
#     model = llm,
#     tools = [],
#     system_prompt = "You are a helpful assistant",
#     checkpointer = InMemorySaver(),
#     middleware = [summarizer],
# )

# cfg = {"configurable":{"thread_id":"session_123"}}

# agent.invoke({"messages": [{"role": "user", "content": "My name is Prasad."}]}, config=cfg)
# agent.invoke({"messages": [{"role": "user", "content": "I work at Azkashine."}]}, config=cfg)
# agent.invoke({"messages": [{"role": "user", "content": "I live in Nellore."}]}, config=cfg)


# r = agent.invoke({"messages": [{"role": "user", "content": "What do you know about me?"}]}, config=cfg)
# print(f"Response: {r['messages'][-1].content}")



# ── Example 4: state_schema — store extra fields alongside messages ────────────

# FREE_TURN_LIMIT = 2

# class CustomState(AgentState):
    
#     turn_count: int=0  
#     user_plan: str="free"  # "free" | "pro" 


# @wrap_model_call
# def rate_limit(request: ModelRequest[CustomState], handler):
#     if request.state.get("turn_count", 0) >= FREE_TURN_LIMIT and request.state.get("user_plan", "free") == "free":
#         return AIMessage(content="Free limit reached. Upgrade to pro.")
#     return handler(request)


# agent = create_agent(
#     model = llm,
#     tools = [],
#     system_prompt = "You are a helpful assistant",
#     checkpointer = InMemorySaver(),
#     state_schema = CustomState,
#     middleware=[rate_limit]
# )
# cfg = {"configurable": {"thread_id": "state-session"}}

# r1 = agent.invoke(
#     {"messages": [{"role": "user", "content": "What is Python?"}], "turn_count": 1},
#     config=cfg,
# )
# print("reply:", r1["messages"][-1].content[:100])

# print("turn_count:", r1["turn_count"])

# r2 = agent.invoke(
#     {"messages": [{"role": "user", "content": "What is LangChain?"}], "turn_count": r1["turn_count"] + 1},
#     config=cfg,
# )

# print("turn_count:", r2["turn_count"])

# print("reply:", r2["messages"][-1].content[:100])



# ── Example 5: REMOVE_ALL_MESSAGES — clear thread history ─────────────────────


agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful assistant",
    checkpointer=InMemorySaver(),

)

cfg = {"configurable":{"thread_id":"session_123"}}

agent.invoke(
    {"messages":[{"role":"user","content":"Hi, My name is Prasad"}]},
    config=cfg
)

r = agent.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=cfg)
print("Before clear:", r["messages"][-1].content)

agent.update_state(cfg, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]})


r = agent.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=cfg)
print("After clear:", r["messages"][-1].content)