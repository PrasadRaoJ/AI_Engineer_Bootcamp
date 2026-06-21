from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call
from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel
import os


llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)


class Context(BaseModel):
    user_name: str
    language: str   # "telugu" | "english"
    role: str       # "admin" | "customer"


@dynamic_prompt
def personalized_prompt(request: ModelRequest[Context]) -> str:
    name = request.runtime.context.user_name
    lang = request.runtime.context.language
    return f"You are a formal Slipkart support agent. Address the customer as {name}. Reply in {lang}."
           
@tool
def get_order_status(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Get the delivery status of a Slipkart order."""
    return f"ORD123: Out for delivery. ORD456: Delivered."

get_order_status.metadata = {"roles": ["admin", "customer"]}


@tool
def cancel_order(order_id: str, runtime: ToolRuntime[Context]) -> str:
    """Cancel a Slipkart order."""
    if runtime.context.role != "admin":
        return "Permission denied. Only admins can cancel."
    return f"Order {order_id} cancelled."

cancel_order.metadata = {"roles": ["admin"]}


@wrap_model_call
def filter_by_role(request: ModelRequest[Context], handler):
    role = request.runtime.context.role
    filtered = [t for t in request.tools if role in t.metadata.get("roles", [role])]
    request = request.override(tools=filtered)
    print(f"  [middleware] {role} sees: {[t.name for t in request.tools]}")
    return handler(request)


agent = create_agent(
    model=llm,
    tools=[get_order_status, cancel_order],
    context_schema=Context,
    middleware=[personalized_prompt, filter_by_role],
)


# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "Cancel order ORD123."}]},
#     context=Context(user_name="Ravi", language="telugu", role="customer"),
# )
# print(result["messages"][-1].content)



result = agent.invoke(
    {"messages": [{"role": "user", "content": "Cancel order ORD123."}]},
    context=Context(user_name="Prasad", language="english", role="admin"),
)
print(result["messages"][-1].content)

+
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "What is my name"}]},
#     context=Context(user_name="Prasad", language="english", role="admin"),
# )
# print(result["messages"][-1].content)

