from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
import os

from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call
from langchain.agents.middleware.types import ModelRequest
from langchain_core.tools import tool

llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)


class Context(BaseModel):
    user_name: str
    language: str       # "telugu" | "english"


@dynamic_prompt
def personalized_prompt(request: ModelRequest[Context]) -> str:
    name = request.runtime.context.user_name
    lang = request.runtime.context.language
    return (
        f"You are a formal Slipkart support agent. "
        f"Address the customer as {name}. Reply in {lang}."
    )


# create_agent requires at least one tool — with tools=[] it loops forever
@tool
def get_store_info(topic: str) -> str:
    """Get Slipkart store information: return_policy, shipping, payment."""
    info = {
        "return_policy": "30-day easy returns on all products. No questions asked.",
        "shipping": "Free shipping on orders above ₹500. Delivered in 3-5 days.",
        "payment": "We accept UPI, credit/debit cards, net banking, and cash on delivery.",
    }
    return info.get(topic, "Please contact support for more details.")




agent = create_agent(
    model=llm,
    tools=[get_store_info],
    context_schema=Context,
    middleware=[personalized_prompt])


result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is your return policy?"}]},
    context=Context(user_name="Ravi", language="telugu")
)
print(result["messages"][-1].content)


result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is your return policy?"}]},
    context=Context(user_name="Prasad", language="english")
)
print(result["messages"][-1].content)