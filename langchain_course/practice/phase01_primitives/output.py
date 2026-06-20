from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, TypedDict 


llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)


class SupportTicket(BaseModel):
    id: int
    title: str
    description: str
    status: Literal["open", "closed"] = Field(default="open")
    priority: Optional[Literal["low", "medium", "high"]] = None


structured_llm = llm.with_structured_output(SupportTicket)


result = structured_llm.invoke([
    SystemMessage("You are a helpful support agent."),
    HumanMessage("Create a support ticket for a user who can't log in to their account.")
])

print(result)  
print("\n")         # SupportTicket object
print("ID: ",result.id)
print("Title: ",result.title)
print("Description: ",result.description)
print("Status: ",result.status)
print("\n") 