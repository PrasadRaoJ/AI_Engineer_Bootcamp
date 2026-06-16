from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, TypedDict

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

# --- 1. Pydantic (recommended) ---

class SupportTicket(BaseModel):
    order_id: str
    days_waiting: int
    is_urgent: bool
    priority: Literal["low", "medium", "high"]
    tags: List[str]
    notes: Optional[str] = None
    issue: str = Field(description="Short description of the customer's problem")

structured_llm = llm.with_structured_output(SupportTicket)

result = structured_llm.invoke([
    SystemMessage("You are a Slipkart support ticket classifier. Extract structured details."),
    HumanMessage("My order ORD123 hasn't arrived in 10 days and I need it urgently for a wedding!"),
])

print("--- Pydantic ---")
print("order_id    :", result.order_id)      # object attribute access
print("days_waiting:", result.days_waiting)
print("is_urgent   :", result.is_urgent)
print("priority    :", result.priority)
print("tags        :", result.tags)
print("issue       :", result.issue)

# --- 2. TypedDict (lighter, returns dict) ---

class SimpleTicket(TypedDict):
    order_id: str
    priority: str

simple_llm = llm.with_structured_output(SimpleTicket)
simple = simple_llm.invoke([
    SystemMessage("You are a Slipkart support classifier."),
    HumanMessage("Order ORD456 is delayed."),
])

print("\n--- TypedDict ---")
print("order_id :", simple["order_id"])   # dict access — not simple.order_id
print("priority :", simple["priority"])

# NOTE: JSON schema with Groq requires a top-level "title" key (used as function name)
# NOTE: Pydantic and TypedDict work without title — recommended over raw JSON schema

# --- 3. JSON Schema (raw dict — no class needed) ---

json_schema = {
    "title": "RefundTicket",       # Groq requires title as the function name
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "description": "Order ID from the message"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "refund_eligible": {"type": "boolean"},
    },
    "required": ["order_id", "priority", "refund_eligible"],
}

js_llm = llm.with_structured_output(json_schema)
js_result = js_llm.invoke([
    SystemMessage("You are a Slipkart support classifier."),
    HumanMessage("Order ORD999 arrived broken. I want my money back."),
])

print("\n--- JSON Schema ---")
print("order_id        :", js_result["order_id"])        # always dict
print("priority        :", js_result["priority"])
print("refund_eligible :", js_result["refund_eligible"])

# --- 4. Pydantic validation fires at object creation ---

print("\n--- Pydantic validation ---")
try:
    SupportTicket(
        order_id="ORD123", days_waiting="ten",   # "ten" is not an int
        is_urgent=True, priority="high",
        tags=[], issue="test",
    )
except Exception as e:
    print("ValidationError caught:", type(e).__name__)  # expected: ValidationError
