from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

llm = ChatOllama(model="llama3.2", temperature=0)

# --- 1. Pydantic schema (recommended) ---
class SupportTicket(BaseModel):
    order_id: str                                     # plain string
    days_waiting: int                                 # integer
    is_urgent: bool                                   # boolean
    priority: Literal["low", "medium", "high"]        # constrained to fixed values
    tags: List[str]                                   # list of strings
    notes: Optional[str] = None                       # nullable

structured_llm = llm.with_structured_output(SupportTicket)

result = structured_llm.invoke([
    SystemMessage("You are a Slipkart support ticket classifier. Extract structured details from the customer message."),
    HumanMessage("My order ORD123 hasn't arrived in 10 days and I need it urgently for a wedding!"),
])

print("--- Pydantic ---")
print("order_id    :", result.order_id)      # expected: ORD123
print("days_waiting:", result.days_waiting)  # expected: 10
print("is_urgent   :", result.is_urgent)     # expected: True
print("priority    :", result.priority)      # expected: high
print("tags        :", result.tags)
print("notes       :", result.notes)

# --- 2. TypedDict schema (lighter, returns dict) ---
from typing import TypedDict

class SimpleTicket(TypedDict):
    order_id: str
    priority: str

simple_llm = llm.with_structured_output(SimpleTicket)

simple = simple_llm.invoke([
    SystemMessage("You are a Slipkart support classifier."),
    HumanMessage("Order ORD456 is delayed."),
])

print("\n--- TypedDict ---")
print("order_id :", simple["order_id"])   # dict access, not attribute
print("priority :", simple["priority"])

# NOTE: Dataclass is NOT supported by Ollama's structured output — skip it.

# --- 3. JSON Schema (raw dict — no class needed) ---
json_schema = {
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
print("order_id        :", js_result["order_id"])        # always dict access
print("priority        :", js_result["priority"])
print("refund_eligible :", js_result["refund_eligible"])
