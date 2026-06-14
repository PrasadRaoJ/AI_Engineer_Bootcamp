import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

llm = ChatOllama(model="llama3.2", temperature=0)

@tool
def get_order_status(order_id: str) -> str:
    """Returns the delivery status of a Slipkart order."""
    return "Order ORD123 is out for delivery. Expected by 6 PM today."

llm_with_tools = llm.bind_tools([get_order_status])

# --- 1. plain LLM — shows start/stream/end events ---
async def plain_events():
    async for event in llm.astream_events(
        [SystemMessage("You are a Slipkart support agent."),
         HumanMessage("Capital of India in one word?")],
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_start":
            print("[LLM started]")
        elif kind == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif kind == "on_chat_model_end":
            print("\n[LLM done]")

# --- 2. LLM + tools — shows tool call decision in end event ---
# NOTE: on_tool_start/on_tool_end only fire when tools run inside a full agent chain
# With bind_tools alone, the model returns a tool_call decision — visible in on_chat_model_end
async def tool_events():
    messages = [
        SystemMessage("You are a Slipkart support agent."),
        HumanMessage("What is the status of my order ORD123?"),
    ]
    async for event in llm_with_tools.astream_events(messages, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_start":
            print("[LLM started]")
        elif kind == "on_chat_model_end":
            output = event["data"]["output"]
            if output.tool_calls:
                for tc in output.tool_calls:
                    print(f"[Tool call decided: {tc['name']}({tc['args']})]")
            else:
                print(f"[LLM replied: {output.content}]")
            print("[LLM done]")

async def main():
    print("── plain LLM events ──")
    await plain_events()
    print("\n── LLM + tool call decision ──")
    await tool_events()

asyncio.run(main())
