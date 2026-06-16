from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

import asyncio
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

@tool
def get_order_status(order_id: str) -> str:
    """Returns the delivery status of a Slipkart order."""
    return "Order ORD123 is out for delivery. Expected by 6 PM today."

llm_with_tools = llm.bind_tools([get_order_status])

# --- 1. plain LLM — start / stream / end events ---
async def plain_events():
    async for event in llm.astream_events(
        [SystemMessage("You are a Slipkart support agent."),
         HumanMessage("Capital of India in one word?")],
        version="v2",   # v1 is deprecated
    ):
        kind = event["event"]
        if kind == "on_chat_model_start":
            print("[LLM started]")
        elif kind == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif kind == "on_chat_model_end":
            print("\n[LLM done]")


# --- 2. LLM + tools — tool call decision visible in on_chat_model_end ---
# NOTE: on_tool_start/on_tool_end only fire inside a full agent (Phase 2+)
# With bind_tools alone, we see the tool call DECISION not execution
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


# --- 3. asyncio.gather — run two streams concurrently ---
async def collect_tokens(question):
    tokens = []
    async for event in llm.astream_events([HumanMessage(question)], version="v2"):
        if event["event"] == "on_chat_model_stream":
            tokens.append(event["data"]["chunk"].content)
    return "".join(tokens)


async def main():
    print("── plain LLM events ──")
    await plain_events()

    print("\n── LLM + tool call decision ──")
    await tool_events()

    print("\n── concurrent streams with asyncio.gather ──")
    answer1, answer2 = await asyncio.gather(
        collect_tokens("Capital of India in one word?"),
        collect_tokens("Capital of Japan in one word?"),
    )
    print("India:", answer1)
    print("Japan:", answer2)


asyncio.run(main())
