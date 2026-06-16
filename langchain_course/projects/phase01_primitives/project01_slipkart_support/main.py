from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from schemas import SupportTicket
from tools import ALL_TOOLS, TOOL_MAP

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

SYSTEM = SystemMessage(
    "You are a formal customer support agent for Slipkart. "
    "Be professional, concise, and empathetic."
)


def run(user_input: str) -> None:
    print(f"\n{'─'*50}")
    print(f"Customer: {user_input}")
    print(f"{'─'*50}")

    # step 1 — classify the ticket
    ticket: SupportTicket = llm.with_structured_output(SupportTicket).invoke([
        SYSTEM,
        HumanMessage(user_input),
    ])
    print(f"[Ticket] order={ticket.order_id} | priority={ticket.priority} | action={ticket.action_needed}")

    # step 2 — tool call
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    history = [SYSTEM, HumanMessage(user_input)]
    response = llm_with_tools.invoke(history)

    if response.tool_calls:
        history.append(response)                                    # AIMessage added ONCE
        for tc in response.tool_calls:
            print(f"[Tool] {tc['name']}({tc['args']})")
            result = TOOL_MAP[tc["name"]].invoke(tc["args"])
            print(f"[Result] {result}")
            history.append(ToolMessage(result, tool_call_id=tc["id"]))

    # step 3 — stream final reply
    print("\nAgent: ", end="")
    for chunk in llm_with_tools.stream(history):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    run("My order ORD789 is delayed. Where is it?")
    run("Please cancel my order ORD123. I no longer need it.")
    run("I received a damaged item in order ORD456. I want a refund.")
