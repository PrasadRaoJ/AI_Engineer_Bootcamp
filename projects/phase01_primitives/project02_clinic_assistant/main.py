from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from schemas import AppointmentRequest
from tools import ALL_TOOLS, TOOL_MAP

llm = ChatOllama(model="llama3.2", temperature=0)

SYSTEM = SystemMessage(
    "You are a formal clinic receptionist at Yapollo Clinic, Nellore. "
    "Be professional, warm, and helpful. Always confirm appointment details clearly. "
    "Today's date is 2026-06-14. Always use year 2026 for any dates mentioned."
)


def run(user_input: str) -> None:
    print(f"\n{'─'*50}")
    print(f"Patient: {user_input}")
    print(f"{'─'*50}")

    # step 1 — classify the request
    request: AppointmentRequest = llm.with_structured_output(AppointmentRequest).invoke([
        SYSTEM,
        HumanMessage(user_input),
    ])
    print(f"[Request] patient={request.patient_name} | urgency={request.urgency} | action={request.action_needed}")

    # step 2 — tool call
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    history = [SYSTEM, HumanMessage(user_input)]
    response = llm_with_tools.invoke(history)

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"[Tool] {tc['name']}({tc['args']})")
            result = TOOL_MAP[tc["name"]].invoke(tc["args"])
            print(f"[Result] {result}")
            history.append(response)
            history.append(ToolMessage(result, tool_call_id=tc["id"]))

    # step 3 — stream final reply
    print("\nReceptionist: ", end="")
    for chunk in llm_with_tools.stream(history):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    run("Hi, I am Priya Mehta. I need to see a general physician on 15th June.")
    run("Can you tell me about Dr. Anusha?")
    run("Please cancel my appointment APT001.")
