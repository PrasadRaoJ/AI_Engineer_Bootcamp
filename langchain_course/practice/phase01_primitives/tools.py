from dotenv import load_dotenv
import os
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

llm = init_chat_model(os.getenv("LLM_MODEL", "llama3.2"), model_provider=os.getenv("LLM_PROVIDER", "ollama"), temperature=0)

@tool
def get_name() -> str:
    """Returns the name of the current user."""
    return "Your name is JP!"

llm_with_tools = llm.bind_tools([get_name])

# step 1 — ask LLM
messages = [HumanMessage("What is my name?")]
response = llm_with_tools.invoke(messages)
print("LLM decision  :", response.tool_calls)  # LLM says: call get_name

# step 2 — execute the tool
messages.append(response)                                          # add AIMessage
tool_result = get_name.invoke(response.tool_calls[0]["args"])      # run the tool
print("Tool output   :", tool_result)
messages.append(ToolMessage(tool_result, tool_call_id=response.tool_calls[0]["id"]))

# step 3 — LLM reads tool output and gives final reply to user
final = llm_with_tools.invoke(messages)
print("Final reply   :", final.content)

