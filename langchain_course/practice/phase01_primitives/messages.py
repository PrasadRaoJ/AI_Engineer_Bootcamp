from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, ChatMessage
import os

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of Andhra Pradesh?"),
]

tool_messages = ToolMessage(content="This is a tool message.",
                            tool_call_id="tool_123",
                            artifact={"eta": "Today", "carrier": "ClueDart"})

print("\nTool Message:\n", tool_messages.content)
print("\nTool Call ID:\n", tool_messages.tool_call_id)
print("\nArtifact:\n", tool_messages.artifact)
