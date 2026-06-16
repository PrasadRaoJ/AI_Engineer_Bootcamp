from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
import os

llm = init_chat_model(os.getenv("LLM_MODEL", "gemini-2.5-flash"), model_provider=os.getenv("LLM_PROVIDER", "google_genai"), temperature=0)
# groq:   LLM_PROVIDER=groq    LLM_MODEL=llama-3.3-70b-versatile
# ollama: LLM_PROVIDER=ollama  LLM_MODEL=llama3.2

# Invoke the model with a prompt and print the response
response = llm.invoke("Tell me a joke.")
print("Response:", response.content)


# Invoke the model with a prompt and print the streaming response
print("streaming response:", end="")

for chunk in llm.stream("Tell me a joke. short one."):
    print("\nChunk:", chunk.content, end="", flush=True)

print()

# batch processing
prompts = ["Tell me a joke.",
           "What is the capital of France?",
           "Who is the president of India?"]

responses = llm.batch(prompts)

print("Batch responses:\n")

for r in responses:
    print('\n', r.content)


print("\n batch as complete response:")

for idx, r in llm.batch_as_completed(prompts):
    print(f"\nResponse for prompt {idx}:\n{r.content}")
