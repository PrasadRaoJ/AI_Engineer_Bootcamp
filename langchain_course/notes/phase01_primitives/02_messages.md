# Messages

## Concept

Every LLM conversation is a list of **messages**. LangChain has typed message classes so the model knows who said what.

## Message types

| Class | Role | When to use |
|-------|------|-------------|
| `SystemMessage` | Sets the model's persona/rules | First message, always |
| `HumanMessage` | User input | Every user turn |
| `AIMessage` | Model response | Returned by `.invoke()`, or used to replay history |
| `ToolMessage` | Result of a tool call | After the model calls a tool |

## Flow

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  SystemMessage  │   │  HumanMessage   │   │   HumanMessage  │
│  (persona/rules)│   │  (user turn 1)  │   │  (user turn 2)  │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                      │
         └──────────┬──────────┘                      │
                    │  messages list                   │
                    ▼                                  │
         ┌─────────────────┐                          │
         │       LLM       │                          │
         └────────┬────────┘                          │
                  │                                   │
                  ▼                                   │
         ┌─────────────────┐                          │
         │    AIMessage    │ ◄── append to history    │
         │   .content      │          │               │
         └─────────────────┘          │               │
                                      ▼               ▼
                             ┌─────────────────────────────┐
                             │        history list         │
                             │  [System, Human, AI, Human] │
                             └──────────────┬──────────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │       LLM       │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │    AIMessage    │
                                   │ (remembers ctx) │
                                   └─────────────────┘
```

## Usage

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage("You are a formal customer service representative for Slipkart. Be professional and Friendly."),
    HumanMessage("My order has not arrived yet. It has been 5 days."),
]
response = llm.invoke(messages)  # returns AIMessage
```

## Building conversation history

To have a back-and-forth, append each turn manually:

```python
history = [SystemMessage("You are a helpful assistant.")]
history.append(HumanMessage("My name is JP."))
history.append(llm.invoke(history))          # AIMessage added to history
history.append(HumanMessage("What's my name?"))
response = llm.invoke(history)               # model remembers "JP"
```

## Gotchas

- Without a `SystemMessage`, the model uses its default persona.
- `AIMessage` is what `.invoke()` returns — you must append it to history yourself; LangChain does not do this automatically.
- `ToolMessage` requires a `tool_call_id` matching the model's tool call — covered in the Tools topic.
