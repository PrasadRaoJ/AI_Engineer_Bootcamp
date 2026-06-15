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

## Key fields per message type

### AIMessage

```python
response = llm.invoke("What is 2+2?")

response.content          # the reply text
response.tool_calls       # list of tool calls (if any)
response.usage_metadata   # token counts: {"input_tokens": 32, "output_tokens": 9, "total_tokens": 41}
```

`usage_metadata` is available on every AIMessage — useful for tracking token cost per call.

### ToolMessage

```python
from langchain_core.messages import ToolMessage

ToolMessage(
    content="Order ORD123 is out for delivery.",   # text sent back to the model
    tool_call_id="abc123",                          # must match AIMessage tool call id
    artifact={"raw_data": {...}},                   # supplementary data NOT sent to model
)
```

- `content` — what the model sees
- `tool_call_id` — must exactly match the id from the AIMessage tool call
- `artifact` — optional raw data (e.g. full API response) stored for your own use, invisible to the LLM

### HumanMessage

```python
HumanMessage("Hello")                  # plain text
HumanMessage("Hello", name="JP")       # name field — behavior varies by provider
```

The `name` field is optional. Some providers use it for user identification; others ignore it.

## Gotchas

- Without a `SystemMessage`, the model uses its default persona.
- `AIMessage` is what `.invoke()` returns — you must append it to history yourself; LangChain does not do this automatically.
- `ToolMessage` requires `tool_call_id` matching the model's tool call exactly — wrong id = model ignores the result.
- `usage_metadata` is a dict, not an object — access as `response.usage_metadata["input_tokens"]`.
- `artifact` on `ToolMessage` is invisible to the LLM — only use it to store data for your own downstream code.
