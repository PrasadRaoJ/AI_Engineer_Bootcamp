# Long-term Memory

## Concept

**Short-term memory** (checkpointer) keeps conversation history within one thread. **Long-term memory** (store) persists data **across** threads and sessions — user profiles, preferences, past order history, anything that should survive beyond a single conversation.

```
Short-term (checkpointer)         Long-term (store)
────────────────────────          ────────────────────────────
Thread A: Ravi's session  ──┐     Users namespace:
Thread B: Priya's session ──┤───► U001 → {name: Ravi, city: Nellore}
Thread C: Ravi new session─┘     U002 → {name: Priya, city: Mumbai}
                                  (survives process restart in production)
```

## Store interface

```
store.put(namespace, key, value)   → save a dict
store.get(namespace, key)          → returns Item or None
store.search(namespace_prefix)     → list of Items matching prefix
store.delete(namespace, key)       → remove an entry
```

- **namespace** — a `tuple[str, ...]` that organizes data hierarchically, e.g. `("users", "U001")`
- **key** — a plain string within that namespace, e.g. `"profile"`, `"preferences"`
- **value** — must be a `dict` — no plain strings, no lists directly

## Setup

```python
from langgraph.store.memory import InMemoryStore    # dev only
from langchain.agents import create_agent

store = InMemoryStore()

agent = create_agent(
    model=llm,
    tools=[...],
    store=store,          # long-term memory
    checkpointer=...,     # short-term memory (optional but common to use both)
)
```

## Store types

| Store | Use case |
|-------|---------|
| `InMemoryStore` | Dev / testing — lost on process restart |
| `PostgreSQL` store | Production — persists across restarts (`pip install langgraph-checkpoint-postgres`) |

## put and get

```python
# Save user profile
store.put(("users", "U001"), "profile", {"name": "Ravi", "city": "Nellore", "vip": True})

# Retrieve — returns an Item object or None if missing
item = store.get(("users", "U001"), "profile")
if item:
    print(item.value)      # {"name": "Ravi", "city": "Nellore", "vip": True}
    print(item.key)        # "profile"
    print(item.namespace)  # ("users", "U001")

# Missing key returns None — always check
missing = store.get(("users", "U999"), "profile")
# missing is None
```

## search

```python
# Find all items under a namespace prefix
results = store.search(("users",))
for r in results:
    print(r.namespace, r.key, r.value)
# ("users", "U001")  profile   {"name": "Ravi", ...}
# ("users", "U001")  preferences  {...}
# ("users", "U002")  profile   {"name": "Priya", ...}
```

## Accessing store inside a tool

Use `ToolRuntime` to read and write the store from inside a tool — same `runtime.store` object:

```python
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool
from pydantic import BaseModel

class Context(BaseModel):
    user_id: str

@tool
def get_my_profile(runtime: ToolRuntime[Context]) -> str:
    """Get the saved profile of the current user."""
    item = runtime.store.get(("users", runtime.context.user_id), "profile")
    if item is None:
        return "No profile found."
    return f"Name: {item.value['name']}, City: {item.value['city']}"

@tool
def save_city(city: str, runtime: ToolRuntime[Context]) -> str:
    """Save or update the city for the current user."""
    uid = runtime.context.user_id
    item = runtime.store.get(("users", uid), "profile")
    profile = item.value if item else {}
    profile["city"] = city
    runtime.store.put(("users", uid), "profile", profile)
    return f"City updated to {city}."
```

## Short-term vs Long-term comparison

| | Short-term (checkpointer) | Long-term (store) |
|--|--------------------------|-------------------|
| Scope | Single thread (conversation) | Cross-thread (all users, all sessions) |
| What's stored | Full message history | Structured data (profiles, preferences) |
| Access | Auto — agent loads/saves per turn | Manual — tools call `runtime.store.get/put` |
| Survives restart | ❌ `InMemorySaver` / ✅ Postgres | ❌ `InMemoryStore` / ✅ Postgres |
| Param on `create_agent` | `checkpointer=` | `store=` |

## Gotchas

- `store.put()` value must be a `dict` — plain strings or lists will fail.
- `store.get()` returns `None` if the key doesn't exist — always check before using `.value`.
- Namespace is a `tuple`, not a string — `("users", "U001")` not `"users/U001"`.
- `InMemoryStore` is shared across all threads in the same process — correct for dev, wrong for multi-process prod.
- In production, `PostgreSQL` store requires `store.setup()` to create tables before first use.
- Use both `store=` and `checkpointer=` together — they serve different purposes and don't conflict.
