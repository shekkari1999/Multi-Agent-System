"""Short-term and long-term memory factories."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


def create_checkpointer() -> MemorySaver:
    """Create short-term memory for thread-level state persistence."""
    return MemorySaver()


def create_in_memory_store() -> InMemoryStore:
    """Create long-term memory store for user profiles/preferences."""
    return InMemoryStore()
