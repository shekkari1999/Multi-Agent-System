"""Long-term memory load and save nodes."""

from __future__ import annotations

from typing import List

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from prompts.memory import CREATE_MEMORY_PROMPT
from state import AgentState


class UserProfile(BaseModel):
    """Schema for persistent customer preference memory."""

    customer_id: str = Field(description="The customer ID of the customer")
    music_preferences: List[str] = Field(
        default_factory=list,
        description="Music preferences captured from customer interactions",
    )


def format_user_memory(user_data: dict) -> str:
    """Format stored user memory into a compact context string."""
    profile = user_data.get("memory") if user_data else None
    if profile and getattr(profile, "music_preferences", None):
        return f"Music Preferences: {', '.join(profile.music_preferences)}"
    return ""


def create_load_memory_node():
    """Create node that loads user memory from the long-term store."""

    def load_memory(state: AgentState, config: RunnableConfig, store: BaseStore):
        user_id = str(state["customer_id"])
        namespace = ("memory_profile", user_id)
        existing_memory = store.get(namespace, "user_memory")

        formatted_memory = ""
        if existing_memory and existing_memory.value:
            formatted_memory = format_user_memory(existing_memory.value)

        return {"loaded_memory": formatted_memory}

    return load_memory


def create_create_memory_node(llm):
    """Create node that summarizes conversation and stores updated memory profile."""
    profile_llm = llm.with_structured_output(UserProfile)

    def create_memory(state: AgentState, config: RunnableConfig, store: BaseStore):
        user_id = str(state["customer_id"])
        namespace = ("memory_profile", user_id)

        existing_memory = store.get(namespace, "user_memory")
        formatted_memory = ""
        if existing_memory and existing_memory.value:
            formatted_memory = format_user_memory(existing_memory.value)

        system_message = SystemMessage(
            content=CREATE_MEMORY_PROMPT.format(
                conversation=state["messages"],
                memory_profile=formatted_memory or "None",
            )
        )

        updated_memory = profile_llm.invoke([system_message])
        store.put(namespace, "user_memory", {"memory": updated_memory})
        return {}

    return create_memory
