"""Shared LangGraph state schema for the multi-agent system."""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps


class AgentState(TypedDict, total=False):
    """State passed across all nodes in the graph."""

    customer_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: str
    remaining_steps: RemainingSteps
