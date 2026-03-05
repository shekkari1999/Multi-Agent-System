"""Customer verification and human-in-the-loop nodes."""

from __future__ import annotations

import ast
from typing import Optional

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from prompts.memory import (
    STRUCTURED_IDENTIFIER_PROMPT,
    VERIFY_SYSTEM_PROMPT,
)
from state import AgentState


class UserInput(BaseModel):
    """Structured identifier extraction schema."""

    identifier: str = Field(
        default="",
        description="Customer identifier. Can be customer ID, email, or phone number.",
    )


def _parse_customer_id(raw_result: str) -> Optional[str]:
    if not raw_result:
        return None

    parsed = ast.literal_eval(raw_result)
    if not parsed:
        return None

    first_item = parsed[0]
    if isinstance(first_item, dict):
        value = first_item.get("CustomerId")
    elif isinstance(first_item, (list, tuple)):
        value = first_item[0]
    else:
        value = first_item

    if value is None:
        return None
    return str(value)


def get_customer_id_from_identifier(db: SQLDatabase, identifier: str) -> Optional[str]:
    """Resolve customer ID using direct ID, email, or phone number."""
    if not identifier:
        return None

    identifier = identifier.strip()
    if identifier.isdigit():
        return str(int(identifier))

    if "@" in identifier:
        escaped = identifier.replace("'", "''")
        result = db.run(f"SELECT CustomerId FROM Customer WHERE Email = '{escaped}';")
        return _parse_customer_id(result)

    escaped = identifier.replace("'", "''")
    result = db.run(f"SELECT CustomerId FROM Customer WHERE Phone = '{escaped}';")
    return _parse_customer_id(result)


def create_verify_info_node(llm, db: SQLDatabase):
    """Create a verification node closure that parses and validates customer identity."""
    structured_llm = llm.with_structured_output(schema=UserInput)

    def verify_info(state: AgentState, config: RunnableConfig):
        if state.get("customer_id") is not None:
            return {}

        user_input = state["messages"][-1]
        parsed_info = structured_llm.invoke(
            [SystemMessage(content=STRUCTURED_IDENTIFIER_PROMPT), user_input]
        )
        identifier = (parsed_info.identifier or "").strip()

        customer_id = get_customer_id_from_identifier(db, identifier)
        if customer_id:
            return {
                "customer_id": customer_id,
                "messages": [
                    SystemMessage(
                        content=(
                            "Thanks for sharing your details. "
                            f"Your account is verified with customer ID {customer_id}."
                        )
                    )
                ],
            }

        response = llm.invoke([SystemMessage(content=VERIFY_SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    return verify_info


def human_input(state: AgentState, config: RunnableConfig):
    """Interrupt graph execution until human input is provided."""
    user_input = interrupt("Please provide your customer ID, email, or phone number.")
    return {"messages": [user_input]}


def should_interrupt(state: AgentState, config: RunnableConfig) -> str:
    """Route based on whether customer identity is already verified."""
    if state.get("customer_id") is not None:
        return "continue"
    return "interrupt"
