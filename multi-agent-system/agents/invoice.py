"""Invoice sub-agent built with LangGraph prebuilt ReAct helper."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from prompts.invoice import INVOICE_SUBAGENT_PROMPT
from state import AgentState
from tools.invoice import create_invoice_tools


def build_invoice_information_subagent(llm, db, checkpointer, store):
    """Build invoice information sub-agent using create_react_agent."""
    invoice_tools = create_invoice_tools(db)
    return create_react_agent(
        llm,
        tools=invoice_tools,
        name="invoice_information_subagent",
        prompt=INVOICE_SUBAGENT_PROMPT,
        state_schema=AgentState,
        checkpointer=checkpointer,
        store=store,
    )
