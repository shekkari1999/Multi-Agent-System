"""Supervisor graph that routes between invoice and music sub-agents."""

from __future__ import annotations

from langgraph_supervisor import create_supervisor

from prompts.supervisor import SUPERVISOR_PROMPT
from state import AgentState


def build_supervisor(llm, music_subagent, invoice_subagent, checkpointer, store):
    """Build and compile the supervisor workflow."""
    supervisor_workflow = create_supervisor(
        agents=[invoice_subagent, music_subagent],
        output_mode="last_message",
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        state_schema=AgentState,
    )
    return supervisor_workflow.compile(
        name="supervisor_graph",
        checkpointer=checkpointer,
        store=store,
    )
