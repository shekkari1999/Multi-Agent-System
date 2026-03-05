"""Swarm-mode agents with explicit handoff tools."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

from prompts.invoice import INVOICE_SUBAGENT_PROMPT
from prompts.music import generate_music_assistant_prompt
from state import AgentState
from tools.invoice import create_invoice_tools
from tools.music import create_music_tools


def build_swarm_agents(llm, db, checkpointer, store):
    """Build swarm agents that can hand off tasks to each other."""
    transfer_to_invoice = create_handoff_tool(
        agent_name="invoice_information_agent_with_handoff",
        description=(
            "Transfer user to the invoice information agent for invoice and purchase questions."
        ),
    )
    transfer_to_music = create_handoff_tool(
        agent_name="music_catalog_agent_with_handoff",
        description=(
            "Transfer user to the music catalog agent for songs, albums, and artist queries."
        ),
    )

    invoice_tools = [transfer_to_music] + create_invoice_tools(db)
    music_tools = [transfer_to_invoice] + create_music_tools(db)

    invoice_agent = create_react_agent(
        llm,
        invoice_tools,
        prompt=INVOICE_SUBAGENT_PROMPT,
        name="invoice_information_agent_with_handoff",
        state_schema=AgentState,
    )

    music_agent = create_react_agent(
        llm,
        music_tools,
        prompt=generate_music_assistant_prompt(),
        name="music_catalog_agent_with_handoff",
        state_schema=AgentState,
    )

    swarm_workflow = create_swarm(
        agents=[invoice_agent, music_agent],
        default_active_agent="invoice_information_agent_with_handoff",
    )

    return swarm_workflow.compile(checkpointer=checkpointer, store=store)
