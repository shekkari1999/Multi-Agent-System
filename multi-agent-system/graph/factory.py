"""Factory methods for assembling graph variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langgraph.graph import END, START, StateGraph

from agents.invoice import build_invoice_information_subagent
from agents.music import build_music_catalog_subagent
from agents.supervisor import build_supervisor
from agents.swarm import build_swarm_agents
from config import Settings, build_llm, load_environment
from database import build_sql_database
from memory import create_checkpointer, create_in_memory_store
from nodes.memory_nodes import (
    create_create_memory_node,
    create_load_memory_node,
)
from nodes.verification import (
    create_verify_info_node,
    human_input,
    should_interrupt,
)
from state import AgentState


@dataclass
class SystemResources:
    """Shared runtime resources used by all graph builders."""

    llm: object
    db: object
    checkpointer: object
    store: object


def create_resources(
    dotenv_path: str = ".env",
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> SystemResources:
    """Create model/database/memory resources for the multi-agent system."""
    load_environment(dotenv_path=dotenv_path)

    settings = Settings.from_env()
    if model_name is not None or temperature is not None:
        settings = Settings(
            provider=settings.provider,
            model_name=model_name or settings.model_name,
            temperature=temperature if temperature is not None else settings.temperature,
            model_api_key=settings.model_api_key,
            model_api_base=settings.model_api_base,
        )

    return SystemResources(
        llm=build_llm(settings),
        db=build_sql_database(),
        checkpointer=create_checkpointer(),
        store=create_in_memory_store(),
    )


def _build_subagents(resources: SystemResources):
    music_subagent = build_music_catalog_subagent(
        llm=resources.llm,
        db=resources.db,
        checkpointer=resources.checkpointer,
        store=resources.store,
    )
    invoice_subagent = build_invoice_information_subagent(
        llm=resources.llm,
        db=resources.db,
        checkpointer=resources.checkpointer,
        store=resources.store,
    )
    return music_subagent, invoice_subagent


def build_supervisor_graph(resources: SystemResources):
    """Build graph with supervisor plus its two specialized sub-agents."""
    music_subagent, invoice_subagent = _build_subagents(resources)
    return build_supervisor(
        llm=resources.llm,
        music_subagent=music_subagent,
        invoice_subagent=invoice_subagent,
        checkpointer=resources.checkpointer,
        store=resources.store,
    )


def build_verification_graph(resources: SystemResources):
    """Build graph with customer verification + supervisor routing."""
    supervisor_graph = build_supervisor_graph(resources)
    verify_info = create_verify_info_node(resources.llm, resources.db)

    workflow = StateGraph(AgentState)
    workflow.add_node("verify_info", verify_info)
    workflow.add_node("human_input", human_input)
    workflow.add_node("supervisor", supervisor_graph)

    workflow.add_edge(START, "verify_info")
    workflow.add_conditional_edges(
        "verify_info",
        should_interrupt,
        {
            "continue": "supervisor",
            "interrupt": "human_input",
        },
    )
    workflow.add_edge("human_input", "verify_info")
    workflow.add_edge("supervisor", END)

    return workflow.compile(
        name="multi_agent_verification_graph",
        checkpointer=resources.checkpointer,
        store=resources.store,
    )


def build_final_graph(resources: SystemResources):
    """Build full graph: verification -> memory load -> supervisor -> memory save."""
    supervisor_graph = build_supervisor_graph(resources)
    verify_info = create_verify_info_node(resources.llm, resources.db)
    load_memory = create_load_memory_node()
    create_memory = create_create_memory_node(resources.llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("verify_info", verify_info)
    workflow.add_node("human_input", human_input)
    workflow.add_node("load_memory", load_memory)
    workflow.add_node("supervisor", supervisor_graph)
    workflow.add_node("create_memory", create_memory)

    workflow.add_edge(START, "verify_info")
    workflow.add_conditional_edges(
        "verify_info",
        should_interrupt,
        {
            "continue": "load_memory",
            "interrupt": "human_input",
        },
    )
    workflow.add_edge("human_input", "verify_info")
    workflow.add_edge("load_memory", "supervisor")
    workflow.add_edge("supervisor", "create_memory")
    workflow.add_edge("create_memory", END)

    return workflow.compile(
        name="multi_agent_final_graph",
        checkpointer=resources.checkpointer,
        store=resources.store,
    )


def build_swarm_graph(resources: SystemResources):
    """Build swarm variant with agent handoffs instead of a central supervisor."""
    return build_swarm_agents(
        llm=resources.llm,
        db=resources.db,
        checkpointer=resources.checkpointer,
        store=resources.store,
    )
