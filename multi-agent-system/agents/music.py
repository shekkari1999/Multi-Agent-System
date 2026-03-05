"""Music catalog sub-agent built from explicit LangGraph nodes."""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from prompts.music import generate_music_assistant_prompt
from state import AgentState
from tools.music import create_music_tools


def build_music_catalog_subagent(llm, db, checkpointer, store):
    """Build a ReAct-style music sub-agent from scratch."""
    music_tools = create_music_tools(db)
    llm_with_tools = llm.bind_tools(music_tools)
    music_tool_node = ToolNode(music_tools)

    def music_assistant(state: AgentState, config: RunnableConfig):
        memory = state.get("loaded_memory", "None") or "None"
        prompt = generate_music_assistant_prompt(memory)
        response = llm_with_tools.invoke([SystemMessage(prompt)] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState, config: RunnableConfig) -> str:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return "end"
        return "continue"

    workflow = StateGraph(AgentState)
    workflow.add_node("music_assistant", music_assistant)
    workflow.add_node("music_tool_node", music_tool_node)

    workflow.add_edge(START, "music_assistant")
    workflow.add_conditional_edges(
        "music_assistant",
        should_continue,
        {
            "continue": "music_tool_node",
            "end": END,
        },
    )
    workflow.add_edge("music_tool_node", "music_assistant")

    return workflow.compile(
        name="music_catalog_subagent",
        checkpointer=checkpointer,
        store=store,
    )
