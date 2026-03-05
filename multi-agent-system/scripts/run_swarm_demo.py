"""Run the swarm variant demo from the command line."""

from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage

from graph.factory import build_swarm_graph, create_resources


def _print_messages(messages) -> None:
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", str(message))
        print(f"[{role}] {content}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the swarm handoff demo.")
    parser.add_argument(
        "--question",
        required=True,
        help="User question to route through the swarm.",
    )
    parser.add_argument(
        "--dotenv",
        default=".env",
        help="Path to .env file.",
    )
    args = parser.parse_args()

    resources = create_resources(dotenv_path=args.dotenv)
    graph = build_swarm_graph(resources)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke({"messages": [HumanMessage(content=args.question)]}, config=config)
    _print_messages(result.get("messages", []))


if __name__ == "__main__":
    main()
