"""Run the full multi-agent graph from the command line."""

from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph.factory import build_final_graph, create_resources


def _print_messages(messages) -> None:
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", str(message))
        print(f"[{role}] {content}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the modular multi-agent graph.")
    parser.add_argument(
        "--question",
        required=True,
        help="User question to send into the graph.",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Optional resume input for human-in-the-loop interruption.",
    )
    parser.add_argument(
        "--user-id",
        default="10",
        help="Configurable user_id for thread config metadata.",
    )
    parser.add_argument(
        "--thread-id",
        default="",
        help="Optional thread ID. If omitted, a new UUID is generated.",
    )
    parser.add_argument(
        "--dotenv",
        default=".env",
        help="Path to .env file.",
    )
    args = parser.parse_args()

    resources = create_resources(dotenv_path=args.dotenv)
    graph = build_final_graph(resources)

    thread_id = args.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id, "user_id": args.user_id}}

    result = graph.invoke({"messages": [HumanMessage(content=args.question)]}, config=config)
    _print_messages(result.get("messages", []))

    if "__interrupt__" in result:
        if not args.resume:
            print("Graph interrupted. Pass --resume to continue this thread.")
            return

        resumed = graph.invoke(Command(resume=args.resume), config=config)
        _print_messages(resumed.get("messages", []))


if __name__ == "__main__":
    main()
