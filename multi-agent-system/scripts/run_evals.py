"""Run LangSmith evaluation suites for the modular multi-agent system."""

from __future__ import annotations

import argparse
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from graph.factory import (
    build_final_graph,
    build_supervisor_graph,
    create_resources,
)

FINAL_DATASET = "LangGraph 101 Multi-Agent: Final Response"
ROUTING_DATASET = "LangGraph 101 Multi-Agent: Single-Step"
TRAJECTORY_DATASET = "LangGraph 101 Multi-Agent: Trajectory Eval"

FINAL_EXAMPLES = [
    {
        "question": (
            "My name is Aaron Mitchell. My number associated with my account is +1 (204) "
            "452-6452. I am trying to find the invoice number for my most recent song "
            "purchase. Could you help me with it?"
        ),
        "response": "The Invoice ID of your most recent purchase was 342.",
    },
    {
        "question": "I'd like a refund.",
        "response": (
            "I need additional information to help you with the refund. Could you please "
            "provide your customer identifier so that we can fetch your purchase history?"
        ),
    },
    {
        "question": "Who recorded Wish You Were Here again?",
        "response": "Wish You Were Here is an album by Pink Floyd",
    },
    {
        "question": "What albums do you have by Coldplay?",
        "response": "There are no Coldplay albums available in our catalog at the moment.",
    },
]

ROUTING_EXAMPLES = [
    {
        "messages": (
            "My customer ID is 1. What's my most recent purchase? and What albums does "
            "the catalog have by U2?"
        ),
        "route": "transfer_to_invoice_information_subagent",
    },
    {
        "messages": "What songs do you have by U2?",
        "route": "transfer_to_music_catalog_subagent",
    },
    {
        "messages": (
            "My name is Aaron Mitchell. My number associated with my account is +1 (204) "
            "452-6452. I am trying to find the invoice number for my most recent song "
            "purchase. Could you help me with it?"
        ),
        "route": "transfer_to_invoice_information_subagent",
    },
    {
        "messages": "Who recorded Wish You Were Here again? What other albums by them do you have?",
        "route": "transfer_to_music_catalog_subagent",
    },
]

TRAJECTORY_EXAMPLES = [
    {
        "question": (
            "My customer ID is 1. What's my most recent purchase? and What albums "
            "does the catalog have by U2?"
        ),
        "trajectory": ["verify_info", "load_memory", "supervisor", "create_memory"],
    },
    {
        "question": "What songs do you have by U2?",
        "trajectory": [
            "verify_info",
            "human_input",
            "verify_info",
            "load_memory",
            "supervisor",
            "create_memory",
        ],
    },
    {
        "question": (
            "My name is Aaron Mitchell. My number associated with my account is +1 (204) "
            "452-6452. I am trying to find the invoice number for my most recent song "
            "purchase. Could you help me with it?"
        ),
        "trajectory": ["verify_info", "load_memory", "supervisor", "create_memory"],
    },
    {
        "question": "Who recorded Wish You Were Here again? What other albums by them do you have?",
        "trajectory": [
            "verify_info",
            "human_input",
            "verify_info",
            "load_memory",
            "supervisor",
            "create_memory",
        ],
    },
]


def ensure_dataset(client: Client, dataset_name: str, inputs: list[dict], outputs: list[dict]) -> None:
    """Create dataset only if it does not already exist."""
    if client.has_dataset(dataset_name=dataset_name):
        return

    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)


class Grade(TypedDict):
    """Schema for custom LLM-as-judge output."""

    reasoning: Annotated[str, ..., "Reasoning for whether the student response is correct"]
    is_correct: Annotated[bool, ..., "True when student response is correct"]


def route_correct(outputs: dict, reference_outputs: dict) -> bool:
    """Route evaluator for supervisor single-step dataset."""
    return outputs["route"] == reference_outputs["route"]


def evaluate_exact_match(outputs: dict, reference_outputs: dict) -> dict:
    """Return exact-match score for trajectory evaluation."""
    return {
        "key": "exact_match",
        "score": outputs["trajectory"] == reference_outputs["trajectory"],
    }


def evaluate_extra_steps(outputs: dict, reference_outputs: dict) -> dict:
    """Count unmatched steps between predicted and expected trajectories."""
    i = j = 0
    unmatched_steps = 0

    expected = reference_outputs["trajectory"]
    predicted = outputs["trajectory"]

    while i < len(expected) and j < len(predicted):
        if expected[i] == predicted[j]:
            i += 1
        else:
            unmatched_steps += 1
        j += 1

    unmatched_steps += len(predicted) - j
    return {"key": "unmatched_steps", "score": unmatched_steps}


async def run(mode: str, dotenv_path: str) -> None:
    resources = create_resources(dotenv_path=dotenv_path)
    final_graph = build_final_graph(resources)
    supervisor_graph = build_supervisor_graph(resources)

    client = Client()

    ensure_dataset(
        client,
        FINAL_DATASET,
        inputs=[{"question": ex["question"]} for ex in FINAL_EXAMPLES],
        outputs=[{"response": ex["response"]} for ex in FINAL_EXAMPLES],
    )
    ensure_dataset(
        client,
        ROUTING_DATASET,
        inputs=[{"messages": ex["messages"]} for ex in ROUTING_EXAMPLES],
        outputs=[{"route": ex["route"]} for ex in ROUTING_EXAMPLES],
    )
    ensure_dataset(
        client,
        TRAJECTORY_DATASET,
        inputs=[{"question": ex["question"]} for ex in TRAJECTORY_EXAMPLES],
        outputs=[{"trajectory": ex["trajectory"]} for ex in TRAJECTORY_EXAMPLES],
    )

    async def run_final_response(inputs: dict) -> dict:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id, "user_id": "10"}}

        result = await final_graph.ainvoke(
            {"messages": [{"role": "user", "content": inputs["question"]}]},
            config=config,
        )

        if "__interrupt__" in result:
            result = await final_graph.ainvoke(
                Command(resume="My customer ID is 10"),
                config=config,
            )

        return {"response": result["messages"][-1].content}

    grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH RESPONSE, and the STUDENT RESPONSE.

Grade criteria:
1. Grade based only on factual accuracy against the ground truth.
2. Student response must not contain conflicting statements.
3. Extra information is allowed if it is factually accurate.

Correctness is True only if all criteria are met."""

    grader_llm = resources.llm.with_structured_output(
        Grade, method="json_schema", strict=True
    )

    async def final_answer_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        prompt = (
            f"QUESTION: {inputs['question']}\n"
            f"GROUND TRUTH RESPONSE: {reference_outputs['response']}\n"
            f"STUDENT RESPONSE: {outputs['response']}"
        )
        grade = await grader_llm.ainvoke(
            [
                {"role": "system", "content": grader_instructions},
                {"role": "user", "content": prompt},
            ]
        )
        return grade["is_correct"]

    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=resources.llm,
    )

    async def run_supervisor_routing(inputs: dict) -> dict:
        result = await supervisor_graph.ainvoke(
            {"messages": [HumanMessage(content=inputs["messages"])]},
            interrupt_before=["music_catalog_subagent", "invoice_information_subagent"],
            config={"configurable": {"thread_id": str(uuid.uuid4()), "user_id": "10"}},
        )
        route = getattr(result["messages"][-1], "name", "")
        return {"route": route}

    async def run_trajectory(inputs: dict) -> dict:
        trajectory = []
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id, "user_id": "10"}}

        async for chunk in final_graph.astream(
            {"messages": [{"role": "user", "content": inputs["question"]}]},
            config=config,
            stream_mode="debug",
        ):
            if chunk.get("type") == "task":
                trajectory.append(chunk["payload"]["name"])

        graph_state = await final_graph.aget_state(config)
        if graph_state.next:
            async for chunk in final_graph.astream(
                Command(resume="My customer ID is 10"),
                config=config,
                stream_mode="debug",
            ):
                if chunk.get("type") == "task":
                    trajectory.append(chunk["payload"]["name"])

        return {"trajectory": trajectory}

    if mode in ("all", "final"):
        await client.aevaluate(
            run_final_response,
            data=FINAL_DATASET,
            evaluators=[final_answer_correct, correctness_evaluator],
            experiment_prefix="modular-agent-final-response",
            num_repetitions=1,
            max_concurrency=4,
        )
        print(f"Completed evaluation: {FINAL_DATASET}")

    if mode in ("all", "routing"):
        await client.aevaluate(
            run_supervisor_routing,
            data=ROUTING_DATASET,
            evaluators=[route_correct],
            experiment_prefix="modular-agent-routing",
            max_concurrency=4,
        )
        print(f"Completed evaluation: {ROUTING_DATASET}")

    if mode in ("all", "trajectory"):
        await client.aevaluate(
            run_trajectory,
            data=TRAJECTORY_DATASET,
            evaluators=[evaluate_extra_steps, evaluate_exact_match],
            experiment_prefix="modular-agent-trajectory",
            num_repetitions=1,
            max_concurrency=4,
        )
        print(f"Completed evaluation: {TRAJECTORY_DATASET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangSmith evaluation suites.")
    parser.add_argument(
        "--mode",
        choices=["all", "final", "routing", "trajectory"],
        default="all",
        help="Which evaluation suite to run.",
    )
    parser.add_argument(
        "--dotenv",
        default=".env",
        help="Path to .env file.",
    )
    args = parser.parse_args()

    import asyncio

    asyncio.run(run(args.mode, args.dotenv))


if __name__ == "__main__":
    main()
