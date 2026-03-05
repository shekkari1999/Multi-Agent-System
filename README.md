# Multi-Agent System

A modular LangGraph application for customer support in a digital music store domain.

It includes:
- identity verification with human-in-the-loop interruption
- supervisor-based routing across specialized agents
- long-term user memory for music preferences
- optional swarm handoff mode
- LangSmith evaluation workflows

## Architecture

The runtime flow for the main graph is:
1. Verify customer identity
2. Load long-term memory
3. Route the request through the supervisor
4. Save updated memory
5. Return final answer

Domain agents:
- `music_catalog_subagent`
- `invoice_information_subagent`

## Project Structure

```text
multi-agent-system/
├── .env.example
├── Makefile
├── pyproject.toml
├── requirements.txt
├── README.md
├── agents/
├── graph/
├── nodes/
├── prompts/
├── scripts/
├── tools/
├── config.py
├── database.py
├── memory.py
├── state.py
└── visualization.py
```

## Requirements

- Python 3.10+
- API key for your selected model provider
- Optional: LangSmith API key for tracing and evaluations

## Setup

```bash
cd multi-agent-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Configuration

Core settings in `.env`:
- `MODEL_PROVIDER`: `openai`, `openai_compatible`, or `litellm`
- `MODEL_NAME`
- `MODEL_TEMPERATURE`
- `MODEL_API_KEY` and `MODEL_API_BASE` when needed

Provider-specific keys:
- `OPENAI_API_KEY` for OpenAI mode
- `XAI_API_KEY` for Grok via LiteLLM

LangSmith:
- `LANGSMITH_API_KEY`
- `LANGSMITH_TRACING=true`
- `LANGSMITH_PROJECT=multi-agent-system`

Example OpenAI config:

```env
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
MODEL_TEMPERATURE=0
OPENAI_API_KEY=...
```

Example Grok via LiteLLM:

```env
MODEL_PROVIDER=litellm
MODEL_NAME=xai/grok-2-latest
MODEL_TEMPERATURE=0
XAI_API_KEY=...
```

## Run

Main graph:

```bash
python -m scripts.run_final_graph \
  --question "My customer ID is 1. What was my most recent purchase?"
```

Swarm mode:

```bash
python -m scripts.run_swarm_demo \
  --question "Do you have albums by the Rolling Stones?"
```

Evaluations:

```bash
python -m scripts.run_evals --mode all
```

Modes:
- `all`
- `final`
- `routing`
- `trajectory`

## Developer Commands

```bash
make install
make run QUESTION="My customer ID is 1. What albums do you have by U2?"
make swarm QUESTION="Do you have songs by AC/DC?"
make eval MODE=final
```

## Notes

- Chinook data is loaded into an in-memory SQLite database at runtime.
- Graph visualization helper is available in `visualization.py`.
