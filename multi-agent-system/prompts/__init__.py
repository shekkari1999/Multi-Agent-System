"""Prompt templates used by the system."""

from prompts.invoice import INVOICE_SUBAGENT_PROMPT
from prompts.memory import (
    CREATE_MEMORY_PROMPT,
    STRUCTURED_IDENTIFIER_PROMPT,
    VERIFY_SYSTEM_PROMPT,
)
from prompts.music import generate_music_assistant_prompt
from prompts.supervisor import SUPERVISOR_PROMPT

__all__ = [
    "CREATE_MEMORY_PROMPT",
    "INVOICE_SUBAGENT_PROMPT",
    "STRUCTURED_IDENTIFIER_PROMPT",
    "SUPERVISOR_PROMPT",
    "VERIFY_SYSTEM_PROMPT",
    "generate_music_assistant_prompt",
]
