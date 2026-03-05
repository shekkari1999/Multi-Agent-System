"""Configuration and model construction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    provider: str
    model_name: str
    temperature: float
    model_api_key: Optional[str] = None
    model_api_base: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("MODEL_PROVIDER", "openai").strip().lower(),
            model_name=os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct"),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
            model_api_key=os.getenv("MODEL_API_KEY"),
            model_api_base=os.getenv("MODEL_API_BASE"),
        )


def load_environment(dotenv_path: str = ".env") -> None:
    """Load environment variables from a .env file."""
    load_dotenv(dotenv_path=dotenv_path, override=True)


def _build_openai_chat_model(settings: Settings) -> BaseChatModel:
    kwargs = {
        "model": settings.model_name,
        "temperature": settings.temperature,
    }
    if settings.model_api_key:
        kwargs["api_key"] = settings.model_api_key
    if settings.model_api_base:
        kwargs["base_url"] = settings.model_api_base
    return ChatOpenAI(**kwargs)


def _build_litellm_chat_model(settings: Settings) -> BaseChatModel:
    from langchain_community.chat_models import ChatLiteLLM

    kwargs = {
        "model": settings.model_name,
        "temperature": settings.temperature,
    }
    if settings.model_api_key:
        kwargs["api_key"] = settings.model_api_key
    if settings.model_api_base:
        kwargs["api_base"] = settings.model_api_base

    try:
        return ChatLiteLLM(**kwargs)
    except TypeError:
        kwargs.pop("api_key", None)
        kwargs.pop("api_base", None)
        return ChatLiteLLM(**kwargs)


def build_llm(settings: Optional[Settings] = None) -> BaseChatModel:
    """Construct the primary chat model for all agents."""
    resolved_settings = settings or Settings.from_env()
    if resolved_settings.provider in {"openai", "openai_compatible"}:
        return _build_openai_chat_model(resolved_settings)
    if resolved_settings.provider == "litellm":
        return _build_litellm_chat_model(resolved_settings)

    raise ValueError(
        "Unsupported MODEL_PROVIDER. Use one of: openai, openai_compatible, litellm."
    )
