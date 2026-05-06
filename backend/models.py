"""Model resolution — Google AI Studio only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from backend.config import Settings

# Single Google AI Studio model
DEFAULT_MODELS: list[str] = [
    "google/gemini-2.0-flash",
]

# Context window sizes (tokens)
CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.0-flash-thinking-exp": 1_000_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
}

# Models that support vision
VISION_MODELS: set[str] = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking-exp",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
}


def resolve_model(spec: str, settings: Settings) -> Model:
    """Resolve a 'google/model_id' spec to a Pydantic AI Model."""
    provider = provider_from_spec(spec)
    model_id = model_id_from_spec(spec)
    if provider != "google":
        raise ValueError(
            f"Only the 'google' provider is supported. Got: {provider!r}. "
            "Use a spec like 'google/gemini-2.0-flash'."
        )
    return GoogleModel(
        model_id,
        provider=GoogleProvider(api_key=settings.gemini_api_key),
    )


def resolve_model_settings(spec: str) -> ModelSettings:
    """Get model settings with Google AI Studio thinking enabled."""
    return GoogleModelSettings(
        max_tokens=64_000,
        google_thinking_config={
            "thinking_level": "high",
            "include_thoughts": True,
        },
    )


def model_id_from_spec(spec: str) -> str:
    """Extract just the model ID from a spec."""
    parts = spec.split("/")
    return parts[1] if len(parts) >= 2 else spec


def provider_from_spec(spec: str) -> str:
    """Extract the provider from a spec."""
    return spec.split("/", 1)[0]


def effort_from_spec(spec: str) -> str | None:
    """Extract effort level from a spec like 'google/gemini-2.0-flash/high'."""
    parts = spec.split("/")
    if len(parts) >= 3 and parts[2] in ("low", "medium", "high", "max"):
        return parts[2]
    return None


def supports_vision(spec: str) -> bool:
    """Check if a model spec supports vision."""
    return model_id_from_spec(spec) in VISION_MODELS


def context_window(spec: str) -> int:
    """Get context window size for a model spec."""
    return CONTEXT_WINDOWS.get(model_id_from_spec(spec), 1_000_000)
