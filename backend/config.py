"""Pydantic Settings — credentials from .env file + environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # GZCTF
    gzctf_url: str = "http://localhost:8080"
    gzctf_user: str = "admin"
    gzctf_pass: str = "admin"
    gzctf_token: str = ""
    gzctf_game_id: int = 1

    # Google AI Studio
    gemini_api_key: str = ""

    # Infra
    sandbox_image: str = "ctf-sandbox"
    max_concurrent_challenges: int = 10
    max_attempts_per_challenge: int = 3
    container_memory_limit: str = "16g"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
