"""Model access for the agent crew.

Providers, picked by ``CUTAWAY_LLM_PROVIDER`` (default ``auto``):

* ``azure``   Microsoft Foundry / Azure OpenAI deployment (AZURE_OPENAI_ENDPOINT + key)
* ``github``  GitHub Models, free and rate limited (GITHUB_TOKEN)
* ``offline`` no model calls; agents fall back to deterministic logic

``auto`` uses azure when configured, then github, then offline. Every model answer is
cached on disk, so the same request never costs a second call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / ".cache" / "llm"
GITHUB_MODELS_URL = "https://models.github.ai/inference"

T = TypeVar("T", bound=BaseModel)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str

    @classmethod
    def from_env(cls) -> "LLMSettings":
        load_dotenv()
        provider = os.getenv("CUTAWAY_LLM_PROVIDER", "auto").lower()
        if provider == "auto":
            if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
                provider = "azure"
            elif os.getenv("GITHUB_TOKEN"):
                provider = "github"
            else:
                provider = "offline"
        if provider == "azure":
            model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-5.4-mini")
        elif provider == "github":
            model = os.getenv("GITHUB_MODELS_MODEL", "openai/gpt-4.1-mini")
        else:
            model = "offline"
        return cls(provider, model)

    @property
    def online(self) -> bool:
        return self.provider != "offline"


def _chat_client(settings: LLMSettings):
    # Corporate/university networks often intercept TLS; trust the OS certificate store.
    import truststore

    truststore.inject_into_ssl()
    from agent_framework.openai import OpenAIChatCompletionClient

    if settings.provider == "azure":
        return OpenAIChatCompletionClient(
            model=settings.model,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
    return OpenAIChatCompletionClient(
        model=settings.model, api_key=os.environ["GITHUB_TOKEN"], base_url=GITHUB_MODELS_URL,
    )


class LLM:
    """Runs Microsoft Agent Framework agents with structured output and a disk cache."""

    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings.from_env()
        self._client = None
        self.calls = 0
        self.cache_hits = 0

    @property
    def online(self) -> bool:
        return self.settings.online

    def _cache_path(self, name: str, instructions: str, prompt: str, schema: type[BaseModel]) -> Path:
        key = json.dumps([self.settings.provider, self.settings.model, name, instructions, prompt,
                          schema.model_json_schema()], sort_keys=True)
        return CACHE_DIR / f"{name}-{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    async def run(self, name: str, instructions: str, prompt: str, schema: type[T]) -> T:
        """Run one agent turn and parse its answer into ``schema``."""
        if not self.online:
            raise RuntimeError("LLM is offline")
        path = self._cache_path(name, instructions, prompt, schema)
        if path.exists():
            self.cache_hits += 1
            return schema.model_validate_json(path.read_text(encoding="utf-8"))

        from agent_framework import Agent

        if self._client is None:
            self._client = _chat_client(self.settings)
        agent = Agent(self._client, instructions, name=name)
        options: dict = {"response_format": schema}
        if "gpt-5" not in self.settings.model and "/o" not in self.settings.model:
            options["temperature"] = 0.4  # reasoning models reject temperature
        response = await agent.run(prompt, options=options)
        self.calls += 1
        value = response.value
        if value is None:
            value = schema.model_validate_json(response.text)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(value.model_dump_json(), encoding="utf-8")
        log.info("agent %s answered via %s/%s", name, self.settings.provider, self.settings.model)
        return value

    def run_sync(self, name: str, instructions: str, prompt: str, schema: type[T]) -> T:
        return asyncio.run(self.run(name, instructions, prompt, schema))
