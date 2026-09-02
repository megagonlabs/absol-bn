"""LLM plumbing shared by every LLM-backed model: provider client, response
cache, usage accounting, and call budget."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Mapping, Optional

from openai import OpenAI

from . import prompts
from .llm_cache import LlmCache, make_key


ChatMessage = Dict[str, str]

# ``provider`` only selects which API key and base URL the OpenAI-compatible
# client targets. Keys are read from the environment; the repo convention is a
# gitignored ``.env`` of ``export KEY=value`` lines, sourced before training —
# plain ``KEY=value`` sets a shell variable the Python process never inherits.
PROVIDER_CONFIG = {
    "openai":    {"api_key_env": "OPENAI_API_KEY",    "base_url": "https://api.openai.com/v1"},
    "fireworks": {"api_key_env": "FIREWORKS_API_KEY", "base_url": "https://api.fireworks.ai/inference/v1"},
    "gemini":    {"api_key_env": "GEMINI_API_KEY",    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
}


def build_llm_client(llm_provider: str) -> OpenAI:
    """Build an OpenAI client for the given provider, failing early if unusable."""
    if llm_provider not in PROVIDER_CONFIG:
        raise ValueError(
            f"Unknown llm_provider '{llm_provider}'. Expected one of {sorted(PROVIDER_CONFIG)}."
        )
    cfg = PROVIDER_CONFIG[llm_provider]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        # Fail here rather than deep inside the first request, where the provider
        # reports it as a generic 401 hours into a run.
        raise RuntimeError(
            f"llm_provider='{llm_provider}' requires {cfg['api_key_env']}, which is not set. "
            f"Add `export {cfg['api_key_env']}=...` to .env and run `source .env` first."
        )
    return OpenAI(
        api_key=api_key,
        base_url=cfg["base_url"],
        timeout=180.0,
        max_retries=5,
    )


class LlmCallBudgetExceeded(RuntimeError):
    """Raised when a session exceeds its configured uncached-call budget."""


def _new_usage_bucket() -> Dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "truncated": 0,
        "call_records": [],
    }


class LLMSession:
    """One model's connection to an OpenAI-compatible chat-completions endpoint.

    ``cache_path=None`` disables response caching. ``max_calls`` caps *uncached*
    calls and raises :class:`LlmCallBudgetExceeded` on overrun.
    """

    def __init__(
        self,
        settings: Any,
        *,
        cache_path: Optional[str] = None,
        max_calls: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.model_name = settings.model_name
        self.provider = settings.provider
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.reasoning_effort = settings.reasoning_effort
        self.variable_descriptions: Dict[str, str] = dict(settings.variable_descriptions)
        self.cache_path = cache_path
        self.max_calls = max_calls
        self.verbose = verbose
        self.usage_details: Dict[str, Dict[str, Any]] = {}
        self.calls_made = 0
        self._open()

    def _open(self) -> None:
        self._client = build_llm_client(self.provider)
        self._cache = LlmCache(self.cache_path) if self.cache_path else None
        self._lock = threading.Lock()

    # The client, cache handle, and lock cannot be pickled. Dropping and
    # rebuilding them here keeps every owning model picklable as-is.

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        for field in ("_client", "_cache", "_lock"):
            state.pop(field, None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()

    # ---------------- prompt helpers ----------------

    def describe(self, variables: List[str]) -> str:
        """Format the shared variable-description block used across prompts.

        Sorted rather than set-iteration order: this block goes into the prompt
        and therefore into the cache key, and set order varies with per-process
        string-hash randomization, so unsorted output missed the cache on every
        fresh process.
        """
        if not self.variable_descriptions:
            return ""

        descriptions = [
            f"- {variable}: {self.variable_descriptions[variable]}"
            for variable in sorted(set(variables))
            if variable in self.variable_descriptions
        ]
        if descriptions:
            return "\nVariable descriptions:\n" + "\n".join(descriptions) + "\n\n"
        return ""

    # ---------------- usage accounting ----------------

    def _record_usage(
        self,
        bucket: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cached_prompt_tokens: int = 0,
        truncated: bool = False,
    ) -> None:
        with self._lock:
            stats = self.usage_details.setdefault(bucket, _new_usage_bucket())
            stats.setdefault("truncated", 0)
            stats["calls"] += 1
            stats["prompt_tokens"] += prompt_tokens
            stats["cached_prompt_tokens"] += cached_prompt_tokens
            stats["completion_tokens"] += completion_tokens
            stats["total_tokens"] += total_tokens
            if truncated:
                stats["truncated"] += 1
            # Per-call record so cost can be priced correctly under tiered pricing.
            stats["call_records"].append({
                "prompt_tokens": prompt_tokens,
                "cached_prompt_tokens": cached_prompt_tokens,
                "completion_tokens": completion_tokens,
            })

    # ---------------- requests ----------------

    def send(
        self,
        message: str,
        operation_type: str = "general",
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a single-turn system+user exchange and return the reply text.

        ``system_prompt``, when provided, overrides the per-operation system
        prompt looked up by ``operation_type``. Use it for op-specific variants
        (e.g. parent-ordering with/without adaptive-bagging trajectory context)
        without widening the dispatch table.
        """
        content = (
            system_prompt if system_prompt is not None
            else prompts.get_system_prompt(operation_type)
        )
        messages: List[ChatMessage] = [
            {"role": "system", "content": content},
            {"role": "user", "content": message},
        ]
        return self._complete(messages, operation_type, logged_prompt=message)

    def send_history(
        self, message_history: List[ChatMessage], operation_type: str = "general"
    ) -> str:
        """Send an already-assembled multi-turn conversation."""
        logged_prompt = next(
            (m["content"] for m in reversed(message_history) if m["role"] == "user"),
            "",
        )
        return self._complete(
            list(message_history), operation_type, logged_prompt=logged_prompt
        )

    def _complete(
        self,
        messages: List[ChatMessage],
        operation_type: str,
        *,
        logged_prompt: str,
    ) -> str:
        arguments: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            arguments["temperature"] = self.temperature
        if self.reasoning_effort:
            arguments["reasoning_effort"] = self.reasoning_effort

        cache_key = None
        if self._cache is not None:
            cache_key = make_key(
                model=arguments["model"],
                messages=arguments["messages"],
                max_completion_tokens=arguments.get("max_completion_tokens"),
                temperature=arguments.get("temperature"),
                reasoning_effort=arguments.get("reasoning_effort"),
            )
            hit = self._cache.get(cache_key)
            if hit is not None:
                content = hit["content"]
                usage: Mapping[str, int] = hit.get("usage") or {}
                self._record_usage(
                    f"{operation_type}_cached",
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                    cached_prompt_tokens=usage.get("cached_prompt_tokens", 0),
                )
                if self.verbose:
                    logging.info(f"[cache hit] Prompt: {logged_prompt}")
                    logging.info(f"[cache hit] Response: {content}")
                return content

        # The budget counts *uncached* attempts: a cache hit is free, so it
        # should not push us closer to the abort threshold.
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise LlmCallBudgetExceeded(
                f"Exceeded max_calls={self.max_calls} during operation "
                f"'{operation_type}'. Aborting run."
            )
        self.calls_made += 1

        response = self._client.chat.completions.create(**arguments)

        finish_reason = response.choices[0].finish_reason
        truncated = finish_reason == "length"
        if truncated:
            logging.warning(
                f"[{operation_type}] hit max_completion_tokens={self.max_tokens} "
                f"(finish_reason=length); response will be unparseable and fall back."
            )

        usage_dict = None
        if getattr(response, "usage", None):
            details = getattr(response.usage, "prompt_tokens_details", None)
            cached_prompt_tokens = getattr(details, "cached_tokens", 0) or 0
            self._record_usage(
                operation_type,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                cached_prompt_tokens=cached_prompt_tokens,
                truncated=truncated,
            )
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "cached_prompt_tokens": cached_prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        message_content = response.choices[0].message.content
        if message_content is None:
            raise ValueError(f"Model returned no content (finish_reason={finish_reason})")
        content = message_content.strip()

        if self._cache is not None and cache_key is not None and not truncated:
            self._cache.put(cache_key, content, usage_dict)

        if self.verbose:
            logging.info(f"Prompt: {logged_prompt}")
            logging.info(f"Response: {content}")
            if getattr(response, "usage", None):
                logging.info(
                    f"Tokens used: {response.usage.total_tokens} "
                    f"(prompt: {response.usage.prompt_tokens}, "
                    f"completion: {response.usage.completion_tokens})"
                )

        return content
