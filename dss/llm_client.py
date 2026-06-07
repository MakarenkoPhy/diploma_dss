"""
Унифицированный интерфейс к LLM-провайдерам.

Все клиенты реализуют единый метод complete(system, user) → str.
Реализован экспоненциальный backoff для rate limit (429) и серверных ошибок (5xx).
"""

from __future__ import annotations
import os
import time
from abc import ABC, abstractmethod


class LLMClient(ABC):
    name: str = "abstract"
    model: str = "n/a"

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        ...


def _should_retry(exc) -> bool:
    """Определяет, стоит ли повторять запрос после данной ошибки."""
    code = getattr(exc, "status_code", None)
    if code is not None:
        return code in (429, 500, 502, 503, 529)
    # Сетевые ошибки без статус-кода (ConnectionError, TimeoutError и т.п.)
    return True


def _backoff_sleep(attempt: int) -> None:
    """Экспоненциальная задержка: 2, 4, 8 с (не более 30 с)."""
    delay = min(2 ** attempt, 30)
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        use_cache: bool = True,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.use_cache = use_cache
        self.max_retries = max_retries
        self._client = None
        self.cache_stats = {"created": 0, "read": 0, "uncached_input": 0}

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        client = self._get_client()

        if self.use_cache:
            system_param = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_param = system

        last_exc = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                _backoff_sleep(attempt)
            try:
                resp = client.messages.create(
                    model=self.model,
                    system=system_param,
                    messages=[{"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout=60.0,
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.cache_stats["created"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
                    self.cache_stats["read"]    += getattr(usage, "cache_read_input_tokens", 0) or 0
                    self.cache_stats["uncached_input"] += getattr(usage, "input_tokens", 0) or 0
                return resp.content[0].text
            except Exception as e:
                last_exc = e
                if not _should_retry(e):
                    raise
        raise RuntimeError(
            f"Anthropic: все {self.max_retries + 1} попытки исчерпаны. "
            f"Последняя ошибка: {last_exc}"
        )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_retries = max_retries
        self._client = None
        self.cache_stats = {"cached_input": 0, "uncached_input": 0}

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, timeout=60.0)
        return self._client

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        last_exc = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                _backoff_sleep(attempt)
            try:
                resp = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    details = getattr(usage, "prompt_tokens_details", None)
                    cached = getattr(details, "cached_tokens", 0) or 0
                    self.cache_stats["cached_input"]   += cached
                    self.cache_stats["uncached_input"] += (usage.prompt_tokens - cached)
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                if not _should_retry(e):
                    raise
        raise RuntimeError(
            f"OpenAI: все {self.max_retries + 1} попытки исчерпаны. "
            f"Последняя ошибка: {last_exc}"
        )


# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-совместимый)
# ---------------------------------------------------------------------------

class DeepSeekClient(LLMClient):
    name = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url
        self.max_retries = max_retries

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        import openai
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0,
        )
        last_exc = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                _backoff_sleep(attempt)
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=1e-8,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                if not _should_retry(e):
                    raise
        raise RuntimeError(
            f"DeepSeek: все {self.max_retries + 1} попытки исчерпаны. "
            f"Последняя ошибка: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Ручной ввод (для отладки и работы эксперта-человека)
# ---------------------------------------------------------------------------

class ManualClient(LLMClient):
    name = "manual"
    model = "manual"

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        print("\n" + "─" * 70)
        print("[SYSTEM]")
        print(system)
        print("─" * 70)
        print("[USER]")
        print(user)
        print("─" * 70)
        return input("Ваш ответ: ")


def make_client(provider: str, **kwargs) -> LLMClient:
    p = provider.lower()
    if p == "anthropic":
        return AnthropicClient(**kwargs)
    if p == "openai":
        return OpenAIClient(**kwargs)
    if p == "deepseek":
        return DeepSeekClient(**kwargs)
    if p == "manual":
        return ManualClient()
    raise ValueError(f"Неизвестный провайдер: {provider}")
