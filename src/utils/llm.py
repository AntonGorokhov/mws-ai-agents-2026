"""OpenRouter LLM client with retry and fallback."""

import time
import logging
import httpx

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE, MODELS

logger = logging.getLogger(__name__)


def call_llm(
    model: str,
    system: str,
    user: str,
    fallback: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    retries: int = 2,
) -> dict:
    """Call OpenRouter LLM. Returns {"content": str, "model_used": str, "tokens": dict}."""
    models_to_try = [model]
    if fallback:
        models_to_try.append(fallback)

    last_error = None
    for current_model in models_to_try:
        for attempt in range(retries + 1):
            try:
                result = _request(current_model, system, user, temperature, max_tokens)
                return result
            except (httpx.HTTPStatusError, httpx.TimeoutException, Exception) as e:
                last_error = e
                logger.warning(
                    "LLM call failed: model=%s attempt=%d error=%s",
                    current_model, attempt + 1, str(e),
                )
                if attempt < retries:
                    time.sleep(2 ** attempt)

    raise RuntimeError(f"All LLM calls failed. Last error: {last_error}")


def _request(
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(OPENROUTER_BASE, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {
        "content": choice,
        "model_used": model,
        "tokens": {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        },
    }
