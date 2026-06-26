"""
LLM generation layer with Ollama primary + fallback chain.
Design doc Section 10.
"""
from __future__ import annotations

import logging
import time
from typing import AsyncGenerator, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from api.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a precise document assistant. Answer ONLY from the provided context.
If the answer is not in the context, say "I don't have enough information to answer that from the provided documents."
Always cite your sources using [Source: filename] notation at the end of your answer."""


# ── Ollama generation ─────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
async def generate_ollama(
    prompt: str,
    model: str,
    system: str = SYSTEM_PROMPT,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """Call Ollama /api/generate and return {text, model, prompt_tokens, completion_tokens}."""
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": f"<|system|>\n{system}\n\n<|user|>\n{prompt}\n\n<|assistant|>",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["<|user|>", "<|system|>"],
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "text": data.get("response", "").strip(),
        "model": model,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "latency_ms": latency_ms,
    }


# ── Streaming generation ───────────────────────────────────────────────────────
async def stream_ollama(
    prompt: str,
    model: str,
    system: str = SYSTEM_PROMPT,
) -> AsyncGenerator[str, None]:
    """Streaming generation — yields text tokens as they arrive."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": f"<|system|>\n{system}\n\n<|user|>\n{prompt}\n\n<|assistant|>",
                "stream": True,
                "options": {"temperature": 0.1},
            },
        ) as resp:
            import json as _json
            async for line in resp.aiter_lines():
                if line:
                    try:
                        chunk = _json.loads(line)
                        if token := chunk.get("response"):
                            yield token
                        if chunk.get("done"):
                            break
                    except Exception:
                        continue


# ── Fallback chain ─────────────────────────────────────────────────────────────
async def generate_with_fallback(
    query: str,
    context: str,
    cached_answer: Optional[str] = None,
) -> dict:
    """
    Fallback chain (design doc Section 10.2):
    1. Ollama primary model (qwen2.5:7b)
    2. Ollama fallback model (llama3.1:8b)
    3. Cached similar answer
    4. Error response with trace info
    """
    prompt = f"""Context:
{context}

Question: {query}

Answer concisely and cite sources."""

    # 1. Primary model
    try:
        result = await generate_ollama(prompt, model=settings.ollama_model)
        result["fallback_used"] = False
        return result
    except Exception as e:
        logger.warning(f"Primary model ({settings.ollama_model}) failed: {e}")

    # 2. Fallback model
    try:
        result = await generate_ollama(prompt, model=settings.ollama_fallback_model)
        result["fallback_used"] = True
        return result
    except Exception as e:
        logger.warning(f"Fallback model ({settings.ollama_fallback_model}) failed: {e}")

    # 3. Cached similar answer
    if cached_answer:
        logger.warning("Using cached similar answer as last resort")
        return {
            "text": cached_answer,
            "model": "cache",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "fallback_used": True,
        }

    # 4. Error response
    return {
        "text": "I'm temporarily unable to generate a response. Please try again shortly.",
        "model": "error",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0,
        "fallback_used": True,
        "error": True,
    }
