"""Local LLM client for Ollama-backed text generation."""

from __future__ import annotations

import logging
from typing import Any

import ollama

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "phi3:mini"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class LLMError(Exception):
    """Raised when the LLM client cannot generate a response."""


def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
) -> str:
    """
    Generate a text response from a local Ollama model.

    Args:
        prompt: Full prompt text sent to the model.
        model: Ollama model name (default: `phi3:mini`).
        host: Ollama server base URL.

    Returns:
        Generated response text from the model.

    Raises:
        ValueError: If the prompt is empty.
        LLMError: If Ollama is unreachable or generation fails.
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    logger.info("Generating response with model '%s'", model)

    client = ollama.Client(host=host)

    try:
        response: dict[str, Any] = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
        )

    except ConnectionError as exc:
        logger.exception("Could not connect to Ollama at %s", host)

        raise LLMError(
            f"Ollama is not reachable at '{host}'. "
            "Ensure Ollama is running and the model is pulled."
        ) from exc

    except Exception as exc:
        logger.exception("Ollama generation failed for model '%s'", model)

        raise LLMError(f"LLM generation failed: {exc}") from exc

    answer = str(response.get("response", "")).strip()

    if not answer:
        raise LLMError("Ollama returned an empty response")

    logger.info("Generated response (%d character(s))", len(answer))

    return answer