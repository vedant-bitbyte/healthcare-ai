"""Local LLM client for Ollama-backed text generation."""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

import ollama
import torch

from src.config.settings import HF_ADAPTER_PATH, HF_BASE_MODEL

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "phi3:mini"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class LLMError(Exception):
    """Raised when the LLM client cannot generate a response."""


class HFSingleton:
    """Singleton for loading the fine-tuned Hugging Face PEFT model."""
    _model = None
    _tokenizer = None

    @classmethod
    def load(cls, base_model_name: str = HF_BASE_MODEL, adapter_path: str = HF_ADAPTER_PATH):
        if cls._model is None:
            from src.evaluation.inference import load_model_for_inference
            logger.info("Loading fine-tuned Hugging Face model from %s...", adapter_path)
            cls._model, cls._tokenizer = load_model_for_inference(base_model_name, adapter_path)
            logger.info("Fine-tuned model successfully loaded into memory.")
        return cls._model, cls._tokenizer


def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
) -> str:
    """
    Generate a text response from a local model (Ollama or Fine-tuned HF).
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if model == "fine-tuned":
        logger.info("Generating response using local fine-tuned PEFT model.")
        try:
            hf_model, tokenizer = HFSingleton.load()

            # Format input prompt as user message
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt_text, return_tensors="pt").to(hf_model.device)

            with torch.no_grad():
                output_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.0
                )

            input_length = inputs.input_ids.shape[1]
            generated_ids = output_ids[0][input_length:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if not answer:
                raise LLMError("Hugging Face model returned an empty response")

            logger.info("Generated fine-tuned response (%d character(s))", len(answer))
            return answer

        except Exception as exc:
            logger.exception("HF generation failed")
            raise LLMError(f"HF LLM generation failed: {exc}") from exc

    logger.info("Generating response with Ollama model '%s'", model)

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
