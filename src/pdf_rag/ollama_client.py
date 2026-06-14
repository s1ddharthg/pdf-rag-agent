"""Helpers for interacting with the local Ollama model registry."""

import logging
from typing import Any, Tuple

import ollama

logger = logging.getLogger(__name__)


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """Extract model name strings from an Ollama ``list()`` response."""
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects.
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()

        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def get_available_models() -> Tuple[str, ...]:
    """Return the names of locally available Ollama models."""
    models_info = ollama.list()
    return extract_model_names(models_info)
