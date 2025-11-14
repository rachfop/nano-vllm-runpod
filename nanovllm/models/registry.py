"""Simple registry so nano-vllm can support multiple architectures.

New model backends should call ``register_model(<model_type>)`` at import
time. ``model_type`` should match ``AutoConfig.model_type`` for the
corresponding Hugging Face model. ``ModelRunner`` can then pick up the
appropriate implementation dynamically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from transformers import PretrainedConfig

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
    "available_model_types",
]

T = TypeVar("T")

# Maps Hugging Face config.model_type -> nano-vllm model class.
MODEL_REGISTRY: dict[str, type] = {}


def register_model(model_type: str) -> Callable[[type[T]], type[T]]:
    """Decorator that registers a model implementation.

    Example:
        @register_model("qwen3")
        class Qwen3ForCausalLM(nn.Module):
            ...
    """

    def decorator(cls: type[T]) -> type[T]:
        MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator


def get_model_class(config: PretrainedConfig) -> type:
    """Return the registered nano-vllm model class for ``config``."""
    model_type = getattr(config, "model_type", None)
    if model_type is None:
        raise ValueError("Config object is missing `model_type`")
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<none>"
        msg = f"Unsupported model_type '{model_type}'. Registered types: {available}"
        raise ValueError(msg) from exc


def available_model_types() -> list[str]:
    """Return list of registered model types (for better error messages)."""
    return sorted(MODEL_REGISTRY.keys())
