"""Built-in nano-vllm model implementations and registry helpers."""

# Import built-in architectures so they register themselves on import.
from . import qwen3 as _qwen3  # noqa: F401
from .registry import available_model_types, get_model_class, register_model

__all__ = [
    "available_model_types",
    "get_model_class",
    "register_model",
]
