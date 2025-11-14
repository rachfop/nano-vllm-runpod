"""Thin wrapper that exposes :class:`LLMEngine` with a friendlier name."""

from .engine import LLMEngine


class LLM(LLMEngine):
    pass
