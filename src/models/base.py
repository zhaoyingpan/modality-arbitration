from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelInput:
    prompt: str
    image_path: Path | None
    image_blur_radius: float = 0.0
    context: str | None = None
    question: str | None = None


@dataclass(frozen=True)
class GenerationBatch:
    texts: list[str]
    metrics: dict[str, Any]
    sample_metrics: list[dict[str, Any]] = field(default_factory=list)


class ModelAdapter(ABC):
    """Small interface shared by local Hugging Face and future API models."""

    @abstractmethod
    def generate(self, inputs: list[ModelInput]) -> GenerationBatch:
        raise NotImplementedError
