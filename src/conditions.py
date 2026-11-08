from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConditionName(str, Enum):
    IMAGE_ONLY = "image_only"
    CONTEXT_ONLY = "context_only"
    IMAGE_CONTEXT = "image_context"
    IMAGE_CONTEXT_INSTRUCT_IMAGE = "image_context_instruct_image"
    IMAGE_CONTEXT_INSTRUCT_TEXT = "image_context_instruct_text"


@dataclass(frozen=True)
class Condition:
    name: ConditionName
    use_image: bool
    use_context: bool
    instruction: str | None = None

    def prompt(self, question: str, context: str) -> str:
        parts: list[str] = []
        if self.use_context:
            parts.append(f"Text context: {context}")
        if self.instruction:
            parts.append(self.instruction)
        parts.extend((question, "Answer the question directly with a short answer."))
        return "\n".join(parts)


CONDITIONS: dict[ConditionName, Condition] = {
    ConditionName.IMAGE_ONLY: Condition(ConditionName.IMAGE_ONLY, True, False),
    ConditionName.CONTEXT_ONLY: Condition(ConditionName.CONTEXT_ONLY, False, True),
    ConditionName.IMAGE_CONTEXT: Condition(ConditionName.IMAGE_CONTEXT, True, True),
    ConditionName.IMAGE_CONTEXT_INSTRUCT_IMAGE: Condition(
        ConditionName.IMAGE_CONTEXT_INSTRUCT_IMAGE,
        True,
        True,
        "Answer the question based on the image",
    ),
    ConditionName.IMAGE_CONTEXT_INSTRUCT_TEXT: Condition(
        ConditionName.IMAGE_CONTEXT_INSTRUCT_TEXT,
        True,
        True,
        "Answer the question based on the text",
    ),
}


def parse_conditions(values: list[str] | None) -> list[Condition]:
    names = list(ConditionName) if not values else [ConditionName(value) for value in values]
    return [CONDITIONS[name] for name in names]

