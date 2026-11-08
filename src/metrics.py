from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Iterable, Mapping, Any


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_SPACES = re.compile(r"\s+")
_PUNCTUATION = str.maketrans({char: " " for char in string.punctuation})


def normalize_answer(value: str) -> str:
    value = value.casefold().translate(_PUNCTUATION)
    value = _ARTICLES.sub(" ", value)
    return _SPACES.sub(" ", value).strip()


def is_correct(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_answer(prediction: str, ground_truth: str) -> bool:
    """Whether the full normalized reference occurs as a contiguous token sequence."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(ground_truth).split()
    if not reference_tokens or len(reference_tokens) > len(prediction_tokens):
        return False
    width = len(reference_tokens)
    return any(
        prediction_tokens[start : start + width] == reference_tokens
        for start in range(len(prediction_tokens) - width + 1)
    )


def score_prediction_contains_answers(
    prediction: str, answer_vision: str, answer_text: str
) -> dict[str, bool]:
    return {
        "correct_vision": contains_answer(prediction, answer_vision),
        "correct_text": contains_answer(prediction, answer_text),
    }


def score_prediction(prediction: str, answer_vision: str, answer_text: str) -> dict[str, bool]:
    return {
        "correct_vision": is_correct(prediction, answer_vision),
        "correct_text": is_correct(prediction, answer_text),
    }


def aggregate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    result: dict[str, Any] = {}
    for condition, items in grouped.items():
        count = len(items)
        result[condition] = {
            "count": count,
            "accuracy_vision": sum(
                bool(x.get("exact_correct_vision", x.get("correct_vision"))) for x in items
            ) / count,
            "accuracy_text": sum(
                bool(x.get("exact_correct_text", x.get("correct_text"))) for x in items
            ) / count,
        }
    return result
