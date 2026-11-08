from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GAUSSIAN_BLUR_RADII: dict[int, float] = {0: 0.0, 1: 2.0, 2: 5.0, 3: 10.0}
DEFAULT_CORRELATIONQA_VISION_ANSWERS = Path(
    "datasets/correlationqa/vision_answers_gpt_4_1_mini_prompt_v1_4.jsonl"
)
PHD_SUBSET_CONTEXT_KEYS = {
    # The MPS paper calls this subset PhD-ica in prose and PhD-iac in tables.
    # The pinned current Hugging Face data stores the corresponding context as `sec`.
    "phd-ica": "phd-ica",
    "phd-icc": "phd-icc",
}


@dataclass(frozen=True)
class BenchmarkExample:
    index: int
    image_id: str
    task: str
    question: str
    context: str
    answer_vision: str | None
    answer_text: str | None
    image_path: Path
    benchmark: str = "mc2"
    context_source: str = "context"
    image_blur_level: int = 0
    image_blur_radius: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MC2PairedExample:
    """One MC² example with both conflicting and aligned text contexts."""

    index: int
    image_id: str
    task: str
    question: str
    context: str
    aligned_context: str
    answer_vision: str
    answer_text: str
    image_path: Path


def _image_index(image_root: Path) -> dict[str, Path]:
    paths = sorted(image_root.rglob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No .jpg files found beneath {image_root}")
    index: dict[str, Path] = {}
    for path in paths:
        image_id = path.stem.rsplit("_", 1)[-1]
        index.setdefault(image_id, path)
    return index


def load_mc2(
    data_file: Path,
    image_root: Path,
    limit: int | None = None,
    *,
    conflict_ratio: float = 1.0,
    seed: int = 42,
    image_blur_level: int = 0,
) -> list[BenchmarkExample]:
    if not 0.0 <= conflict_ratio <= 1.0:
        raise ValueError("conflict_ratio must be between 0 and 1 inclusive")
    if image_blur_level not in GAUSSIAN_BLUR_RADII:
        raise ValueError("image_blur_level must be one of 0, 1, 2, or 3")
    with data_file.open(encoding="utf-8") as handle:
        rows: Any = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {data_file}")
    selected_rows = rows[:limit]
    conflict_count = int(len(selected_rows) * conflict_ratio + 0.5)
    conflict_indices = set(random.Random(seed).sample(range(len(selected_rows)), conflict_count))
    images = _image_index(image_root)
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(selected_rows):
        image_id = str(row["image_id"]).zfill(12)
        if image_id not in images:
            raise FileNotFoundError(f"No image found for MC2 image_id={image_id}")
        use_conflicting_context = index in conflict_indices
        if not use_conflicting_context and "aligned_context" not in row:
            raise ValueError(
                f"MC2 row {index} has no aligned_context, required when conflict_ratio < 1"
            )
        examples.append(
            BenchmarkExample(
                index=index,
                image_id=image_id,
                task=str(row["task"]),
                question=str(row["question"]),
                context=str(row["context"] if use_conflicting_context else row["aligned_context"]),
                answer_vision=str(row["answer_vision"]),
                answer_text=str(row["answer_text"]),
                image_path=images[image_id],
                context_source="context" if use_conflicting_context else "aligned_context",
                image_blur_level=image_blur_level,
                image_blur_radius=GAUSSIAN_BLUR_RADII[image_blur_level],
            )
        )
    return examples


def load_mc2_pairs(
    data_file: Path,
    image_root: Path,
    limit: int | None = None,
    *,
    start: int = 0,
) -> list[MC2PairedExample]:
    """Load paired conflicting/aligned contexts for activation patching."""
    if start < 0:
        raise ValueError("start must be non-negative")
    with data_file.open(encoding="utf-8") as handle:
        rows: Any = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {data_file}")
    images = _image_index(image_root)
    examples: list[MC2PairedExample] = []
    stop = None if limit is None else start + limit
    for index, row in enumerate(rows[start:stop], start=start):
        if "aligned_context" not in row:
            raise ValueError(f"MC2 row {index} has no aligned_context")
        image_id = str(row["image_id"]).zfill(12)
        if image_id not in images:
            raise FileNotFoundError(f"No image found for MC2 image_id={image_id}")
        examples.append(
            MC2PairedExample(
                index=index,
                image_id=image_id,
                task=str(row["task"]),
                question=str(row["question"]),
                context=str(row["context"]),
                aligned_context=str(row["aligned_context"]),
                answer_vision=str(row["answer_vision"]),
                answer_text=str(row["answer_text"]),
                image_path=images[image_id],
            )
        )
    return examples


def load_phd(
    data_file: Path,
    image_root: Path,
    subset: str,
    limit: int | None = None,
    *,
    image_blur_level: int = 0,
) -> list[BenchmarkExample]:
    """Load one fixed 1,000-question PhD context subset with vision-only labels."""
    if subset not in PHD_SUBSET_CONTEXT_KEYS:
        choices = ", ".join(PHD_SUBSET_CONTEXT_KEYS)
        raise ValueError(f"PhD subset must be one of: {choices}")
    if image_blur_level not in GAUSSIAN_BLUR_RADII:
        raise ValueError("image_blur_level must be one of 0, 1, 2, or 3")
    with data_file.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Expected a fixed PhD split object with samples in {data_file}")
    if payload.get("benchmark") != "phd":
        raise ValueError(f"Unexpected benchmark in {data_file}: {payload.get('benchmark')!r}")
    rows = payload["samples"]
    expected = payload.get("sampling", {}).get("sample_count_per_subset")
    if expected is not None and len(rows) != int(expected):
        raise ValueError(
            f"Fixed PhD split contains {len(rows)} samples; metadata declares {expected}"
        )
    selected_rows = rows[:limit]
    root = image_root.resolve()
    examples: list[BenchmarkExample] = []
    for position, row in enumerate(selected_rows):
        index = int(row.get("dataset_index", position))
        if index != position:
            raise ValueError(
                f"PhD split indices must be contiguous from zero; found {index} at {position}"
            )
        contexts = row.get("contexts")
        if not isinstance(contexts, dict) or subset not in contexts:
            raise ValueError(f"PhD row {index} has no context for {subset}")
        relative_image = Path(str(row["image_path"]))
        image_path = (image_root / relative_image).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"PhD row {index} image escapes image root") from error
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing PhD image for row {index}: {image_path}")
        examples.append(
            BenchmarkExample(
                index=index,
                image_id=str(row["image_id"]),
                task=str(row["task"]),
                question=str(row["question"]),
                context=str(contexts[subset]),
                answer_vision=str(row["answer_vision"]),
                answer_text=None,
                image_path=image_path,
                benchmark=subset,
                context_source=subset,
                image_blur_level=image_blur_level,
                image_blur_radius=GAUSSIAN_BLUR_RADII[image_blur_level],
            )
        )
    return examples


def load_mmcd(
    data_file: Path,
    image_root: Path,
    limit: int | None = None,
    *,
    image_blur_level: int = 0,
) -> list[BenchmarkExample]:
    """Load prepared MMCD metadata and extracted images."""
    if image_blur_level not in GAUSSIAN_BLUR_RADII:
        raise ValueError("image_blur_level must be one of 0, 1, 2, or 3")
    rows: list[Any] = []
    with data_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {data_file}:{line_number}") from error
    selected_rows = rows[:limit]
    root = image_root.resolve()
    required = {
        "dataset_index", "id", "caption", "shape", "text_color", "image_color",
        "image_filename",
    }
    examples: list[BenchmarkExample] = []
    for position, row in enumerate(selected_rows):
        if not isinstance(row, dict):
            raise ValueError(f"MMCD row {position} must be a JSON object")
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"MMCD row {position} is missing fields: {', '.join(sorted(missing))}"
            )
        index = int(row["dataset_index"])
        if index != position:
            raise ValueError(
                f"MMCD indices must be contiguous from zero; found {index} at {position}"
            )
        shape = str(row["shape"]).strip()
        if not shape:
            raise ValueError(f"MMCD row {index} has an empty shape")
        image_path = (image_root / str(row["image_filename"])).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"MMCD row {index} image escapes image root") from error
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing MMCD image for row {index}: {image_path}")
        examples.append(
            BenchmarkExample(
                index=index,
                image_id=str(row["id"]),
                task="color",
                question=f"what color is the {shape}?",
                context=str(row["caption"]),
                answer_vision=str(row["image_color"]),
                answer_text=str(row["text_color"]),
                image_path=image_path,
                benchmark="mmcd",
                context_source="caption",
                image_blur_level=image_blur_level,
                image_blur_radius=GAUSSIAN_BLUR_RADII[image_blur_level],
            )
        )
    return examples


def load_clash(
    data_file: Path,
    image_root: Path,
    limit: int | None = None,
    *,
    image_blur_level: int = 0,
) -> list[BenchmarkExample]:
    """Load CLASH's filtered synthetic-caption test split."""
    if image_blur_level not in GAUSSIAN_BLUR_RADII:
        raise ValueError("image_blur_level must be one of 0, 1, 2, or 3")
    with data_file.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"Expected a CLASH object with a samples list in {data_file}")
    rows = payload["samples"][:limit]
    root = image_root.resolve()
    required = {
        "image_id", "original_caption", "conflicting_caption", "question",
        "change_type", "answers", "changed_words", "category",
    }
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"CLASH row {index} must be a JSON object")
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"CLASH row {index} is missing fields: {', '.join(sorted(missing))}"
            )
        answers = row["answers"]
        if not isinstance(answers, dict):
            raise ValueError(f"CLASH row {index} answers must be an object")
        missing_answers = {"image_only", "text_only"} - answers.keys()
        if missing_answers:
            raise ValueError(
                f"CLASH row {index} is missing answers: "
                f"{', '.join(sorted(missing_answers))}"
            )
        image_id = str(row["image_id"]).zfill(12)
        image_path = (image_root / f"{image_id}.jpg").resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"CLASH row {index} image escapes image root") from error
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing CLASH image for row {index}: {image_path}")
        examples.append(
            BenchmarkExample(
                index=index,
                image_id=image_id,
                task=str(row["change_type"]),
                question=str(row["question"]),
                context=str(row["conflicting_caption"]),
                answer_vision=str(answers["image_only"]),
                answer_text=str(answers["text_only"]),
                image_path=image_path,
                benchmark="clash",
                context_source="conflicting_caption",
                image_blur_level=image_blur_level,
                image_blur_radius=GAUSSIAN_BLUR_RADII[image_blur_level],
                metadata={
                    "original_caption": str(row["original_caption"]),
                    "change_type": str(row["change_type"]),
                    "category": str(row["category"]),
                    "changed_words": row["changed_words"],
                    "irrelevant_but_plausible": answers.get("irrelevant_but_plausible"),
                },
            )
        )
    return examples


# Backward-compatible import name used by existing scripts and tests.
MC2Example = BenchmarkExample


_CORRELATIONQA_QUESTION_RE = re.compile(
    r"^(?P<context>.+?)\s+"
    r"(?P<question>(?:What|Who)\b[^?]*\?\s+Answer no more than five words\.)$",
    flags=re.IGNORECASE | re.DOTALL,
)


def split_correlationqa_question(raw_question: str) -> tuple[str, str]:
    """Separate CorrelationQA's descriptive context from its final question."""
    normalized = raw_question.strip()
    match = _CORRELATIONQA_QUESTION_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(
            "Could not split CorrelationQA question into context and final question: "
            f"{raw_question!r}"
        )
    return match.group("context").strip(), match.group("question").strip()


def load_correlationqa(
    data_path: Path,
    image_root: Path,
    limit: int | None = None,
    *,
    vision_answers_file: Path | None = DEFAULT_CORRELATIONQA_VISION_ANSWERS,
    image_blur_level: int = 0,
) -> list[BenchmarkExample]:
    """Load local CorrelationQA Parquet metadata and pre-extracted row images."""
    if image_blur_level not in GAUSSIAN_BLUR_RADII:
        raise ValueError("image_blur_level must be one of 0, 1, 2, or 3")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "CorrelationQA requires pyarrow to read its local Parquet shards"
        ) from error
    parquet_files = (
        [data_path] if data_path.is_file() else sorted(data_path.rglob("*.parquet"))
    )
    if not parquet_files:
        raise FileNotFoundError(f"No CorrelationQA Parquet files found beneath {data_path}")
    table = parquet.read_table(
        parquet_files,
        columns=["image_path", "category", "question", "true_answer"],
    )
    row_count = table.num_rows if limit is None else min(limit, table.num_rows)
    rows = table.slice(0, row_count).to_pylist()
    vision_answers: dict[int, dict[str, Any]] = {}
    if vision_answers_file is not None:
        if not vision_answers_file.is_file():
            raise FileNotFoundError(
                f"CorrelationQA vision answers do not exist: {vision_answers_file}. "
                "Run scripts/generate_correlationqa_vision_answers.py first."
            )
        with vision_answers_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    generated = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on {vision_answers_file}:{line_number}"
                    ) from error
                generated_index = int(generated.get("dataset_index", -1))
                if generated_index in vision_answers:
                    raise ValueError(
                        f"Duplicate CorrelationQA vision answer index {generated_index}"
                    )
                vision_answers[generated_index] = generated
        missing_indices = [index for index in range(row_count) if index not in vision_answers]
        if missing_indices:
            raise ValueError(
                f"{vision_answers_file} is incomplete for the selected data: "
                f"missing index {missing_indices[0]} ({len(missing_indices)} missing total)"
            )
    examples: list[BenchmarkExample] = []
    for index, row in enumerate(rows):
        missing = {"image_path", "category", "question", "true_answer"} - row.keys()
        if missing:
            raise ValueError(
                f"CorrelationQA row {index} is missing fields: {', '.join(sorted(missing))}"
            )
        source_name = Path(str(row["image_path"])).name
        image_path = image_root / f"{index}.png"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"No extracted CorrelationQA image for row {index}: {image_path}"
            )
        answer_text = str(row["true_answer"])
        raw_question = str(row["question"])
        context, question = split_correlationqa_question(raw_question)
        generated = vision_answers.get(index)
        answer_vision = (
            None if generated is None else str(generated.get("answer_vision", "")).strip()
        )
        if generated is not None:
            if not answer_vision:
                raise ValueError(f"Missing answer_vision for CorrelationQA row {index}")
            if generated.get("question") not in {raw_question, question}:
                raise ValueError(f"Vision-answer question mismatch at CorrelationQA row {index}")
            if generated.get("image_path") != str(image_path):
                raise ValueError(f"Vision-answer image mismatch at CorrelationQA row {index}")
            if generated.get("answer_text") != answer_text:
                raise ValueError(f"Text-answer mismatch at CorrelationQA row {index}")
        examples.append(
            BenchmarkExample(
                index=index,
                image_id=Path(source_name).stem or str(index),
                task=str(row["category"]),
                question=question,
                context=context,
                answer_vision=answer_vision,
                answer_text=answer_text,
                image_path=image_path,
                benchmark="correlationqa",
                context_source="raw_question_description",
                image_blur_level=image_blur_level,
                image_blur_radius=GAUSSIAN_BLUR_RADII[image_blur_level],
            )
        )
    return examples
