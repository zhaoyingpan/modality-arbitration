#!/usr/bin/env python3
"""Rewrite MC2 contexts to align text evidence with the vision answer.

The output is checkpointed after every successfully processed example.  At any
point it is a valid JSON list containing the completed prefix of the input, so
an interrupted invocation can be resumed safely by running the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm


DEFAULT_INPUT = Path("datasets/mc2/evaluate_final.json")
DEFAULT_OUTPUT = Path("datasets/mc2/evaluate_final_aligned.json")
DEFAULT_MODEL = "gpt-4.1-mini"

INSTRUCTIONS = """You rewrite a context so its relevant evidence agrees with a target answer.
Replace every statement, description, implication, synonym, or paraphrase semantically associated with the original text answer with content consistent with the vision answer.
Preserve all unrelated content as closely as possible, including wording, detail, tone, style, paragraphing, and structure.
Resolve any nearby wording that would otherwise contradict the vision answer, but do not invent unrelated details.
Return only the rewritten context in the requested structured format."""


class AlignedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aligned_context: str = Field(description="The rewritten context aligned with answer_vision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--api-max-retries", type=int, default=5)
    parser.add_argument("--api-timeout", type=float, default=180.0)
    return parser.parse_args()


def read_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain a JSON list of dictionaries")
    return value


def validate_row(row: dict[str, Any], index: int) -> None:
    missing = {"context", "answer_text", "answer_vision"} - row.keys()
    if missing:
        raise ValueError(f"Input row {index} is missing fields: {sorted(missing)}")


def validate_resume(
    source: list[dict[str, Any]], completed: list[dict[str, Any]], output: Path
) -> None:
    if len(completed) > len(source):
        raise ValueError(f"{output} has more rows than the source dataset")
    for index, output_row in enumerate(completed):
        if "aligned_context" not in output_row:
            raise ValueError(f"Existing output row {index} has no aligned_context")
        original_fields = {key: value for key, value in output_row.items() if key != "aligned_context"}
        if original_fields != source[index]:
            raise ValueError(f"Existing output row {index} does not match the source")


def atomic_write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def rewrite_context(
    client: OpenAI,
    row: dict[str, Any],
    *,
    model: str,
    max_output_tokens: int,
) -> str:
    payload = {
        "context": row["context"],
        "original_text_answer": row["answer_text"],
        "target_vision_answer": row["answer_vision"],
    }
    response = client.responses.parse(
        model=model,
        instructions=INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=False),
        text_format=AlignedContext,
        max_output_tokens=max_output_tokens,
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError(f"No structured result returned: {response.output_text!r}")
    return parsed.aligned_context


def main() -> None:
    args = parse_args()
    load_dotenv()
    source = read_list(args.input)
    for index, row in enumerate(source):
        validate_row(row, index)

    completed = read_list(args.output) if args.output.exists() else []
    validate_resume(source, completed, args.output)
    if len(completed) == len(source):
        print(f"Already complete: {args.output} ({len(completed)} rows)")
        return

    client = OpenAI(max_retries=args.api_max_retries, timeout=args.api_timeout)
    progress = tqdm(range(len(completed), len(source)), initial=len(completed), total=len(source))
    for index in progress:
        row = source[index]
        aligned = rewrite_context(
            client,
            row,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        # Dict expansion preserves every source field and appends only the new field.
        completed.append({**row, "aligned_context": aligned})
        atomic_write_json(args.output, completed)
        progress.set_postfix_str(f"saved={index + 1}")

    print(f"Wrote {len(completed)} rows to {args.output}")


if __name__ == "__main__":
    main()
