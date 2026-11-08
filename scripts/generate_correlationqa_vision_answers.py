#!/usr/bin/env python3
"""Generate CorrelationQA vision-source answers with GPT-4.1-mini.

Each API request contains only the row image and decomposed final question. The
leading description from CorrelationQA's raw question field is excluded.
Results are appended one row at a time to a resumable JSONL file; the existing
file is never overwritten.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from data import BenchmarkExample, load_correlationqa


DEFAULT_DATA_PATH = Path(
    "datasets/huggingface/hub/datasets--ScottHan--CorrelationQA/snapshots/"
    "730efdd9eee2dc1c3221f2a5528ce06e3203ecbb/data"
)
DEFAULT_IMAGE_ROOT = Path("datasets/correlation_qa_images")
DEFAULT_OUTPUT = Path(
    "datasets/correlationqa/vision_answers_gpt_4_1_mini_prompt_v1_4.jsonl"
)
DEFAULT_MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "1.4"
QUESTION_FORMAT = "decomposed_final_question_only"
INSTRUCTIONS = (
    "Answer the following question. An image is provided as additional context. "
    "Give the most factually correct answer from the image to the question. "
    "Return only the answer in no more than 2 words, without explanation. "
    "Do not use negation in the answer."
)
NEGATION_WORDS = {"no", "not", "never", "none", "neither", "nor", "without"}
FORMAT_RETRY_INSTRUCTION = (
    "The previous response violated the output constraints. "
    "Name the depicted subject directly using a positive one- or two-word noun phrase."
)


def format_retry_instruction(previous_answer: str | None) -> str:
    if previous_answer is None:
        return FORMAT_RETRY_INSTRUCTION
    return (
        f"The previous answer was {json.dumps(previous_answer)} and violated the output "
        "constraints. Shorten it to at most two words while preserving the same meaning. "
        "Return only the shortened positive answer."
    )


class VisionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_vision: str = Field(
        description="An answer of no more than two words supported by the visual content of the image"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-api-calls-per-entry",
        type=int,
        default=3,
        help="Hard cap on application-level API calls for one dataset entry",
    )
    parser.add_argument(
        "--constraint-failures-output",
        type=Path,
        help="JSONL for entries saved from a final answer that violates output constraints",
    )
    return parser.parse_args()


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def answer_word_count(answer: str) -> int:
    """Count natural-language words, treating a hyphenated term as one word."""
    return len(re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", answer, flags=re.UNICODE))


def answer_uses_negation(answer: str) -> bool:
    words = {
        word.lower()
        for word in re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", answer, flags=re.UNICODE)
    }
    return bool(words & NEGATION_WORDS) or "n't" in answer.lower()


def read_completed(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on {path}:{line_number}")
            rows.append(row)
    return rows


def validate_resume(
    examples: list[BenchmarkExample], completed: list[dict[str, Any]], output: Path
) -> None:
    if len(completed) > len(examples):
        raise ValueError(f"{output} contains more rows than the selected dataset")
    for index, row in enumerate(completed):
        example = examples[index]
        if row.get("dataset_index") != index:
            raise ValueError(f"{output} is not a contiguous prefix at row {index}")
        if row.get("question") != example.question:
            raise ValueError(f"Question mismatch in {output} at row {index}")
        if row.get("image_path") != str(example.image_path):
            raise ValueError(f"Image mismatch in {output} at row {index}")
        if not str(row.get("answer_vision", "")).strip():
            raise ValueError(f"Missing answer_vision in {output} at row {index}")


def generate_answer(
    client: OpenAI,
    example: BenchmarkExample,
    *,
    model: str,
    max_output_tokens: int,
    image_detail: str,
    max_api_calls_per_entry: int = 3,
) -> tuple[str, Any, float, bool, str | None, int]:
    started_at = perf_counter()
    last_error = "No response attempted"
    previous_answer: str | None = None
    last_answer: str | None = None
    last_response: Any | None = None
    for attempt in range(max_api_calls_per_entry):
        instructions = (
            INSTRUCTIONS
            if attempt == 0
            else f"{INSTRUCTIONS} {format_retry_instruction(previous_answer)}"
        )
        try:
            response = client.responses.parse(
                model=model,
                instructions=instructions,
                input=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_data_url(example.image_path),
                            "detail": image_detail,
                        },
                        {"type": "input_text", "text": example.question},
                    ],
                }],
                text_format=VisionAnswer,
                temperature=0,
                max_output_tokens=max_output_tokens,
                store=False,
            )
        except Exception as error:
            last_error = f"API call failed: {type(error).__name__}: {error}"
            continue
        parsed = response.output_parsed
        if parsed is None or not parsed.answer_vision.strip():
            last_error = f"No structured vision answer returned: {response.output_text!r}"
            previous_answer = None
            continue
        answer = parsed.answer_vision.strip()
        previous_answer = answer
        last_answer = answer
        last_response = response
        word_count = answer_word_count(answer)
        if 1 <= word_count <= 2 and not answer_uses_negation(answer):
            return answer, response, perf_counter() - started_at, True, None, attempt + 1
        if answer_uses_negation(answer):
            last_error = f"Vision answer must not use negation: {answer!r}"
        else:
            last_error = (
                f"Vision answer must contain one or two words, got {word_count}: {answer!r}"
            )
    if last_answer is not None and last_response is not None:
        return (
            last_answer,
            last_response,
            perf_counter() - started_at,
            False,
            last_error,
            max_api_calls_per_entry,
        )
    raise RuntimeError(
        f"No answer was returned after {max_api_calls_per_entry} API calls; {last_error}"
    )


def response_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)


def default_constraint_failures_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}_constraint_failures.jsonl")


def constraint_failure_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "failure_log_schema_version": "1.0",
        **record,
    }


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "x"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def synchronize_constraint_failures(
    output: Path, completed: list[dict[str, Any]]
) -> set[int]:
    existing = read_completed(output) if output.exists() else []
    indices: set[int] = set()
    for row in existing:
        index = row.get("dataset_index")
        if not isinstance(index, int) or index in indices:
            raise ValueError(f"Invalid or duplicate dataset_index in {output}: {index!r}")
        indices.add(index)
    for record in completed:
        index = record.get("dataset_index")
        if record.get("constraint_satisfied") is False and index not in indices:
            append_jsonl_record(output, constraint_failure_record(record))
            indices.add(index)
    return indices


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.max_output_tokens < 1:
        raise SystemExit("--max-output-tokens must be positive")
    if args.max_api_calls_per_entry < 1:
        raise SystemExit("--max-api-calls-per-entry must be positive")
    load_dotenv(args.env_file)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(f"OPENAI_API_KEY is missing; set it in {args.env_file}")

    examples = load_correlationqa(
        args.data_path,
        args.image_root,
        limit=args.limit,
        vision_answers_file=None,
    )
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; pass --resume to continue it"
        )
    completed = read_completed(args.output) if args.output.exists() else []
    validate_resume(examples, completed, args.output)
    failures_output = (
        args.constraint_failures_output
        if args.constraint_failures_output is not None
        else default_constraint_failures_output(args.output)
    )
    if failures_output.resolve() == args.output.resolve():
        raise ValueError("Constraint-failures output must differ from the main output")
    failure_indices = synchronize_constraint_failures(failures_output, completed)
    if len(completed) == len(examples):
        print(f"Already complete: {args.output} ({len(completed)} rows)")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Disable SDK retries so max_api_calls_per_entry is a hard cap on HTTP requests.
    client = OpenAI(max_retries=0, timeout=args.api_timeout)
    mode = "a" if args.output.exists() else "x"
    with args.output.open(mode, encoding="utf-8") as handle:
        progress = tqdm(
            range(len(completed), len(examples)),
            initial=len(completed),
            total=len(examples),
            desc=f"vision-answers:{args.model}",
        )
        for index in progress:
            example = examples[index]
            (
                answer,
                response,
                latency_s,
                constraint_satisfied,
                constraint_failure_reason,
                api_call_count,
            ) = generate_answer(
                client,
                example,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                image_detail=args.image_detail,
                max_api_calls_per_entry=args.max_api_calls_per_entry,
            )
            record = {
                "record_schema_version": "1.0",
                "dataset": "correlationqa",
                "dataset_index": index,
                "sample_id": f"correlationqa:{index}",
                "image_id": example.image_id,
                "image_filename": example.image_path.name,
                "image_path": str(example.image_path),
                "category": example.task,
                "question": example.question,
                "question_format": QUESTION_FORMAT,
                # Retained for later conflict evaluation; never included in the API request.
                "answer_text": example.answer_text,
                "answer_vision": answer,
                "answer_vision_word_count": answer_word_count(answer),
                "generator_model": args.model,
                "prompt_version": PROMPT_VERSION,
                "instructions": INSTRUCTIONS,
                "image_detail": args.image_detail,
                "max_output_tokens": args.max_output_tokens,
                "format_retries": args.max_api_calls_per_entry - 1,
                "max_api_calls_per_entry": args.max_api_calls_per_entry,
                "api_call_count": api_call_count,
                "format_retry_instruction": FORMAT_RETRY_INSTRUCTION,
                "constraint_satisfied": constraint_satisfied,
                "constraint_failure_reason": constraint_failure_reason,
                "temperature": 0,
                "response_id": response.id,
                "usage": response_usage(response),
                "api_latency_s": latency_s,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if not constraint_satisfied and index not in failure_indices:
                append_jsonl_record(
                    failures_output,
                    constraint_failure_record(record),
                )
                failure_indices.add(index)
            progress.set_postfix_str(f"saved={index + 1}")

    print(f"Wrote {len(examples)} rows to {args.output}")


if __name__ == "__main__":
    main()
