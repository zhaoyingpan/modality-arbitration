from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from metrics import contains_answer


DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
JUDGE_PROMPT_VERSION = "1.5"
JUDGE_INSTRUCTIONS = """You are an impartial answer-equivalence judge.
Decide whether a candidate answer is semantically equivalent to the provided reference answer for the given question.
Judge meaning, not exact wording. Accept synonyms, paraphrases, equivalent numbers, harmless grammatical or formatting differences, and reasonable differences in specificity or detail. Accept the subset or superset of a ground-truth answer as long as any omitted or added detail does not materially change or contradict the core answer.
Mark the result incorrect only if it contradicts or is semantically inconsistent with the ground-truth answer.
Return the requested structured result only."""


class ReferenceJudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match: bool = Field(description="Candidate is semantically equivalent to the reference answer")


REFERENCE_TYPES = ("answer_vision", "answer_text")
REFERENCE_SUFFIXES = {"answer_vision": "vision", "answer_text": "text"}


def _available_reference_types(row: dict[str, Any]) -> list[str]:
    available = [
        reference_type
        for reference_type in REFERENCE_TYPES
        if reference_type in row and row[reference_type] is not None
    ]
    if available:
        return available
    # Aggregation also accepts legacy/minimal parsed rows without raw answers.
    return [
        reference_type
        for reference_type in REFERENCE_TYPES
        if f"parser_correct_{REFERENCE_SUFFIXES[reference_type]}" in row
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
    return rows


def _judge_reference(
    client: OpenAI,
    *,
    question: str,
    candidate: str,
    reference_type: str,
    reference_answer: str,
    model: str,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "candidate": candidate,
        "reference_type": reference_type,
        "reference_answer": reference_answer,
    }
    base = {"reference_type": reference_type, "reference_answer": reference_answer}
    try:
        response = client.responses.parse(
            model=model,
            instructions=JUDGE_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=ReferenceJudgeVerdict,
            temperature=0,
            max_output_tokens=128,
            store=False,
        )
    except Exception as error:
        return {
            **base,
            "status": "api_failure",
            "parser_failed": False,
            "api_failed": True,
            "error": f"{type(error).__name__}: {error}",
            "raw_output": None,
            "parsed_result": None,
            "response_id": None,
            "usage": None,
            "match": None,
        }
    verdict = response.output_parsed
    usage = getattr(response, "usage", None)
    usage_record = usage.model_dump() if usage is not None else None
    if verdict is None:
        return {
            **base,
            "status": "parsing_failure",
            "parser_failed": True,
            "api_failed": False,
            "error": "No parsed structured verdict returned",
            "raw_output": response.output_text,
            "parsed_result": None,
            "response_id": response.id,
            "usage": usage_record,
            "match": None,
        }
    return {
        **base,
        "status": "parsed",
        "parser_failed": False,
        "api_failed": False,
        "error": None,
        "raw_output": response.output_text,
        "parsed_result": verdict.model_dump(),
        "response_id": response.id,
        "usage": usage_record,
        "match": verdict.match,
    }


def judge_one(client: OpenAI, row: dict[str, Any], model: str) -> dict[str, Any]:
    inference_result = row.get("inference_result", row.get("prediction"))
    if inference_result is None:
        raise ValueError("Prediction row has neither inference_result nor legacy prediction field")
    question = row.get("question") or _question_from_prompt(str(row.get("prompt", "")))
    results = {
        reference_type: _judge_reference(
            client,
            question=str(question),
            candidate=str(inference_result),
            reference_type=reference_type,
            reference_answer=str(row[reference_type]),
            model=model,
        )
        for reference_type in _available_reference_types(row)
    }
    statuses = [result["status"] for result in results.values()]
    if all(status == "parsed" for status in statuses):
        status = "parsed"
    elif any(status == "parsed" for status in statuses):
        status = "partial_failure"
    elif any(status == "parsing_failure" for status in statuses):
        status = "parsing_failure"
    else:
        status = "api_failure"
    parsed_result = {
        f"{REFERENCE_SUFFIXES[reference_type]}_match": result["match"]
        for reference_type, result in results.items()
    }
    output = {
        "judge_status": status,
        "judge_results": results,
        "parser_failed": any(result["parser_failed"] for result in results.values()),
        "api_failed": any(result["api_failed"] for result in results.values()),
        "judge_error": {key: result["error"] for key, result in results.items()},
        "judge_raw_output": {key: result["raw_output"] for key, result in results.items()},
        "judge_response_id": {key: result["response_id"] for key, result in results.items()},
        "judge_usage": {key: result["usage"] for key, result in results.items()},
        "parsed_result": parsed_result,
    }
    for reference_type, result in results.items():
        output[f"parser_correct_{REFERENCE_SUFFIXES[reference_type]}"] = result["match"]
    return output


def _question_from_prompt(prompt: str) -> str:
    # Current inference prompts put the question immediately before this suffix.
    marker = "\nAnswer the question directly with a short answer."
    before = prompt.split(marker, 1)[0]
    return before.rsplit("\n", 1)[-1].strip()


def _parser_fields_last(row: dict[str, Any]) -> None:
    correctness = {
        suffix: row.pop(f"parser_correct_{suffix}")
        for suffix in ("vision", "text")
        if f"parser_correct_{suffix}" in row
    }
    failed = row.pop("parser_failed")
    for suffix in ("vision", "text"):
        if suffix in correctness:
            row[f"parser_correct_{suffix}"] = correctness[suffix]
    row["parser_failed"] = failed


def _aggregate_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    reference_types = [
        reference_type
        for reference_type in REFERENCE_TYPES
        if any(reference_type in _available_reference_types(row) for row in items)
    ]
    def reference_status(row: dict[str, Any], gt_type: str) -> str:
        result = row.get("judge_results", {}).get(gt_type)
        if result:
            return str(result.get("status", "parsing_failure"))
        return str(row.get("judge_status", "parsing_failure"))

    parsed_by_reference = {
        REFERENCE_SUFFIXES[reference_type]: [
            row for row in items if reference_status(row, reference_type) == "parsed"
        ]
        for reference_type in reference_types
    }
    parsed = [
        row for row in items
        if all(reference_status(row, reference_type) == "parsed" for reference_type in reference_types)
    ]
    parsing_failures = sum(
        bool(row.get("parser_failed", row.get("judge_status") == "parsing_failure"))
        for row in items
    )
    api_failures = sum(
        bool(row.get("api_failed", row.get("judge_status") == "api_failure")) for row in items
    )
    parsed_count = len(parsed)
    reference_counts = {
        suffix: len(reference_rows) for suffix, reference_rows in parsed_by_reference.items()
    }
    reference_parsing_failure_rates = {
        REFERENCE_SUFFIXES[reference_type]: sum(
            reference_status(row, reference_type) == "parsing_failure" for row in items
        ) / total if total else 0.0
        for reference_type in reference_types
    }
    reference_api_failure_rates = {
        REFERENCE_SUFFIXES[reference_type]: sum(
            reference_status(row, reference_type) == "api_failure" for row in items
        ) / total if total else 0.0
        for reference_type in reference_types
    }
    unique_batches: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(items):
        metrics = row.get("batch_metrics")
        if metrics:
            unique_batches.setdefault(str(row.get("batch_id", f"legacy:{position}")), metrics)
    batches = list(unique_batches.values())
    total_end_to_end = sum(float(x["end_to_end_inference_latency_s"]) for x in batches)
    total_generation = sum(float(x["generation_latency_s"]) for x in batches)
    output_tokens = sum(int(x["output_tokens"]) for x in batches)
    input_tokens = sum(int(x["input_tokens"]) for x in batches)
    measured_requests = sum(int(x["batch_size"]) for x in batches)
    ttfts = [float(x["ttft_s"]) for x in batches if x.get("ttft_s") is not None]
    decode_batches = [x for x in batches if x.get("decode_latency_s") is not None]
    total_decode_latency = sum(float(x["decode_latency_s"]) for x in decode_batches)
    total_decode_tokens = sum(int(x["decode_tokens"]) for x in decode_batches)
    peak_memories = [int(x["peak_gpu_memory_bytes"]) for x in batches if x.get("peak_gpu_memory_bytes") is not None]
    seeds = sorted({row.get("random_seed") for row in items if row.get("random_seed") is not None})
    run_ids = sorted({str(row["run_id"]) for row in items if row.get("run_id")})
    timestamps = sorted(
        str(row.get("judge_timestamp_utc") or row.get("run_timestamp_utc"))
        for row in items
        if row.get("judge_timestamp_utc") or row.get("run_timestamp_utc")
    )
    metrics = {
        "timestamp_utc": timestamps[-1] if timestamps else datetime.now(timezone.utc).isoformat(),
        "benchmark": str(items[0].get("dataset", "mc2")) if items else "mc2",
        "model": str(items[0].get("model", "")) if items else "",
        "method": str(items[0].get("method", "base")) if items else "base",
        "judge_model": str(items[0].get("judge_model", DEFAULT_JUDGE_MODEL)) if items else DEFAULT_JUDGE_MODEL,
        "parser_type": str(items[0].get("parser_type", "llm")) if items else "llm",
        "judge_prompt_version": str(items[0].get("judge_prompt_version", "")) if items else "",
        "random_seed": seeds[0] if len(seeds) == 1 else json.dumps(seeds),
        "run_ids": json.dumps(run_ids),
        "method_config": json.dumps(items[0].get("method_config", {}), sort_keys=True) if items else "{}",
        "generation_config": json.dumps(items[0].get("generation_config", {}), sort_keys=True) if items else "{}",
        "judge_config": json.dumps(items[0].get("judge_config", {}), sort_keys=True) if items else "{}",
        "total": total,
        "gt_types": reference_types,
        "parsed_count": parsed_count,
        "parsed_rate": parsed_count / total if total else 0.0,
        "parsing_failure_count": parsing_failures,
        "parsing_failure_rate": parsing_failures / total if total else 0.0,
        "api_failure_count": api_failures,
        "api_failure_rate": api_failures / total if total else 0.0,
        "end_to_end_inference_latency_s": (
            total_end_to_end / measured_requests if measured_requests else None
        ),
        "ttft_s": sum(ttfts) / len(ttfts) if ttfts else None,
        "decode_latency_per_output_token_s": (
            total_decode_latency / total_decode_tokens if total_decode_tokens else None
        ),
        "requests_per_second": (
            measured_requests / total_end_to_end if total_end_to_end else None
        ),
        "token_throughput_tokens_per_second": (
            output_tokens / total_generation if total_generation else None
        ),
        "peak_gpu_memory_bytes": max(peak_memories) if peak_memories else None,
        "input_tokens": input_tokens if batches else None,
        "output_tokens": output_tokens if batches else None,
    }
    for reference_type in reference_types:
        suffix = REFERENCE_SUFFIXES[reference_type]
        count = reference_counts[suffix]
        metrics[f"parsed_count_{suffix}"] = count
        metrics[f"parsing_failure_rate_{suffix}"] = reference_parsing_failure_rates[suffix]
        metrics[f"api_failure_rate_{suffix}"] = reference_api_failure_rates[suffix]
        metrics[f"accuracy_{suffix}"] = (
            sum(
                bool(row[f"parser_correct_{suffix}"])
                for row in parsed_by_reference[suffix]
            ) / count if count else None
        )
    return metrics


def aggregate_judgments(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["condition"])].append(row)
    summary: dict[str, Any] = {"overall": _aggregate_group(all_rows)}
    for condition, items in grouped.items():
        summary[condition] = _aggregate_group(items)
    return summary


CSV_FIELDS = [
    "timestamp_utc",
    "benchmark",
    "subfolder_path",
    "model",
    "condition",
    "method",
    "parser_type",
    "judge_model",
    "judge_prompt_version",
    "random_seed",
    "gt_type",
    "n",
    "total_n",
    "accuracy",
    "extract_failure_rate",
    "api_failure_rate",
    "end_to_end_inference_latency_s",
    "ttft_s",
    "decode_latency_per_output_token_s",
    "requests_per_second",
    "token_throughput_tokens_per_second",
    "peak_gpu_memory_bytes",
    "input_tokens",
    "output_tokens",
    "run_ids",
    "method_config",
    "generation_config",
    "judge_config",
]


def write_summary_csv(
    summary: dict[str, Any], output_file: Path, *, subfolder_path: str = ""
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scopes = sorted(scope for scope in summary if scope != "overall") or ["overall"]
    with output_file.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        write_header = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for scope in scopes:
            metrics = summary[scope]
            shared = {
                "timestamp_utc": metrics["timestamp_utc"],
                "benchmark": metrics["benchmark"],
                "subfolder_path": subfolder_path,
                "model": metrics["model"],
                "condition": scope,
                "method": metrics["method"],
                "parser_type": metrics["parser_type"],
                "judge_model": metrics["judge_model"],
                "judge_prompt_version": metrics["judge_prompt_version"],
                "random_seed": metrics["random_seed"],
                "n": metrics["parsed_count"],
                "total_n": metrics["total"],
                "extract_failure_rate": metrics["parsing_failure_rate"],
                "api_failure_rate": metrics["api_failure_rate"],
                "end_to_end_inference_latency_s": metrics["end_to_end_inference_latency_s"],
                "ttft_s": metrics["ttft_s"],
                "decode_latency_per_output_token_s": metrics["decode_latency_per_output_token_s"],
                "requests_per_second": metrics["requests_per_second"],
                "token_throughput_tokens_per_second": metrics["token_throughput_tokens_per_second"],
                "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "run_ids": metrics["run_ids"],
                "method_config": metrics["method_config"],
                "generation_config": metrics["generation_config"],
                "judge_config": metrics["judge_config"],
            }
            for gt_type in metrics["gt_types"]:
                metric_suffix = REFERENCE_SUFFIXES[gt_type]
                writer.writerow(
                    {
                        **shared,
                        "gt_type": gt_type,
                        "n": metrics[f"parsed_count_{metric_suffix}"],
                        "accuracy": metrics[f"accuracy_{metric_suffix}"],
                        "extract_failure_rate": metrics[
                            f"parsing_failure_rate_{metric_suffix}"
                        ],
                        "api_failure_rate": metrics[f"api_failure_rate_{metric_suffix}"],
                    }
                )
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _score_references(row: dict[str, Any]) -> dict[str, bool]:
    return {
        reference_type: contains_answer(
            str(row["inference_result"]), str(row[reference_type])
        )
        for reference_type in _available_reference_types(row)
    }


def _exact_result_fields(scores: dict[str, bool]) -> dict[str, Any]:
    judge_results = {
        reference_type: {
            "reference_type": reference_type,
            "status": "parsed",
            "parser_failed": False,
            "api_failed": False,
            "match": match,
            "source": "exact_matcher",
        }
        for reference_type, match in scores.items()
    }
    output = {
        "judge_status": "parsed",
        "parser_result_source": "exact_matcher",
        "parser_failed": False,
        "api_failed": False,
        "judge_error": None,
        "judge_raw_output": None,
        "judge_response_id": None,
        "judge_usage": None,
        "judge_results": judge_results,
        "parsed_result": {
            f"{REFERENCE_SUFFIXES[reference_type]}_match": match
            for reference_type, match in scores.items()
        },
    }
    for reference_type, match in scores.items():
        output[f"parser_correct_{REFERENCE_SUFFIXES[reference_type]}"] = match
    return output


def run_hybrid_parsing(
    client: OpenAI,
    input_file: Path,
    output_file: Path,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    judge_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_rows = _read_jsonl(input_file)
    parsed_rows: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8") as handle:
        for row in tqdm(source_rows, desc=f"parser:hybrid:{model}"):
            scores = _score_references(row)
            parsed = dict(row)
            parsed.update({
                "record_schema_version": "1.0",
                "parser_type": "hybrid",
                "judge_model": model,
                "judge_timestamp_utc": timestamp,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "judge_instructions": JUDGE_INSTRUCTIONS,
                "judge_config": judge_config or {
                    "temperature": 0,
                    "max_output_tokens": 128,
                    "structured_output": "ReferenceJudgeVerdict",
                },
                "exact_match_result": {
                    f"{REFERENCE_SUFFIXES[reference_type]}_match": match
                    for reference_type, match in scores.items()
                },
                "exact_match_mode": "normalized_reference_contiguous_token_subset",
            })
            # MC2 short-circuits when exactly one of its two references matches.
            # A single-reference benchmark short-circuits on a positive match.
            if sum(scores.values()) == 1:
                parsed.update(_exact_result_fields(scores))
            else:
                parsed.update(judge_one(client, row, model))
                parsed["parser_result_source"] = "llm"
            _parser_fields_last(parsed)
            handle.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            handle.flush()
            parsed_rows.append(parsed)
    return aggregate_judgments(parsed_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse and score saved benchmark predictions")
    parser.add_argument("--run-name", required=True, help="Run folder created by inference")
    parser.add_argument("--results-root", type=Path, default=Path("results/mc2/qwen2_5_vl_7b"))
    parser.add_argument("--input", type=Path, help="Override the run-scoped raw-generation path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_dotenv(".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing; set it in .env")
    client = OpenAI(max_retries=2, timeout=60.0)
    input_file = args.input or args.results_root / "runs" / args.run_name / "inference" / "raw_generations.jsonl"
    output_file = args.results_root / "runs" / args.run_name / "inference" / "predictions.jsonl"
    summary_file = args.results_root / "runs" / "performance.csv"
    judge_config = {
        "temperature": 0,
        "max_output_tokens": 128,
        "structured_output": "ReferenceJudgeVerdict",
        "max_retries": 2,
        "timeout_seconds": 60.0,
    }
    summary = run_hybrid_parsing(
        client, input_file, output_file, model=DEFAULT_JUDGE_MODEL,
        judge_config=judge_config,
    )
    write_summary_csv(
        summary,
        summary_file,
        subfolder_path=f"runs/{args.run_name}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
