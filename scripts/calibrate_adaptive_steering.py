#!/usr/bin/env python3
"""Fit the fixed MC2 adaptive-steering similarity threshold."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from conditions import CONDITIONS, ConditionName
from data import load_mc2
from models.base import ModelInput
from models.registry import MODEL_DEFAULTS, create_model_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", default="qwen3-vl-8b")
    parser.add_argument("--model")
    parser.add_argument("--data-file", type=Path, default=Path("datasets/mc2/evaluate_final_aligned.json"))
    parser.add_argument("--image-root", type=Path, default=Path("datasets/mc2/noise_images/coco_noise_gs"))
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steering-artifact",
        default="artifacts/steering/qwen3_vl_8b_resolution_heads_top10_n0012.json",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def select_threshold(conflicting: list[float], aligned: list[float]) -> tuple[float, dict]:
    values = sorted(set(conflicting + aligned))
    candidates = [values[0] - 1e-9]
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    candidates.append(values[-1] + 1e-9)
    best = None
    for threshold in candidates:
        conflict_accuracy = sum(value < threshold for value in conflicting) / len(conflicting)
        aligned_accuracy = sum(value >= threshold for value in aligned) / len(aligned)
        balanced_accuracy = (conflict_accuracy + aligned_accuracy) / 2
        candidate = (balanced_accuracy, conflict_accuracy, aligned_accuracy, -threshold)
        if best is None or candidate > best[0]:
            best = (candidate, threshold)
    assert best is not None
    threshold = best[1]
    conflict_accuracy = sum(value < threshold for value in conflicting) / len(conflicting)
    aligned_accuracy = sum(value >= threshold for value in aligned) / len(aligned)
    labels = [0] * len(conflicting) + [1] * len(aligned)
    scores = conflicting + aligned
    return threshold, {
        "n_indices": len(conflicting),
        "n_pairs": len(scores),
        "conflicting_accuracy": conflict_accuracy,
        "aligned_accuracy": aligned_accuracy,
        "accuracy": (conflict_accuracy + aligned_accuracy) / 2,
        "balanced_accuracy": (conflict_accuracy + aligned_accuracy) / 2,
        "roc_auc": auc(labels, scores),
    }


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")
    conflicting = load_mc2(args.data_file, args.image_root, conflict_ratio=1, seed=args.seed)
    aligned = load_mc2(args.data_file, args.image_root, conflict_ratio=0, seed=args.seed)
    if args.sample_size > len(conflicting):
        raise ValueError("sample-size exceeds MC2 size")
    selected = random.Random(args.seed).sample(range(len(conflicting)), args.sample_size)
    completed: dict[int, dict] = {}
    if args.resume and args.records.exists():
        with args.records.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                completed[int(row["dataset_index"])] = row
    args.records.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "x"
    model = create_model_adapter(
        args.model_family,
        args.model or MODEL_DEFAULTS[args.model_family],
        dtype="bfloat16",
        device_map="auto",
        attn_implementation="sdpa",
        # Match adaptive inference's custom eager-attention numerics while
        # leaving the controller inactive during both unimodal probes.
        method="steering_image",
        method_config={
            "artifact_path": args.steering_artifact,
            "top_k": args.top_k,
            "alpha": 0.0,
            "epsilon": 1e-12,
            "suppress_opposite": True,
            "apply_to": "all",
        },
        seed=args.seed,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    prompt = CONDITIONS[ConditionName.IMAGE_CONTEXT]
    with args.records.open(mode, encoding="utf-8") as handle:
        for position, index in enumerate(selected, 1):
            if index in completed:
                continue
            conflict_example = conflicting[index]
            aligned_example = aligned[index]

            def similarity(example) -> float:
                item = ModelInput(
                    prompt=prompt.prompt(example.question, example.context),
                    image_path=example.image_path,
                    context=example.context,
                    question=example.question,
                )
                return model.compute_unimodal_similarities([item])[0]

            row = {
                "dataset_index": index,
                "sample_order": position,
                "random_seed": args.seed,
                "conflicting_similarity": similarity(conflict_example),
                "aligned_similarity": similarity(aligned_example),
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            completed[index] = row
            print(f"[{position}/{args.sample_size}] index={index} "
                  f"conflict={row['conflicting_similarity']:.6f} "
                  f"aligned={row['aligned_similarity']:.6f}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    rows = [completed[index] for index in selected]
    conflict_scores = [float(row["conflicting_similarity"]) for row in rows]
    aligned_scores = [float(row["aligned_similarity"]) for row in rows]
    threshold, metrics = select_threshold(conflict_scores, aligned_scores)
    artifact = {
        "method": "adaptive_steering",
        "model_family": args.model_family,
        "model": args.model or MODEL_DEFAULTS[args.model_family],
        "benchmark": "mc2",
        "seed": args.seed,
        "sample_size": args.sample_size,
        "sample_indices": selected,
        "top_token_count": 50,
        "steering_top_k": args.top_k,
        "probe_attention_path": "custom eager attention with inactive steering",
        "threshold": threshold,
        "threshold_selection": "maximum balanced accuracy; similarity < threshold is conflicting",
        "evaluation_split": "threshold-fitting 200-index paired calibration set",
        "metrics": metrics,
        "elapsed_s_this_invocation": elapsed,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": str(args.records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
