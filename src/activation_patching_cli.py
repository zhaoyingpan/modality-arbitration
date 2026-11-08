from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from activation_patching import (
    ANSWER_SCORE,
    PATCH_POSITION,
    PATCH_SITE,
    PATCHING_INSTRUCTION,
    ActivationPatchingDiagnoser,
    import_screening_predictions,
    load_jsonl_by_index,
    run_activation_patching,
    write_filtered_dataset,
)
from cli import seed_everything, software_versions
from data import load_mc2_pairs
from models.registry import MODEL_DEFAULTS, RESULT_SLUGS, get_hf_model_class


LOCAL_FAMILIES = tuple(MODEL_DEFAULTS)
DATA_FILE = Path("datasets/mc2/evaluate_final_one_token.json")
IMAGE_ROOT = Path("datasets/mc2/noise_images/coco_noise_gs")
CALIBRATION_SIZES = {
    "qwen2.5-vl-7b": 10,
    "qwen3-vl-8b": 12,
    "llava-ov-7b": 10,
    "internvl3-8b": 10,
}
MAX_NEW_TOKENS = 128
HEAD_BATCH_SIZE = 4
DTYPE = "bfloat16"
DEVICE_MAP = "auto"
SEED = 42


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MC2 head-output activation patching")
    parser.add_argument("--model-family", choices=LOCAL_FAMILIES, default="qwen3-vl-8b")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--image-only-predictions",
        type=Path,
        required=True,
        help="Hybrid-parsed image-only predictions",
    )
    parser.add_argument(
        "--text-only-predictions",
        type=Path,
        required=True,
        help="Hybrid-parsed context-only predictions",
    )
    parser.add_argument(
        "--conflict-predictions",
        type=Path,
        required=True,
        help="Hybrid-parsed image-context predictions used to label baseline preference",
    )
    return parser


def _load_diagnoser(args: argparse.Namespace) -> ActivationPatchingDiagnoser:
    return ActivationPatchingDiagnoser.from_pretrained(
        MODEL_DEFAULTS[args.model_family],
        get_hf_model_class(args.model_family),
        dtype=DTYPE,
        device_map=DEVICE_MAP,
        max_new_tokens=MAX_NEW_TOKENS,
        head_batch_size=HEAD_BATCH_SIZE,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results_root = Path("results/mc2") / RESULT_SLUGS[args.model_family]
    run_root = results_root / "activation_patching" / args.run_name
    screening_file = run_root / "unimodal_screening.jsonl"
    filtered_file = run_root / "filtered_mc2.json"
    output_file = run_root / "samples.jsonl"
    manifest_file = run_root / "run_manifest.json"
    eligible_limit = CALIBRATION_SIZES[args.model_family]
    manifest = {
        "benchmark": "mc2",
        "diagnosis": "head_output_activation_patching",
        "run_name": args.run_name,
        "model": MODEL_DEFAULTS[args.model_family],
        "model_family": args.model_family,
        "data_file": str(DATA_FILE),
        "image_root": str(IMAGE_ROOT),
        "eligible_limit": eligible_limit,
        "instruction": PATCHING_INSTRUCTION,
        "filter": "image_only correct on answer_vision AND context_only correct on answer_text",
        "screening_source": "import",
        "screening_prediction_files": {
            "image_only": str(args.image_only_predictions),
            "context_only": str(args.text_only_predictions),
            "image_context": str(args.conflict_predictions),
        },
        "inputs": {
            "conflict": "image + conflicting context",
            "vision_only": "image + question",
            "text_only": "conflicting context + question",
        },
        "margin": "log_p(answer_vision) - log_p(answer_text)",
        "answer_score": ANSWER_SCORE,
        "patch_site": PATCH_SITE,
        "patch_position": PATCH_POSITION,
        "head_batch_size": HEAD_BATCH_SIZE,
        "seed": SEED,
        "generation_config": {
            "greedy": True,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": DTYPE,
            "device_map": DEVICE_MAP,
            "attention_implementation": "sdpa",
            "deterministic_algorithms": False,
        },
        "software_versions": software_versions(),
        "artifacts": {
            "unimodal_screening": str(screening_file),
            "filtered_dataset": str(filtered_file),
            "samples": str(output_file),
        },
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_file.exists():
        with manifest_file.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    else:
        raise FileExistsError(manifest_file)

    seed_everything(SEED, False)
    examples = load_mc2_pairs(
        DATA_FILE,
        IMAGE_ROOT,
    )
    screening_summary = import_screening_predictions(
        examples,
        args.image_only_predictions,
        args.text_only_predictions,
        screening_file,
        conflict_predictions=args.conflict_predictions,
    )
    screening_rows = load_jsonl_by_index(screening_file)
    diagnosis_examples = [
        example
        for example in examples
        if example.index in screening_rows and bool(screening_rows[example.index]["retained"])
    ]
    diagnosis_examples = diagnosis_examples[:eligible_limit]
    filtered_count = write_filtered_dataset(
        filtered_file,
        diagnosis_examples,
        screening_rows,
    )
    print(json.dumps({
        "screening": screening_summary,
        "filtered_count": filtered_count,
        "screening_output": str(screening_file),
        "filtered_output": str(filtered_file),
    }, indent=2))
    diagnoser = _load_diagnoser(args)
    patching_summary = run_activation_patching(
        diagnoser,
        diagnosis_examples,
        screening_rows,
        output_file,
    )
    print(json.dumps({
        "activation_patching": patching_summary,
        "output": str(output_file),
    }, indent=2))
    del diagnoser
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
