from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI

from conditions import ConditionName, parse_conditions
from data import (
    DEFAULT_CORRELATIONQA_VISION_ANSWERS,
    load_clash,
    load_correlationqa,
    load_mc2,
    load_mmcd,
    load_phd,
)
from judge import (
    DEFAULT_JUDGE_MODEL,
    run_hybrid_parsing,
    write_summary_csv,
)
from models.registry import MODEL_DEFAULTS, RESULT_SLUGS, create_model_adapter
from methods import MethodName, parse_method, resolve_method_config
from runner import run_evaluation


DEFAULT_MODEL_FAMILY = "qwen2.5-vl-7b"
DEFAULT_MC2_DATA = Path("datasets/mc2/evaluate_final_aligned.json")
DEFAULT_MC2_IMAGES = Path("datasets/mc2/noise_images/coco_noise_gs")
DEFAULT_PHD_DATA = Path("datasets/phd/evaluation_seed42.json")
DEFAULT_PHD_ROOT = Path("datasets/phd/source")
DEFAULT_MMCD_ROOT = Path("datasets/mmcd")
DEFAULT_CLASH_DATA = Path("datasets/clash/synthetic_captions_test_filtered.json")
DEFAULT_CLASH_IMAGES = Path("datasets/clash/images/train2017")
DEFAULT_CORRELATIONQA_DATA = Path(
    "datasets/huggingface/hub/datasets--ScottHan--CorrelationQA/snapshots/"
    "730efdd9eee2dc1c3221f2a5528ce06e3203ecbb/data"
)
DEFAULT_CORRELATIONQA_IMAGES = Path("datasets/correlation_qa_images")


def benchmark_name(value: str) -> str:
    normalized = value.lower()
    if normalized == "correqa":
        return "correlationqa"
    if normalized not in {"mc2", "correlationqa", "phd", "mmcd", "clash"}:
        raise argparse.ArgumentTypeError(
            "Benchmark must be mc2, correlationqa, correqa, phd, mmcd, or clash"
        )
    return normalized


def phd_subset_name(value: str) -> str:
    normalized = value.lower()
    aliases = {
        "phd-ica": "phd-ica",
        "phd-iac": "phd-ica",
        "phd-sec": "phd-ica",
        "phd-icc": "phd-icc",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError("PhD subset must be phd-ica or phd-icc")
    return aliases[normalized]


def ratio(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a floating-point value") from error
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1 inclusive")
    return parsed


def json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Expected a JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def software_versions() -> dict:
    def package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_names": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate vision-language models")
    parser.add_argument("--benchmark", type=benchmark_name, default="mc2")
    parser.add_argument("--model-family", choices=sorted(MODEL_DEFAULTS), default=DEFAULT_MODEL_FAMILY)
    parser.add_argument("--model", help="Model ID/path; defaults according to --model-family")
    parser.add_argument(
        "--data-file", type=Path, help="Benchmark JSON/JSONL or Parquet directory"
    )
    parser.add_argument(
        "--conflict-ratio",
        type=ratio,
        default=1.0,
        help=(
            "Fraction of evaluated samples using original conflicting context; "
            "0 uses aligned_context for all samples and 1 uses context for all samples"
        ),
    )
    parser.add_argument(
        "--image-root", type=Path, help="Benchmark image directory"
    )
    parser.add_argument(
        "--vision-answers-file",
        type=Path,
        default=DEFAULT_CORRELATIONQA_VISION_ANSWERS,
        help="Generated CorrelationQA GPT-4.1-mini vision-answer JSONL",
    )
    parser.add_argument(
        "--phd-subset",
        type=phd_subset_name,
        help="PhD context subset: phd-ica (source field sec) or phd-icc",
    )
    parser.add_argument(
        "--image-blur-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Gaussian image blur: 0=off, 1=radius 2, 2=radius 5, 3=radius 10",
    )
    parser.add_argument("--run-name", required=True, help="Unique experiment folder name")
    parser.add_argument(
        "--results-root", type=Path,
        help="Defaults to results/<benchmark>/<model-family>",
    )
    parser.add_argument("--condition", action="append", choices=[x.value for x in ConditionName])
    parser.add_argument(
        "--max-new-tokens", type=int,
        help="Defaults to 512 for CoT and 128 for all other methods",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=[item.value for item in MethodName], default="base")
    parser.add_argument(
        "--method-config",
        type=json_object,
        default={},
        help='Method-specific JSON, e.g. \'{"alpha": 1.0, "layer": 20}\'',
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    method_name = parse_method(args.method)
    if args.max_new_tokens is None:
        args.max_new_tokens = 512 if method_name is MethodName.COT else 128
    elif args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")
    args.method_config = resolve_method_config(
        method_name, args.method_config, model_family=args.model_family
    )
    if args.benchmark == "correlationqa":
        args.data_file = args.data_file or DEFAULT_CORRELATIONQA_DATA
        args.image_root = args.image_root or DEFAULT_CORRELATIONQA_IMAGES
        if args.conflict_ratio != 1.0:
            raise SystemExit("--conflict-ratio is only applicable to MC2")
        if args.condition is None:
            args.condition = [ConditionName.IMAGE_CONTEXT.value]
        supported = {
            ConditionName.IMAGE_ONLY.value,
            ConditionName.CONTEXT_ONLY.value,
            ConditionName.IMAGE_CONTEXT.value,
        }
        if any(value not in supported for value in args.condition):
            raise SystemExit(
                "CorrelationQA supports --condition image_only, context_only, or "
                "image_context"
            )
    elif args.benchmark == "phd":
        if args.phd_subset is None:
            raise SystemExit("--benchmark phd requires --phd-subset phd-ica or phd-icc")
        if args.conflict_ratio != 1.0:
            raise SystemExit("--conflict-ratio is only applicable to MC2")
        args.data_file = args.data_file or DEFAULT_PHD_DATA
        args.image_root = args.image_root or DEFAULT_PHD_ROOT
        if args.condition is None:
            args.condition = [ConditionName.IMAGE_CONTEXT.value]
        supported = {
            ConditionName.IMAGE_ONLY.value,
            ConditionName.CONTEXT_ONLY.value,
            ConditionName.IMAGE_CONTEXT.value,
        }
        if any(value not in supported for value in args.condition):
            raise SystemExit(
                "PhD supports --condition image_only, context_only, or image_context"
            )
    elif args.benchmark == "mmcd":
        if args.phd_subset is not None:
            raise SystemExit("--phd-subset is only applicable to --benchmark phd")
        if args.conflict_ratio != 1.0:
            raise SystemExit("--conflict-ratio is only applicable to MC2")
        config_root = DEFAULT_MMCD_ROOT / "default"
        args.data_file = args.data_file or config_root / "data.jsonl"
        args.image_root = args.image_root or config_root / "images"
        if args.condition is None:
            args.condition = [ConditionName.IMAGE_CONTEXT.value]
    elif args.benchmark == "clash":
        if args.phd_subset is not None:
            raise SystemExit("--phd-subset is only applicable to --benchmark phd")
        if args.conflict_ratio != 1.0:
            raise SystemExit("--conflict-ratio is only applicable to MC2")
        args.data_file = args.data_file or DEFAULT_CLASH_DATA
        args.image_root = args.image_root or DEFAULT_CLASH_IMAGES
        if args.condition is None:
            args.condition = [ConditionName.IMAGE_CONTEXT.value]
    else:
        if args.phd_subset is not None:
            raise SystemExit("--phd-subset is only applicable to --benchmark phd")
        args.data_file = args.data_file or DEFAULT_MC2_DATA
        args.image_root = args.image_root or DEFAULT_MC2_IMAGES
    selected = parse_conditions(args.condition)
    if method_name in {MethodName.CAP_IMAGE, MethodName.CAP_TEXT}:
        required_cap_condition = ConditionName.IMAGE_CONTEXT
        if any(condition.name is not required_cap_condition for condition in selected):
            raise SystemExit(
                f"{method_name.value} requires --condition {required_cap_condition.value} "
                f"for {args.benchmark}"
            )
    local_internal_methods = {
        MethodName.VCD,
        MethodName.MPS_IMAGE,
        MethodName.MPS_TEXT,
        MethodName.STEERING_IMAGE,
        MethodName.STEERING_TEXT,
        MethodName.ADAPTIVE_STEERING_IMAGE,
        MethodName.ADAPTIVE_STEERING_TEXT,
    }
    if method_name in local_internal_methods:
        if any(not condition.use_image for condition in selected):
            raise SystemExit(f"{args.method} requires an image-using condition")
    if method_name in {
        MethodName.STEERING_IMAGE,
        MethodName.STEERING_TEXT,
        MethodName.ADAPTIVE_STEERING_IMAGE,
        MethodName.ADAPTIVE_STEERING_TEXT,
    }:
        supported_steering_families = {
            "qwen3-vl-8b", "qwen2.5-vl-7b", "llava-ov-7b", "internvl3-8b"
        }
        if args.model_family not in supported_steering_families:
            supported = ", ".join(sorted(supported_steering_families))
            raise SystemExit(f"{args.method} supports these model families: {supported}")
        if any(not condition.use_context for condition in selected):
            raise SystemExit(f"{args.method} requires a context-using condition")
    args.model = args.model or MODEL_DEFAULTS[args.model_family]
    args.results_root = (
        args.results_root
        or Path("results") / args.benchmark / RESULT_SLUGS[args.model_family]
    )
    seed_everything(args.seed, False)
    run_root = args.results_root / "runs" / args.run_name
    raw_output_file = run_root / "inference" / "raw_generations.jsonl"
    predictions_file = run_root / "inference" / "predictions.jsonl"
    summary_file = args.results_root / "runs" / "performance.csv"
    manifest_file = run_root / "run_manifest.json"
    manifest = {
        "benchmark": args.benchmark,
        "run_name": args.run_name,
        "model": args.model,
        "model_family": args.model_family,
        "data_file": str(args.data_file),
        "conflict_ratio": args.conflict_ratio,
        "image_root": str(args.image_root),
        "vision_answers_file": (
            str(args.vision_answers_file)
            if args.benchmark == "correlationqa"
            else None
        ),
        "phd_subset": args.phd_subset if args.benchmark == "phd" else None,
        "mmcd_config": "default" if args.benchmark == "mmcd" else None,
        "image_blur_level": args.image_blur_level,
        "conditions": [condition.name.value for condition in selected],
        "seed": args.seed,
        "method": args.method,
        "method_config": args.method_config,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "dtype": "bfloat16",
            "device_map": args.device_map,
            "attention_implementation": "sdpa",
            "batch_size": 1,
            "deterministic_algorithms": False,
        },
        "judge_config": {
            "enabled": True,
            "parser": "hybrid",
            "model": DEFAULT_JUDGE_MODEL,
            "max_retries": 2,
            "timeout_seconds": 60.0,
        },
        "software_versions": software_versions(),
        "artifacts": {
            "raw_generations": str(raw_output_file),
            "predictions": str(predictions_file),
            "performance_csv": str(summary_file),
        },
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    if args.benchmark == "correlationqa":
        examples = load_correlationqa(
            args.data_file,
            args.image_root,
            vision_answers_file=args.vision_answers_file,
            image_blur_level=args.image_blur_level,
        )
    elif args.benchmark == "phd":
        examples = load_phd(
            args.data_file,
            args.image_root,
            args.phd_subset,
            image_blur_level=args.image_blur_level,
        )
    elif args.benchmark == "mmcd":
        examples = load_mmcd(
            args.data_file,
            args.image_root,
            image_blur_level=args.image_blur_level,
        )
    elif args.benchmark == "clash":
        examples = load_clash(
            args.data_file,
            args.image_root,
            image_blur_level=args.image_blur_level,
        )
    else:
        examples = load_mc2(
            args.data_file,
            args.image_root,
            conflict_ratio=args.conflict_ratio,
            seed=args.seed,
            image_blur_level=args.image_blur_level,
        )
    conditions = selected
    model_kwargs = dict(
        dtype="bfloat16",
        device_map=args.device_map,
        attn_implementation="sdpa",
        max_new_tokens=args.max_new_tokens,
        method=args.method,
        method_config=args.method_config,
        seed=args.seed,
    )
    model = create_model_adapter(args.model_family, args.model, **model_kwargs)
    summary = run_evaluation(
        model,
        examples,
        conditions,
        raw_output_file,
        model_name=args.model,
        seed=args.seed,
        method=args.method,
        method_config=args.method_config,
        generation_config={
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "dtype": "bfloat16",
            "device_map": args.device_map,
            "attention_implementation": "sdpa",
            "batch_size": 1,
            "deterministic_algorithms": False,
        },
        benchmark=args.phd_subset if args.benchmark == "phd" else args.benchmark,
    )
    print(json.dumps({"generated": summary}, indent=2))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    load_dotenv(".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing; set it in .env")
    client = OpenAI(max_retries=2, timeout=60.0)
    judge_config = {
        "temperature": 0,
        "max_output_tokens": 128,
        "structured_output": "ReferenceJudgeVerdict",
        "max_retries": 2,
        "timeout_seconds": 60.0,
    }
    judged_summary = run_hybrid_parsing(
        client, raw_output_file, predictions_file, model=DEFAULT_JUDGE_MODEL,
        judge_config=judge_config,
    )
    write_summary_csv(
        judged_summary,
        summary_file,
        subfolder_path=f"runs/{args.run_name}",
    )
    print(json.dumps({"llm_judge": judged_summary}, indent=2))


if __name__ == "__main__":
    main()
