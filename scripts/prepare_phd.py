#!/usr/bin/env python
"""Download PhD metadata/images and create the fixed 1,000-question evaluation split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm


PHD_REPO_ID = "AIMClab-RUC/PhD"
PHD_REVISION = "04240516cacf90fee2804ab477f494a0db94cb5a"
PHD_TASKS = ("attribute", "sentiment", "positional", "counting", "object")
SOURCE_CONTEXT_KEYS = {"phd-ica": "sec", "phd-icc": "icc"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, default=Path("datasets/phd"))
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--samples-per-task", type=int, default=200)
    result.add_argument("--max-download-workers", type=int, default=8)
    result.add_argument(
        "--skip-download", action="store_true",
        help="Require metadata and selected images locally; do not access Hugging Face.",
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixed_samples(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    samples_per_task: int = 200,
) -> list[dict[str, Any]]:
    """Stratify by task and question polarity, yielding two questions per source row."""
    if samples_per_task < 2 or samples_per_task % 2:
        raise ValueError("samples_per_task must be a positive even number")
    sources_per_task = samples_per_task // 2
    grouped: dict[str, list[int]] = defaultdict(list)
    for source_index, row in enumerate(rows):
        if "ccs_description" in row:
            continue
        task = str(row.get("task", ""))
        if task in PHD_TASKS:
            grouped[task].append(source_index)
    rng = random.Random(seed)
    selected: list[tuple[str, int]] = []
    for task in PHD_TASKS:
        candidates = grouped[task]
        if len(candidates) < sources_per_task:
            raise ValueError(
                f"PhD task {task} has {len(candidates)} sources; need {sources_per_task}"
            )
        selected.extend((task, index) for index in rng.sample(candidates, sources_per_task))

    samples: list[dict[str, Any]] = []
    for task, source_index in selected:
        row = rows[source_index]
        contexts = row.get("context")
        if not isinstance(contexts, dict):
            raise ValueError(f"PhD source row {source_index} has no context object")
        missing_contexts = set(SOURCE_CONTEXT_KEYS.values()) - contexts.keys()
        if missing_contexts:
            raise ValueError(
                f"PhD source row {source_index} is missing contexts: "
                f"{', '.join(sorted(missing_contexts))}"
            )
        for polarity, answer in (("yes", "yes"), ("no", "no")):
            samples.append({
                "dataset_index": len(samples),
                "source_index": source_index,
                "question_polarity": polarity,
                "image_id": str(row["image_id"]).zfill(12),
                "task": task,
                "question": str(row[f"{polarity}_question"]),
                "answer_vision": answer,
                "contexts": {
                    subset: str(contexts[source_key])
                    for subset, source_key in SOURCE_CONTEXT_KEYS.items()
                },
            })
    return samples


def _download_metadata(source_root: Path, *, skip_download: bool) -> Path:
    data_file = source_root / "data.json"
    if data_file.is_file():
        return data_file
    if skip_download:
        raise FileNotFoundError(f"Missing PhD metadata: {data_file}")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=PHD_REPO_ID,
        repo_type="dataset",
        revision=PHD_REVISION,
        allow_patterns=["data.json"],
        local_dir=source_root,
    )
    return data_file


def _download_and_resolve_images(
    samples: list[dict[str, Any]],
    source_root: Path,
    *,
    skip_download: bool,
    max_workers: int,
) -> None:
    image_ids = sorted({str(sample["image_id"]) for sample in samples})
    def resolve_one(image_id: str) -> tuple[str, str]:
        candidates = (
            Path("images/train2014") / f"COCO_train2014_{image_id}.jpg",
            Path("images/val2014") / f"COCO_val2014_{image_id}.jpg",
        )
        matches = [path for path in candidates if (source_root / path).is_file()]
        if not matches and not skip_download:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError
            for candidate in candidates:
                try:
                    hf_hub_download(
                        repo_id=PHD_REPO_ID,
                        filename=str(candidate),
                        repo_type="dataset",
                        revision=PHD_REVISION,
                        local_dir=source_root,
                    )
                    break
                except EntryNotFoundError:
                    continue
            matches = [path for path in candidates if (source_root / path).is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one train/val image for PhD image_id={image_id}; "
                f"found {len(matches)}"
            )
        return image_id, str(matches[0])

    resolved: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(resolve_one, image_id): image_id for image_id in image_ids}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Download/validate PhD images"
        ):
            image_id, path = future.result()
            resolved[image_id] = path
    for sample in samples:
        sample["image_path"] = resolved[str(sample["image_id"])]


def main() -> None:
    args = parser().parse_args()
    if args.samples_per_task != 200:
        raise SystemExit(
            "The project-fixed PhD evaluation requires --samples-per-task 200 "
            "(1,000 questions per subset)"
        )
    if args.max_download_workers < 1:
        raise SystemExit("--max-download-workers must be positive")
    source_root = args.dataset_root / "source"
    output_file = args.dataset_root / f"evaluation_seed{args.seed}.json"
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite fixed split: {output_file}")
    data_file = _download_metadata(source_root, skip_download=args.skip_download)
    with data_file.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {data_file}")
    samples = build_fixed_samples(
        rows, seed=args.seed, samples_per_task=args.samples_per_task
    )
    _download_and_resolve_images(
        samples,
        source_root,
        skip_download=args.skip_download,
        max_workers=args.max_download_workers,
    )
    task_counts = Counter(sample["task"] for sample in samples)
    polarity_counts = Counter(sample["question_polarity"] for sample in samples)
    payload = {
        "schema_version": "1.0",
        "benchmark": "phd",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": PHD_REPO_ID,
            "revision": PHD_REVISION,
            "data_file": str(data_file),
            "data_sha256": _sha256(data_file),
            "source_context_mapping": SOURCE_CONTEXT_KEYS,
            "naming_note": (
                "The requested/paper prose name PhD-ica maps to the pinned source's "
                "specious-context field `sec`; the paper also calls it PhD-iac."
            ),
        },
        "sampling": {
            "seed": args.seed,
            "algorithm": "python_random_stratified_source_rows_then_both_question_polarities",
            "tasks": list(PHD_TASKS),
            "samples_per_task": args.samples_per_task,
            "source_rows_per_task": args.samples_per_task // 2,
            "question_polarities_per_source": ["yes", "no"],
            "sample_count_per_subset": len(samples),
            "task_counts": dict(task_counts),
            "polarity_counts": dict(polarity_counts),
        },
        "available_subsets": list(SOURCE_CONTEXT_KEYS),
        "samples": samples,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(output_file),
        "samples_per_subset": len(samples),
        "unique_images": len({sample["image_id"] for sample in samples}),
        "task_counts": dict(task_counts),
        "polarity_counts": dict(polarity_counts),
    }, indent=2))


if __name__ == "__main__":
    main()
