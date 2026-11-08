#!/usr/bin/env python
"""Prepare the pinned CLASH test split and only its referenced COCO images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


CLASH_REPO_ID = "tpopordanoska/clash"
CLASH_REVISION = "d9609c3178be1ab87151ce3668703f0eb2e7d6a6"
CLASH_DATA_FILE = "data/synthetic_captions_test_filtered.json"
CLASH_DATA_URL = (
    f"https://raw.githubusercontent.com/{CLASH_REPO_ID}/{CLASH_REVISION}/{CLASH_DATA_FILE}"
)
COCO_TRAIN2017_URL = "http://images.cocodataset.org/train2017"
EXPECTED_SAMPLES = 1289
EXPECTED_UNIQUE_IMAGES = 1281


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, default=Path("datasets/clash"))
    result.add_argument("--max-download-workers", type=int, default=8)
    result.add_argument("--max-retries", type=int, default=3)
    result.add_argument(
        "--skip-download", action="store_true",
        help="Validate the pinned JSON and selected images without network access.",
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_image_ids(payload: dict[str, Any]) -> list[int]:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("CLASH data must contain a samples list")
    if len(samples) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Pinned CLASH split contains {len(samples)} samples; expected {EXPECTED_SAMPLES}"
        )
    ids = sorted({int(sample["image_id"]) for sample in samples})
    if len(ids) != EXPECTED_UNIQUE_IMAGES:
        raise ValueError(
            f"Pinned CLASH split references {len(ids)} unique images; "
            f"expected {EXPECTED_UNIQUE_IMAGES}"
        )
    return ids


def _request_bytes(url: str, *, max_retries: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "modality-conflict/0.1"})
    error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as caught:
            error = caught
            if attempt + 1 < max_retries:
                time.sleep(2 ** attempt)
    assert error is not None
    raise error


def _download_atomic(url: str, target: Path, *, max_retries: int) -> None:
    data = _request_bytes(url, max_retries=max_retries)
    partial = target.with_suffix(target.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as handle:
        handle.write(data)
    os.replace(partial, target)


def _validate_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def main() -> None:
    args = parser().parse_args()
    if args.max_download_workers < 1:
        raise SystemExit("--max-download-workers must be positive")
    if args.max_retries < 1:
        raise SystemExit("--max-retries must be positive")
    data_file = args.dataset_root / "synthetic_captions_test_filtered.json"
    image_root = args.dataset_root / "images" / "train2017"
    manifest_file = args.dataset_root / "manifest.json"
    if manifest_file.exists():
        raise FileExistsError(f"Refusing to overwrite CLASH manifest: {manifest_file}")
    if not data_file.is_file():
        if args.skip_download:
            raise FileNotFoundError(f"Missing CLASH test data: {data_file}")
        data_file.parent.mkdir(parents=True, exist_ok=True)
        _download_atomic(CLASH_DATA_URL, data_file, max_retries=args.max_retries)
    with data_file.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {data_file}")
    image_ids = required_image_ids(payload)

    def prepare_image(image_id: int) -> Path:
        filename = f"{image_id:012d}.jpg"
        path = image_root / filename
        if not path.is_file():
            if args.skip_download:
                raise FileNotFoundError(f"Missing selected CLASH image: {path}")
            _download_atomic(
                f"{COCO_TRAIN2017_URL}/{filename}", path,
                max_retries=args.max_retries,
            )
        _validate_image(path)
        return path

    prepared: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.max_download_workers) as executor:
        futures = {executor.submit(prepare_image, image_id): image_id for image_id in image_ids}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Download/validate CLASH images"
        ):
            prepared.append(future.result())
    manifest = {
        "schema_version": "1.0",
        "benchmark": "clash",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": CLASH_REPO_ID,
            "revision": CLASH_REVISION,
            "file": CLASH_DATA_FILE,
            "url": CLASH_DATA_URL,
            "sha256": _sha256(data_file),
            "coco_split": "train2017",
            "coco_image_base_url": COCO_TRAIN2017_URL,
        },
        "samples": len(payload["samples"]),
        "unique_images": len(prepared),
        "download_scope": "only_images_referenced_by_synthetic_captions_test_filtered",
        "field_mapping": {
            "context": "conflicting_caption",
            "question": "question",
            "answer_vision": "answers.image_only",
            "answer_text": "answers.text_only",
        },
        "evaluation_note": (
            "This project reports modality-preference accuracy against image_only and "
            "text_only separately; this differs from CLASH's primary conflict-detection score."
        ),
        "image_bytes": sum(path.stat().st_size for path in prepared),
    }
    with manifest_file.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "data_file": str(data_file),
        "image_root": str(image_root),
        "manifest": str(manifest_file),
        "samples": len(payload["samples"]),
        "unique_images": len(prepared),
        "image_bytes": manifest["image_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
