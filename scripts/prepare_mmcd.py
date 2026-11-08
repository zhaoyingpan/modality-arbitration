#!/usr/bin/env python
"""Download and prepare a pinned MMCD configuration for local evaluation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm


MMCD_REPO_ID = "anonymous052025/multimodal-modality-conflict-dataset"
MMCD_REVISION = "7f503e82c5a43979c56ae2ac7aaadfcd5dcbb878"
CONFIG_FILES = {
    "default": "data/train-00000-of-00001.parquet",
    "train_basic": "train_basic/train-00000-of-00001.parquet",
}
REQUIRED_COLUMNS = {"id", "caption", "image", "shape", "text_color", "image_color"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, default=Path("datasets/mmcd"))
    result.add_argument("--config", choices=sorted(CONFIG_FILES), default="default")
    result.add_argument(
        "--skip-download", action="store_true",
        help="Use the pinned source Parquet already stored under dataset-root/source.",
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_image(value: Any, *, index: int) -> Image.Image:
    if not isinstance(value, dict):
        raise ValueError(f"MMCD row {index} image must be an Arrow struct")
    image_bytes = value.get("bytes")
    if image_bytes is None:
        raise ValueError(f"MMCD row {index} image has no embedded bytes")
    with Image.open(io.BytesIO(image_bytes)) as source:
        return source.convert("RGB")


def main() -> None:
    args = parser().parse_args()
    source_relative = Path(CONFIG_FILES[args.config])
    source_file = args.dataset_root / "source" / source_relative
    output_root = args.dataset_root / args.config
    data_file = output_root / "data.jsonl"
    manifest_file = output_root / "manifest.json"
    image_root = output_root / "images"
    existing = [path for path in (data_file, manifest_file) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite prepared MMCD artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    if not source_file.is_file():
        if args.skip_download:
            raise FileNotFoundError(f"Missing pinned MMCD source: {source_file}")
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=MMCD_REPO_ID,
            filename=str(source_relative),
            repo_type="dataset",
            revision=MMCD_REVISION,
            local_dir=args.dataset_root / "source",
        )

    import pyarrow.parquet as parquet

    table = parquet.read_table(source_file)
    missing = REQUIRED_COLUMNS - set(table.column_names)
    if missing:
        raise ValueError(f"MMCD source is missing columns: {', '.join(sorted(missing))}")
    rows = table.select(sorted(REQUIRED_COLUMNS)).to_pylist()
    output_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    text_vision_conflicts = 0
    with data_file.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(tqdm(rows, desc=f"Prepare MMCD/{args.config}")):
            image_filename = f"{index:06d}.png"
            image_path = image_root / image_filename
            if image_path.exists():
                raise FileExistsError(f"Refusing to overwrite MMCD image: {image_path}")
            _extract_image(row["image"], index=index).save(image_path, format="PNG")
            text_color = str(row["text_color"])
            image_color = str(row["image_color"])
            text_vision_conflicts += text_color.casefold() != image_color.casefold()
            normalized = {
                "dataset_index": index,
                "source_index": index,
                "id": str(row["id"]),
                "caption": str(row["caption"]),
                "shape": str(row["shape"]),
                "text_color": text_color,
                "image_color": image_color,
                "image_filename": image_filename,
            }
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            handle.flush()
    manifest = {
        "schema_version": "1.0",
        "benchmark": "mmcd",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": args.config,
        "num_samples": len(rows),
        "num_text_vision_conflicts": text_vision_conflicts,
        "question_template": "what color is the {shape}?",
        "field_mapping": {
            "context": "caption",
            "answer_text": "text_color",
            "answer_vision": "image_color",
        },
        "source": {
            "repo_id": MMCD_REPO_ID,
            "revision": MMCD_REVISION,
            "file": str(source_relative),
            "sha256": _sha256(source_file),
        },
    }
    with manifest_file.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "data_file": str(data_file),
        "image_root": str(image_root),
        "manifest": str(manifest_file),
        "samples": len(rows),
        "text_vision_conflicts": text_vision_conflicts,
    }, indent=2))


if __name__ == "__main__":
    main()
