#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from resolution_heads import rank_resolution_heads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank resolution heads from activation-patching C^I/C^T matrices"
    )
    parser.add_argument("samples_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--model-family", default="qwen3-vl-8b")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    return parser


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"dataset_index", "c_vision_layer_head", "c_text_layer_head"}.issubset(row):
                raise ValueError(f"{path}:{line_number} is not an activation-patching row")
            rows.append(row)
    if not rows:
        raise ValueError(f"No activation-patching rows found in {path}")
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.samples_file)
    artifact = rank_resolution_heads(
        rows,
        model_family=args.model_family,
        top_k=args.top_k,
        epsilon=args.epsilon,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output_file),
        "n": artifact["n"],
        "top_k": artifact["top_k"],
        "selected_heads": artifact["selected_heads"],
    }, indent=2))


if __name__ == "__main__":
    main()
