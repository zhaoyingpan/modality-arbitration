#!/usr/bin/env python3
"""Create an MC2 diagnosis subset with single-token reference answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("datasets/mc2/evaluate_final_aligned.json")
DEFAULT_OUTPUT = Path("datasets/mc2/evaluate_final_one_token.json")
ANSWER_FIELDS = ("answer_vision", "answer_text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file instead of refusing to overwrite it.",
    )
    parser.add_argument(
        "--match",
        choices=("both", "any"),
        default="both",
        help=(
            "Keep rows where both answers are single-token (default), or where "
            "at least one answer is single-token."
        ),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a JSON list of dictionaries")
    for index, row in enumerate(rows):
        missing = set(ANSWER_FIELDS) - row.keys()
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
        for field in ANSWER_FIELDS:
            if not isinstance(row[field], str):
                raise TypeError(f"Row {index} field {field!r} must be a string")
    return rows


def is_single_token(value: str) -> bool:
    """Use a transparent, model-independent whitespace token definition."""
    return len(value.strip().split()) == 1


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    predicate = all if args.match == "both" else any
    selected = [
        row
        for row in rows
        if predicate(is_single_token(row[field]) for field in ANSWER_FIELDS)
    ]

    if args.output.resolve() == args.input.resolve():
        raise ValueError("Output must differ from input; the source dataset is not overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with args.output.open(mode, encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"Wrote {len(selected)} of {len(rows)} rows to {args.output} "
        f"(match={args.match}, whitespace-tokenized)"
    )


if __name__ == "__main__":
    main()
