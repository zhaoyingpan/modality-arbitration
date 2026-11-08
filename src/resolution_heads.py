from __future__ import annotations

from typing import Any


def rank_resolution_heads(
    rows: list[dict[str, Any]],
    *,
    model_family: str = "qwen3-vl-8b",
    top_k: int = 10,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one activation-patching row is required")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not model_family:
        raise ValueError("model_family must not be empty")
    first_vision = rows[0]["c_vision_layer_head"]
    first_text = rows[0]["c_text_layer_head"]
    num_layers = len(first_vision)
    if num_layers == 0 or len(first_text) != num_layers:
        raise ValueError("Causal-effect matrices must contain at least one layer")
    num_heads = len(first_vision[0])
    if num_heads == 0:
        raise ValueError("Causal-effect matrices must contain at least one head")
    sums = [[0.0 for _ in range(num_heads)] for _ in range(num_layers)]
    for row_index, row in enumerate(rows):
        vision = row["c_vision_layer_head"]
        text = row["c_text_layer_head"]
        if len(vision) != num_layers or len(text) != num_layers:
            raise ValueError(f"Row {row_index} has an inconsistent layer count")
        for layer in range(num_layers):
            if len(vision[layer]) != num_heads or len(text[layer]) != num_heads:
                raise ValueError(f"Row {row_index}, layer {layer} has an inconsistent head count")
            for head in range(num_heads):
                sums[layer][head] += min(
                    max(float(vision[layer][head]), 0.0),
                    max(float(text[layer][head]), 0.0),
                )

    count = len(rows)
    entries = [
        {"layer": layer, "head": head, "B": sums[layer][head] / count}
        for layer in range(num_layers)
        for head in range(num_heads)
    ]
    entries.sort(key=lambda entry: (-entry["B"], entry["layer"], entry["head"]))
    maximum = float(entries[0]["B"])
    denominator = maximum + epsilon
    selected_count = min(top_k, len(entries))
    for rank, entry in enumerate(entries, 1):
        entry["rank"] = rank
        entry["strength"] = float(entry["B"] / denominator)
        entry["selected"] = rank <= selected_count
    return {
        "schema_version": "1.0",
        "method": "steering",
        "model_family": model_family,
        "score_name": "shared_positive_effect",
        "score_definition": "B_lh = mean_samples(min(ReLU(C^T_lh), ReLU(C^V_lh)))",
        "strength_definition": "s_lh = B_lh / (max_lh(B_lh) + epsilon)",
        "n": count,
        "dataset_indices": [int(row["dataset_index"]) for row in rows],
        "num_layers": num_layers,
        "num_heads": num_heads,
        "top_k": selected_count,
        "epsilon": epsilon,
        "max_B": maximum,
        "selected_heads": [entry.copy() for entry in entries[:selected_count]],
        "ranking": entries,
    }
