from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any


class MethodName(str, Enum):
    BASE = "base"
    CAP_IMAGE = "cap_image"
    CAP_TEXT = "cap_text"
    COT = "cot"
    VCD = "vcd"
    MPS_IMAGE = "mps_image"
    MPS_TEXT = "mps_text"
    STEERING_IMAGE = "steering_image"
    STEERING_TEXT = "steering_text"
    ADAPTIVE_STEERING_IMAGE = "adaptive_steering_image"
    ADAPTIVE_STEERING_TEXT = "adaptive_steering_text"


# Zero-based language-model decoder layer indices selected for MPS probing/steering.
MPS_DEFAULT_LAYERS = {
    "qwen2.5-vl-7b": 22,
    "qwen3-vl-8b": 28,
    "llava-ov-7b": 22,
    "internvl3-8b": 16,
}

STEERING_DEFAULT_ARTIFACTS = {
    "qwen3-vl-8b": "artifacts/steering/qwen3_vl_8b_resolution_heads_top10_n0012.json",
    "qwen2.5-vl-7b": "artifacts/steering/qwen2_5_vl_7b_resolution_heads_top10_n0010.json",
    "llava-ov-7b": "artifacts/steering/llava_ov_7b_resolution_heads_top10_n0010.json",
    "internvl3-8b": "artifacts/steering/internvl3_8b_resolution_heads_top10_n0010.json",
}


COT_INSTRUCTION = (
    "Let's think step by step, then answer the question with a short answer."
)
DIRECT_ANSWER_INSTRUCTION = "Answer the question directly with a short answer."
CAP_INSTRUCTIONS = {
    "image": "Answer the question based on the image",
    "text": "Answer the question based on the text",
}
CAP_METHOD_TARGETS = {
    MethodName.CAP_IMAGE: "image",
    MethodName.CAP_TEXT: "text",
}

MPS_VISION_PROBE = (
    "In case there is an inconsistency between the text context and the image content, "
    "you should follow the image content rather than the text context."
)
MPS_TEXT_PROBE = (
    "In case there is an inconsistency between the text context and the image content, "
    "you should follow the text context rather than the image content."
)


def build_mps_probe_prompt(question: str, context: str, instruction: str) -> str:
    """Match the reference MPS order while retaining open-ended answer guidance."""
    return "\n".join((
        f"Text context: {context}",
        instruction,
        question,
        DIRECT_ANSWER_INSTRUCTION,
    ))


def parse_method(value: str) -> MethodName:
    try:
        return MethodName(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in MethodName)
        raise ValueError(f"Unknown method {value!r}; choose one of: {choices}") from error


def resolve_method_config(
    method: MethodName, config: dict[str, Any], model_family: str | None = None
) -> dict[str, Any]:
    resolved = dict(config)
    if method in CAP_METHOD_TARGETS:
        if resolved:
            raise ValueError(f"{method.value} does not accept method_config")
    elif method is MethodName.VCD:
        defaults = {"alpha": 1.0, "beta": 0.1, "gamma": 0.1, "noise_steps": 500}
        defaults.update(resolved)
        resolved = defaults
        if resolved["alpha"] < 0:
            raise ValueError("VCD alpha must be non-negative")
        if not 0 < resolved["beta"] <= 1:
            raise ValueError("VCD beta must be in (0, 1]")
        if not 0 < resolved["gamma"] < 1:
            raise ValueError("VCD gamma must be in (0, 1)")
        if int(resolved["noise_steps"]) < 1:
            raise ValueError("VCD noise_steps must be positive")
    elif method in {MethodName.MPS_IMAGE, MethodName.MPS_TEXT}:
        if "vector_path" not in resolved:
            raise ValueError(
                f"{method.value} requires method_config.vector_path from build_mps_vector"
            )
        resolved["vector_path"] = str(Path(resolved["vector_path"]))
        resolved.setdefault("scale", 1.0)
    elif method in {
        MethodName.STEERING_IMAGE,
        MethodName.STEERING_TEXT,
        MethodName.ADAPTIVE_STEERING_IMAGE,
        MethodName.ADAPTIVE_STEERING_TEXT,
    }:
        artifact_path = STEERING_DEFAULT_ARTIFACTS.get(
            model_family or "qwen3-vl-8b",
            STEERING_DEFAULT_ARTIFACTS["qwen3-vl-8b"],
        )
        defaults = {
            "artifact_path": artifact_path,
            "top_k": 10,
            "alpha": 1.0,
            "epsilon": 1e-12,
            "suppress_opposite": False,
            "apply_to": "all",
        }
        defaults.update(resolved)
        resolved = defaults
        if method in {
            MethodName.ADAPTIVE_STEERING_IMAGE,
            MethodName.ADAPTIVE_STEERING_TEXT,
        }:
            if "threshold_path" not in resolved:
                raise ValueError(f"{method.value} requires method_config.threshold_path")
            resolved["threshold_path"] = str(Path(resolved["threshold_path"]))
            resolved.setdefault("alpha_max", 50.0)
            resolved.setdefault("lambda", 0.05)
            resolved["alpha_max"] = float(resolved["alpha_max"])
            resolved["lambda"] = float(resolved["lambda"])
            if not math.isfinite(resolved["alpha_max"]) or resolved["alpha_max"] < 0:
                raise ValueError("Adaptive steering alpha_max must be finite and non-negative")
            if not math.isfinite(resolved["lambda"]) or resolved["lambda"] <= 0:
                raise ValueError("Adaptive steering lambda must be finite and positive")
            # Adaptive steering is defined here with opposite-modality suppression.
            resolved["alpha"] = 0.0
            resolved["suppress_opposite"] = True
        resolved["artifact_path"] = str(Path(resolved["artifact_path"]))
        if isinstance(resolved["top_k"], bool) or int(resolved["top_k"]) != resolved["top_k"]:
            raise ValueError("Steering top_k must be an integer")
        resolved["top_k"] = int(resolved["top_k"])
        resolved["alpha"] = float(resolved["alpha"])
        resolved["epsilon"] = float(resolved["epsilon"])
        if resolved["top_k"] < 1:
            raise ValueError("Steering top_k must be positive")
        if math.isnan(resolved["alpha"]) or resolved["alpha"] < 0:
            raise ValueError("Steering alpha must be non-negative or positive infinity")
        if not math.isfinite(resolved["epsilon"]) or resolved["epsilon"] <= 0:
            raise ValueError("Steering epsilon must be finite and positive")
        if not isinstance(resolved["suppress_opposite"], bool):
            raise ValueError("Steering suppress_opposite must be true or false")
        if resolved["apply_to"] not in {"all", "generation"}:
            raise ValueError("Steering apply_to must be 'all' or 'generation'")
    return resolved


def transform_prompt(
    prompt: str, method: MethodName, method_config: dict[str, Any] | None = None
) -> str:
    if method in CAP_METHOD_TARGETS:
        modality = CAP_METHOD_TARGETS[method]
        lines = prompt.splitlines()
        if lines and lines[-1] == DIRECT_ANSWER_INSTRUCTION and len(lines) >= 2:
            lines.insert(len(lines) - 2, CAP_INSTRUCTIONS[modality])
        else:
            lines.insert(0, CAP_INSTRUCTIONS[modality])
        return "\n".join(lines)
    if method is MethodName.COT:
        lines = prompt.splitlines()
        if lines and lines[-1] == DIRECT_ANSWER_INSTRUCTION:
            lines[-1] = COT_INSTRUCTION
            return "\n".join(lines)
        return f"{prompt}\n{COT_INSTRUCTION}"
    return prompt
