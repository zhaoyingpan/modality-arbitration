from __future__ import annotations

from typing import Any

from .base import ModelAdapter

MODEL_DEFAULTS = {
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "llava-ov-7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "internvl3-8b": "OpenGVLab/InternVL3-8B-hf",
}

RESULT_SLUGS = {
    "qwen2.5-vl-7b": "qwen2_5_vl_7b",
    "qwen3-vl-8b": "qwen3_vl_8b",
    "llava-ov-7b": "llava_ov_7b",
    "internvl3-8b": "internvl3_8b",
}


def create_model_adapter(family: str, model_path: str, **kwargs: Any) -> ModelAdapter:
    from .hf_vlm import HuggingFaceVLMAdapter
    return HuggingFaceVLMAdapter(model_path, get_hf_model_class(family), **kwargs)


def get_hf_model_class(family: str) -> type:
    from transformers import (
        InternVLForConditionalGeneration,
        LlavaOnevisionForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )
    classes = {
        "qwen2.5-vl-7b": Qwen2_5_VLForConditionalGeneration,
        "qwen3-vl-8b": Qwen3VLForConditionalGeneration,
        "llava-ov-7b": LlavaOnevisionForConditionalGeneration,
        "internvl3-8b": InternVLForConditionalGeneration,
    }
    if family not in classes:
        raise ValueError(f"Unsupported model family: {family}")
    return classes[family]
