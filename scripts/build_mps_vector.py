#!/usr/bin/env python
"""Build the cached modality-preference steering vector used by MPS runs."""
from __future__ import annotations

import argparse
import inspect
import json
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from tqdm import tqdm

from data import load_mc2
from models.hf_vlm import HuggingFaceVLMAdapter, _language_layers
from models.registry import MODEL_DEFAULTS, create_model_adapter
from methods import (
    MPS_DEFAULT_LAYERS,
    MPS_TEXT_PROBE,
    MPS_VISION_PROBE,
    build_mps_probe_prompt,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Probe and cache an MPS steering vector")
    result.add_argument("--model-family", choices=sorted(MODEL_DEFAULTS), required=True)
    result.add_argument("--model")
    result.add_argument("--data-file", type=Path, default=Path("datasets/mc2/evaluate_final_aligned.json"))
    result.add_argument("--image-root", type=Path, default=Path("datasets/mc2/noise_images/coco_noise_gs"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--layer", type=int,
        help="Zero-based decoder layer; defaults to the configured model-family MPS layer",
    )
    result.add_argument("--num-probes", type=int, default=500)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--dtype", default="bfloat16")
    result.add_argument("--device-map", default="auto")
    result.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    return result


def main() -> None:
    args = parser().parse_args()
    offline_started_at = perf_counter()
    if args.layer is None:
        if args.model_family not in MPS_DEFAULT_LAYERS:
            raise SystemExit(
                f"No default MPS layer is configured for {args.model_family}; pass --layer"
            )
        args.layer = MPS_DEFAULT_LAYERS[args.model_family]
    model_name = args.model or MODEL_DEFAULTS[args.model_family]
    adapter = create_model_adapter(
        args.model_family, model_name, dtype=args.dtype, device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    if not isinstance(adapter, HuggingFaceVLMAdapter):
        raise TypeError("MPS requires HuggingFaceVLMAdapter")
    layers = _language_layers(adapter.model)
    if not 0 <= args.layer < len(layers):
        raise SystemExit(f"layer must be in [0, {len(layers)})")
    examples = load_mc2(args.data_file, args.image_root, args.num_probes, seed=args.seed)
    directions: list[torch.Tensor] = []
    text_norms: list[float] = []
    for example in tqdm(examples, desc="MPS probing"):
        prompts = [
            build_mps_probe_prompt(example.question, example.context, MPS_VISION_PROBE),
            build_mps_probe_prompt(example.question, example.context, MPS_TEXT_PROBE),
        ]
        with ExitStack() as stack:
            source = stack.enter_context(Image.open(example.image_path))
            image = source.convert("RGB")
            stack.callback(image.close)
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": item}
            ]}] for item in prompts]
            texts = [adapter.processor.apply_chat_template(
                item, tokenize=False, add_generation_prompt=True
            ) for item in messages]
            encoded = adapter.processor(
                text=texts, images=[image, image], padding=True, return_tensors="pt"
            ).to(adapter.model.device)
            forward_kwargs = {"output_hidden_states": True, "use_cache": False}
            # Some VLM wrappers otherwise project every prompt position over the full
            # vocabulary. Probing only consumes hidden states, so retaining one logit
            # position avoids a large, unnecessary peak-memory allocation.
            if "logits_to_keep" in inspect.signature(adapter.model.forward).parameters:
                forward_kwargs["logits_to_keep"] = 1
            with torch.inference_mode():
                outputs = adapter.model(**encoded, **forward_kwargs)
            state = outputs.hidden_states[args.layer + 1]
            positions = torch.arange(
                encoded.attention_mask.shape[1], device=encoded.attention_mask.device
            ).expand_as(encoded.attention_mask)
            last_indices = positions.masked_fill(encoded.attention_mask == 0, -1).max(dim=1).values
            probe_states = state[torch.arange(2, device=state.device), last_indices]
            vision_state, text_state = probe_states[0].float(), probe_states[1].float()
            directions.append((text_state - vision_state).cpu())
            text_norms.append(float(text_state.norm().item()))
    direction = torch.stack(directions).mean(dim=0)
    weight = (sum(text_norms) / len(text_norms)) / float(direction.norm().item())
    offline_steering_vector_elapsed_s = perf_counter() - offline_started_at
    artifact = {
        "schema_version": "1.0", "method": "mps", "model": model_name,
        "model_family": args.model_family, "layer": args.layer,
        "num_probes": len(directions), "seed": args.seed,
        "vision_probe": MPS_VISION_PROBE, "text_probe": MPS_TEXT_PROBE,
        "probe_prompt_order": "context_instruction_question_answer_guidance",
        "weight": weight, "vector": direction.mul(weight),
        "offline_steering_vector_elapsed_s": offline_steering_vector_elapsed_s,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    torch.save(artifact, args.output)
    print(json.dumps({key: value for key, value in artifact.items() if key != "vector"}, indent=2))


if __name__ == "__main__":
    main()
