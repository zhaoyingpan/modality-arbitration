from __future__ import annotations

import json
import math
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F
from PIL import Image, ImageFilter
from transformers import AutoProcessor
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.generation.streamers import BaseStreamer

from .base import GenerationBatch, ModelAdapter, ModelInput
from .attention_steering import (
    AttentionSteeringController,
    context_token_mask,
    install_attention_steering,
)
from methods import MethodName, parse_method


def _language_layers(model: Any) -> Any:
    candidates = (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
    )
    for path in candidates:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
            return value
        except AttributeError:
            continue
    raise RuntimeError("Could not locate the language-model decoder layers for MPS")


class _VCDLogitsProcessor(LogitsProcessor):
    """Equation 3 plus the adaptive plausibility constraint from VCD."""

    def __init__(self, model: Any, distorted: dict[str, Any], alpha: float, beta: float):
        self.model = model
        self.distorted = distorted
        self.alpha = alpha
        self.log_beta = float(torch.log(torch.tensor(beta)))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        kwargs = {
            key: value for key, value in self.distorted.items()
            if key not in {"input_ids", "attention_mask"}
        }
        kwargs["input_ids"] = input_ids
        kwargs["attention_mask"] = torch.ones_like(input_ids)
        type_key = next(
            (key for key in ("mm_token_type_ids", "token_type_ids") if key in kwargs), None
        )
        token_type_ids = kwargs.get(type_key) if type_key is not None else None
        if token_type_ids is not None and token_type_ids.shape[-1] != input_ids.shape[-1]:
            generated_length = input_ids.shape[-1] - token_type_ids.shape[-1]
            if generated_length < 0:
                raise RuntimeError("VCD token_type_ids are longer than the decoding sequence")
            # Newly decoded positions are ordinary text tokens. Qwen3-VL requires
            # token_type_ids to stay aligned with input_ids when recomputing 3-D RoPE.
            text_types = torch.zeros(
                (*token_type_ids.shape[:-1], generated_length),
                dtype=token_type_ids.dtype,
                device=token_type_ids.device,
            )
            kwargs[type_key] = torch.cat((token_type_ids, text_types), dim=-1)
        with torch.inference_mode():
            distorted_scores = self.model(**kwargs, use_cache=False).logits[:, -1, :]
        contrasted = (1.0 + self.alpha) * scores - self.alpha * distorted_scores
        log_probs = torch.log_softmax(scores, dim=-1)
        plausible = log_probs >= (log_probs.max(dim=-1, keepdim=True).values + self.log_beta)
        return contrasted.masked_fill(~plausible, -torch.inf)


class _FirstTokenTimer(BaseStreamer):
    def __init__(self, started_at: float) -> None:
        self.started_at = started_at
        self._prompt_seen = False
        self.ttft_seconds: float | None = None

    def put(self, value: Any) -> None:
        if not self._prompt_seen:
            self._prompt_seen = True
        elif self.ttft_seconds is None:
            self.ttft_seconds = perf_counter() - self.started_at

    def end(self) -> None:
        pass


class HuggingFaceVLMAdapter(ModelAdapter):
    """Chat-template adapter for native Transformers vision-language models."""

    def __init__(
        self,
        model_path: str,
        model_class: type,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str | None = None,
        max_new_tokens: int = 32,
        trust_remote_code: bool = False,
        method: str = "base",
        method_config: dict[str, Any] | None = None,
        seed: int = 42,
        **_: Any,
    ) -> None:
        if dtype == "auto":
            torch_dtype: Any = "auto"
        else:
            try:
                torch_dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"Unknown torch dtype: {dtype}") from error
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = model_class.from_pretrained(model_path, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.max_new_tokens = max_new_tokens
        self.method = parse_method(method)
        self.method_config = method_config or {}
        self.seed = seed
        self._mps_vector: torch.Tensor | None = None
        self._mps_layer: int | None = None
        self._attention_steering: AttentionSteeringController | None = None
        self._uninstall_attention_steering: Any = None
        self._adaptive_tau: float | None = None
        if self.method in {MethodName.MPS_IMAGE, MethodName.MPS_TEXT}:
            self._load_mps_vector()
        elif self.method in {
            MethodName.STEERING_IMAGE,
            MethodName.STEERING_TEXT,
            MethodName.ADAPTIVE_STEERING_IMAGE,
            MethodName.ADAPTIVE_STEERING_TEXT,
        }:
            direction = (
                "image"
                if self.method in {
                    MethodName.STEERING_IMAGE,
                    MethodName.ADAPTIVE_STEERING_IMAGE,
                }
                else "text"
            )
            if self.method in {
                MethodName.ADAPTIVE_STEERING_IMAGE,
                MethodName.ADAPTIVE_STEERING_TEXT,
            }:
                with Path(self.method_config["threshold_path"]).open(encoding="utf-8") as handle:
                    threshold_artifact = json.load(handle)
                self._adaptive_tau = float(threshold_artifact["threshold"])
                if not math.isfinite(self._adaptive_tau):
                    raise ValueError("Adaptive steering threshold must be finite")
            self._attention_steering = AttentionSteeringController.from_artifact(
                self.method_config["artifact_path"],
                direction=direction,
                top_k=int(self.method_config["top_k"]),
                alpha=(
                    0.0 if self._adaptive_tau is not None
                    else float(self.method_config["alpha"])
                ),
                epsilon=float(self.method_config["epsilon"]),
                suppress_opposite=bool(self.method_config["suppress_opposite"]),
                apply_to=str(self.method_config["apply_to"]),
            )
            self._uninstall_attention_steering = install_attention_steering(
                self.model, self._attention_steering
            )

    def _load_mps_vector(self) -> None:
        artifact = torch.load(
            self.method_config["vector_path"], map_location="cpu", weights_only=True
        )
        if not isinstance(artifact, dict) or "vector" not in artifact or "layer" not in artifact:
            raise ValueError("MPS artifact must contain 'vector' and 'layer'")
        vector = artifact["vector"]
        if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
            raise ValueError("MPS vector must be a one-dimensional tensor")
        self._mps_vector = vector
        self._mps_layer = int(artifact["layer"])
        layers = _language_layers(self.model)
        if not 0 <= self._mps_layer < len(layers):
            raise ValueError(f"MPS layer {self._mps_layer} is outside [0, {len(layers)})")

    def _mps_hook(self, module: Any, args: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        assert self._mps_vector is not None
        sign = 1.0 if self.method is MethodName.MPS_TEXT else -1.0
        delta = self._mps_vector.to(device=hidden.device, dtype=hidden.dtype)
        delta = delta * sign * float(self.method_config.get("scale", 1.0))
        changed = hidden + delta.view(1, 1, -1)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def _distort_images(self, images: list[Image.Image]) -> list[Image.Image]:
        gamma = float(self.method_config["gamma"])
        steps = int(self.method_config["noise_steps"])
        signal = (1.0 - gamma) ** (steps / 2.0)
        noise_scale = (1.0 - signal * signal) ** 0.5
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        distorted: list[Image.Image] = []
        for image in images:
            pixels = torch.from_numpy(__import__("numpy").array(image)).float() / 255.0
            noise = torch.randn(pixels.shape, generator=generator)
            pixels = (signal * pixels + noise_scale * noise).clamp(0, 1)
            array = (pixels.mul(255).byte().numpy())
            distorted.append(Image.fromarray(array, mode="RGB"))
        return distorted

    def _unimodal_similarities(
        self, inputs: list[ModelInput], images: list[Image.Image]
    ) -> torch.Tensor:
        if any(item.question is None or item.context is None for item in inputs):
            raise ValueError("Adaptive steering requires question and context for every sample")
        direct = "Answer the question directly with a short answer."
        image_messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{item.question}\n{direct}"},
            ]}]
            for item, image in zip(inputs, images)
        ]
        text_messages = [
            [{"role": "user", "content": [{
                "type": "text",
                "text": f"Text context: {item.context}\n{item.question}\n{direct}",
            }]}]
            for item in inputs
        ]

        def next_token_logits(messages: list[list[dict[str, Any]]], probe_images=None):
            prompts = [
                self.processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in messages
            ]
            kwargs: dict[str, Any] = {
                "text": prompts, "padding": True, "return_tensors": "pt"
            }
            if probe_images is not None:
                kwargs["images"] = probe_images
            encoded_probe = self.processor(**kwargs).to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**encoded_probe, use_cache=False).logits
            rows = []
            for batch_index, attention in enumerate(encoded_probe.attention_mask):
                positions = torch.nonzero(attention, as_tuple=False).flatten()
                if positions.numel() == 0:
                    raise RuntimeError("Adaptive steering probe produced an empty prompt")
                rows.append(logits[batch_index, positions[-1]].float())
            return torch.stack(rows)

        image_logits = next_token_logits(image_messages, images)
        text_logits = next_token_logits(text_messages)
        similarities: list[torch.Tensor] = []
        top_k = min(50, image_logits.shape[-1])
        for image_row, text_row in zip(image_logits, text_logits):
            # Use the union of both top-50 ID sets and gather each missing side's
            # actual corresponding logit from its full vocabulary distribution.
            token_ids = torch.unique(torch.cat((
                image_row.topk(top_k).indices,
                text_row.topk(top_k).indices,
            )))
            image_vector = F.normalize(image_row[token_ids], p=2, dim=0)
            text_vector = F.normalize(text_row[token_ids], p=2, dim=0)
            similarities.append(torch.dot(image_vector, text_vector))
        return torch.stack(similarities)

    def compute_unimodal_similarities(self, inputs: list[ModelInput]) -> list[float]:
        """Compute adaptive-steering probe similarity without generating answers."""
        with ExitStack() as stack:
            images: list[Image.Image] = []
            for item in inputs:
                if item.image_path is None:
                    raise ValueError("Adaptive steering similarity requires an image")
                source = stack.enter_context(Image.open(item.image_path))
                image = source.convert("RGB")
                if item.image_blur_radius > 0:
                    blurred = image.filter(ImageFilter.GaussianBlur(item.image_blur_radius))
                    image.close()
                    image = blurred
                stack.callback(image.close)
                images.append(image)
            return self._unimodal_similarities(inputs, images).cpu().tolist()

    def generate(self, inputs: list[ModelInput]) -> GenerationBatch:
        if not inputs:
            return GenerationBatch([], {})
        has_images = [item.image_path is not None for item in inputs]
        if any(has_images) and not all(has_images):
            raise ValueError("A batch must not mix image and text-only inputs")
        end_to_end_started_at = perf_counter()
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        for index in cuda_devices:
            torch.cuda.reset_peak_memory_stats(index)
        with ExitStack() as stack:
            messages: list[list[dict[str, Any]]] = []
            images: list[Image.Image] = []
            for item in inputs:
                content: list[dict[str, Any]] = []
                if item.image_path is not None:
                    source = stack.enter_context(Image.open(item.image_path))
                    image = source.convert("RGB")
                    if item.image_blur_radius > 0:
                        blurred = image.filter(ImageFilter.GaussianBlur(item.image_blur_radius))
                        image.close()
                        image = blurred
                    stack.callback(image.close)
                    images.append(image)
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": item.prompt})
                messages.append([{"role": "user", "content": content}])

            adaptive_similarities: torch.Tensor | None = None
            adaptive_alphas: torch.Tensor | None = None
            adaptive_probe_latency = None
            if self._adaptive_tau is not None:
                adaptive_started_at = perf_counter()
                adaptive_similarities = self._unimodal_similarities(inputs, images)
                adaptive_alphas = float(self.method_config["alpha_max"]) * torch.sigmoid(
                    (self._adaptive_tau - adaptive_similarities)
                    / float(self.method_config["lambda"])
                )
                if cuda_devices:
                    torch.cuda.synchronize()
                adaptive_probe_latency = perf_counter() - adaptive_started_at

            prompts = [
                self.processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in messages
            ]
            processor_kwargs: dict[str, Any] = {
                "text": prompts,
                "padding": True,
                "return_tensors": "pt",
            }
            if images:
                processor_kwargs["images"] = images
            encoded = self.processor(**processor_kwargs).to(self.model.device)
            if self._attention_steering is not None:
                configured_image_token = getattr(
                    self.model.config,
                    "image_token_id",
                    getattr(self.model.config, "image_token_index", None),
                )
                if configured_image_token is None:
                    raise ValueError("Steering model config does not define an image token ID")
                image_token_id = int(configured_image_token)
                vision_mask = encoded.input_ids.eq(image_token_id)
                text_context_mask = context_token_mask(
                    encoded.input_ids,
                    self.processor.tokenizer,
                    prompts,
                    [item.context for item in inputs],
                )
                stack.enter_context(self._attention_steering.masks(
                    vision_mask,
                    text_context_mask,
                    prompt_lengths=encoded.attention_mask.sum(dim=1).long(),
                    sample_alphas=adaptive_alphas,
                ))
            distorted_images: list[Image.Image] = []
            logits_processor = None
            if self.method is MethodName.VCD:
                distorted_images = self._distort_images(images)
                distorted_kwargs = dict(processor_kwargs)
                distorted_kwargs["images"] = distorted_images
                distorted = self.processor(**distorted_kwargs).to(self.model.device)
                logits_processor = LogitsProcessorList([_VCDLogitsProcessor(
                    self.model, dict(distorted), float(self.method_config["alpha"]),
                    float(self.method_config["beta"]),
                )])
            input_tokens = int(encoded.attention_mask.sum().item())
            if cuda_devices:
                torch.cuda.synchronize()
            generation_started_at = perf_counter()
            streamer = _FirstTokenTimer(generation_started_at) if len(inputs) == 1 else None
            generation_kwargs: dict[str, Any] = {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "streamer": streamer,
            }
            if logits_processor is not None:
                generation_kwargs["logits_processor"] = logits_processor
            with ExitStack() as generation_stack:
                for image in distorted_images:
                    generation_stack.callback(image.close)
                if self._mps_layer is not None:
                    layer = _language_layers(self.model)[self._mps_layer]
                    generation_stack.enter_context(layer.register_forward_hook(self._mps_hook))
                with torch.inference_mode():
                    generated = self.model.generate(**encoded, **generation_kwargs)
            if cuda_devices:
                torch.cuda.synchronize()
            generation_latency = perf_counter() - generation_started_at
            trimmed = [output[len(source) :] for source, output in zip(encoded.input_ids, generated)]
            output_tokens = sum(int(output.numel()) for output in trimmed)
            texts = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            end_to_end_latency = perf_counter() - end_to_end_started_at
            ttft = streamer.ttft_seconds if streamer is not None else None
            decode_tokens = max(output_tokens - 1, 0) if ttft is not None else None
            decode_latency = max(generation_latency - ttft, 0.0) if ttft is not None else None
            sample_metrics = [
                {
                    "adaptive_similarity": float(similarity),
                    "adaptive_alpha": float(alpha),
                    "adaptive_threshold": self._adaptive_tau,
                    "adaptive_alpha_max": float(self.method_config["alpha_max"]),
                    "adaptive_lambda": float(self.method_config["lambda"]),
                }
                for similarity, alpha in zip(
                    adaptive_similarities.detach().cpu().tolist(),
                    adaptive_alphas.detach().cpu().tolist(),
                )
            ] if adaptive_similarities is not None and adaptive_alphas is not None else []
            return GenerationBatch(texts, {
                "batch_size": len(inputs),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "end_to_end_inference_latency_s": end_to_end_latency,
                "generation_latency_s": generation_latency,
                "ttft_s": ttft,
                "decode_latency_s": decode_latency,
                "decode_tokens": decode_tokens,
                "decode_latency_per_output_token_s": (
                    decode_latency / decode_tokens if decode_latency is not None and decode_tokens else None
                ),
                "requests_per_second": len(inputs) / end_to_end_latency,
                "token_throughput_tokens_per_second": (
                    output_tokens / generation_latency if generation_latency else None
                ),
                "peak_gpu_memory_bytes": (
                    max(torch.cuda.max_memory_allocated(index) for index in cuda_devices)
                    if cuda_devices else None
                ),
                "adaptive_probe_latency_s": adaptive_probe_latency,
            }, sample_metrics)
