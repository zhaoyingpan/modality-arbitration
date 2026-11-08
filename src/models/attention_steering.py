from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import torch
from torch.nn import functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


ATTENTION_STEERING_INTERFACE = "modality_arbitration_steering"


def _repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return hidden_states
    batch, key_value_heads, sequence_length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, repetitions, sequence_length, head_dim
    )
    return expanded.reshape(batch, key_value_heads * repetitions, sequence_length, head_dim)


def _normalized_additive_attention_mask(
    attention_mask: torch.Tensor | None,
    attention_logits: torch.Tensor,
    cache_position: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an additive causal/padding mask for the custom eager attention path.

    Transformers prepares the shared decoder mask for the model's original
    attention backend.  In particular, SDPA may return ``None`` and enforce
    causality internally, or return a boolean mask.  Neither representation can
    be consumed by this custom eager implementation without normalization.
    """
    batch, _, query_length, key_length = attention_logits.shape
    device = attention_logits.device

    query_positions: torch.Tensor | None = None
    if cache_position is not None:
        positions = cache_position.to(device=device)
        if positions.ndim == 1:
            positions = positions.view(1, -1).expand(batch, -1)
        elif positions.ndim == 2 and positions.shape[0] in {1, batch}:
            positions = positions.expand(batch, -1)
        else:
            raise ValueError("cache_position must have shape [query] or [batch, query]")
        if positions.shape[1] != query_length:
            raise ValueError("cache_position length does not match query length")
        query_positions = positions.long()
    elif query_length == key_length:
        # Equal lengths unambiguously identify ordinary prefill positions.
        query_positions = torch.arange(query_length, device=device).view(1, -1).expand(
            batch, -1
        )

    if attention_mask is not None:
        supplied = attention_mask.to(device=device)
        if supplied.ndim == 2:
            supplied = supplied[:, None, None, :key_length]
        elif supplied.ndim == 3:
            supplied = supplied[:, None, -query_length:, :key_length]
        elif supplied.ndim == 4:
            supplied = supplied[..., -query_length:, :key_length]
        else:
            raise ValueError(
                "Attention mask must have 2, 3, or 4 dimensions, "
                f"got shape {tuple(supplied.shape)}"
            )

        if supplied.shape[0] not in {1, batch}:
            raise ValueError("Attention mask batch size does not match attention logits")
        if supplied.shape[-1] != key_length:
            raise ValueError("Attention mask key length does not match attention logits")
        if supplied.shape[-2] not in {1, query_length}:
            raise ValueError("Attention mask query length does not match attention logits")

        if supplied.dtype == torch.bool or not supplied.dtype.is_floating_point:
            supplied_allowed = supplied.bool()
        elif attention_mask.ndim == 2:
            # A raw 2-D Transformers padding mask uses 1 for real tokens and 0
            # for padding, even when represented as floating point.
            supplied_allowed = supplied > 0
        else:
            # Materialized additive masks use zero (or another finite bias) for
            # allowed positions and -inf/finfo.min for blocked positions.
            supplied_allowed = torch.isfinite(supplied) & (
                supplied > torch.finfo(supplied.dtype).min / 2
            )
        if attention_mask.ndim == 2:
            if query_positions is None:
                raise ValueError("Cached decoding with a 2-D mask requires cache_position")
            key_positions = torch.arange(key_length, device=device)
            causal = key_positions.view(1, 1, 1, -1) <= query_positions[:, None, :, None]
            allowed = causal & supplied_allowed
        else:
            allowed = supplied_allowed
            if query_positions is None:
                # A prepared causal mask establishes each query's absolute key position.
                visible = allowed.any(dim=1)
                if not torch.all(visible.any(dim=-1)):
                    raise ValueError("Prepared attention mask has a query with no allowed keys")
                keys = torch.arange(key_length, device=device).view(1, 1, -1)
                query_positions = torch.where(visible, keys, -1).amax(dim=-1)
    else:
        if query_positions is None:
            raise ValueError(
                "Cached decoding requires cache_position when no prepared mask is available"
            )
        key_positions = torch.arange(key_length, device=device)
        allowed = key_positions.view(1, 1, 1, -1) <= query_positions[:, None, :, None]

    allowed = allowed.expand(batch, -1, query_length, key_length)
    additive = torch.zeros(
        allowed.shape, dtype=attention_logits.dtype, device=device
    )
    additive = additive.masked_fill(~allowed, torch.finfo(attention_logits.dtype).min)
    assert query_positions is not None
    return additive, allowed, query_positions


@dataclass
class AttentionSteeringController:
    model_family: str
    direction: str
    alpha: float
    suppress_opposite: bool
    strengths_by_layer: dict[int, dict[int, float]]
    artifact_path: Path
    top_k: int
    epsilon: float
    max_b: float
    apply_to: Literal["all", "generation"] = "all"
    vision_mask: torch.Tensor | None = field(default=None, init=False, repr=False)
    context_mask: torch.Tensor | None = field(default=None, init=False, repr=False)
    prompt_lengths: torch.Tensor | None = field(default=None, init=False, repr=False)
    sample_alphas: torch.Tensor | None = field(default=None, init=False, repr=False)
    active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.direction not in {"image", "text"}:
            raise ValueError("Steering direction must be either 'image' or 'text'")
        if self.top_k <= 0:
            raise ValueError("Steering top_k must be greater than zero")
        if self.apply_to not in {"all", "generation"}:
            raise ValueError("Steering apply_to must be 'all' or 'generation'")
        if math.isnan(self.alpha) or self.alpha < 0:
            raise ValueError("Steering alpha must be non-negative or positive infinity")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Steering epsilon must be finite and positive")

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        direction: str,
        top_k: int,
        alpha: float,
        epsilon: float,
        suppress_opposite: bool,
        apply_to: Literal["all", "generation"] = "all",
    ) -> "AttentionSteeringController":
        if direction not in {"image", "text"}:
            raise ValueError("Steering direction must be either 'image' or 'text'")
        if top_k <= 0:
            raise ValueError("Steering top_k must be greater than zero")
        if math.isnan(alpha) or alpha < 0:
            raise ValueError("Steering alpha must be non-negative or positive infinity")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Steering epsilon must be finite and positive")
        path = Path(artifact_path)
        with path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("method") != "steering":
            raise ValueError(f"Steering artifact has unexpected method: {artifact.get('method')!r}")
        ranking = artifact.get("ranking")
        if not isinstance(ranking, list) or not ranking:
            raise ValueError("Steering artifact must contain a non-empty ranking")
        if top_k > len(ranking):
            raise ValueError(
                f"Steering top_k={top_k} exceeds the artifact ranking size {len(ranking)}"
            )

        normalized: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        for item in ranking:
            if not isinstance(item, dict) or not {"layer", "head", "B"} <= item.keys():
                raise ValueError("Each steering ranking entry requires layer, head, and B")
            layer, head, b_value = int(item["layer"]), int(item["head"]), float(item["B"])
            if layer < 0 or head < 0 or not math.isfinite(b_value) or b_value < 0:
                raise ValueError("Steering layer/head and finite B values must be non-negative")
            if (layer, head) in seen:
                raise ValueError(f"Duplicate steering head ({layer}, {head})")
            seen.add((layer, head))
            normalized.append((layer, head, b_value))

        normalized.sort(key=lambda value: value[2], reverse=True)
        max_b = max(value[2] for value in normalized)
        denominator = max_b + epsilon
        strengths: defaultdict[int, dict[int, float]] = defaultdict(dict)
        for layer, head, b_value in normalized[:top_k]:
            strengths[layer][head] = b_value / denominator
        return cls(
            model_family=str(artifact.get("model_family", "")),
            direction=direction,
            alpha=alpha,
            suppress_opposite=suppress_opposite,
            strengths_by_layer=dict(strengths),
            artifact_path=path,
            top_k=top_k,
            epsilon=epsilon,
            max_b=max_b,
            apply_to=apply_to,
        )

    def set_token_masks(
        self,
        vision_mask: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        prompt_lengths: torch.Tensor | None = None,
        sample_alphas: torch.Tensor | None = None,
    ) -> None:
        if vision_mask.shape != context_mask.shape or vision_mask.ndim != 2:
            raise ValueError("Steering token masks must be same-shaped [batch, sequence] tensors")
        if self.direction == "image" and not torch.all(vision_mask.any(dim=1)):
            raise ValueError("steering_image could not identify visual tokens in every sample")
        if self.direction == "text" and not torch.all(context_mask.any(dim=1)):
            raise ValueError("steering_text could not identify context tokens in every sample")
        if self.suppress_opposite:
            opposite = context_mask if self.direction == "image" else vision_mask
            if not torch.all(opposite.any(dim=1)):
                raise ValueError("Opposite-modality suppression requires both token ranges")
        if self.vision_mask is not None or self.context_mask is not None:
            raise RuntimeError("Steering token masks are already active")
        if prompt_lengths is None:
            prompt_lengths = torch.full(
                (vision_mask.shape[0],), vision_mask.shape[1], dtype=torch.long
            )
        if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != vision_mask.shape[0]:
            raise ValueError("Steering prompt lengths must have shape [batch]")
        if torch.any(prompt_lengths <= 0) or torch.any(prompt_lengths > vision_mask.shape[1]):
            raise ValueError("Steering prompt lengths must fit within the modality masks")
        if sample_alphas is not None:
            if sample_alphas.ndim != 1 or sample_alphas.shape[0] != vision_mask.shape[0]:
                raise ValueError("Adaptive steering alphas must have shape [batch]")
            if not torch.isfinite(sample_alphas).all() or torch.any(sample_alphas < 0):
                raise ValueError("Adaptive steering alphas must be finite and non-negative")
        self.vision_mask = vision_mask.bool()
        self.context_mask = context_mask.bool()
        self.prompt_lengths = prompt_lengths.long()
        self.sample_alphas = sample_alphas.float() if sample_alphas is not None else None
        self.active = True

    @contextmanager
    def masks(
        self,
        vision_mask: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        prompt_lengths: torch.Tensor | None = None,
        sample_alphas: torch.Tensor | None = None,
    ) -> Iterator["AttentionSteeringController"]:
        self.set_token_masks(
            vision_mask,
            context_mask,
            prompt_lengths=prompt_lengths,
            sample_alphas=sample_alphas,
        )
        try:
            yield self
        finally:
            self.clear_token_masks()

    def clear_token_masks(self) -> None:
        self.vision_mask = None
        self.context_mask = None
        self.prompt_lengths = None
        self.sample_alphas = None
        self.active = False

    def modify_logits(
        self,
        layer: int,
        attention_logits: torch.Tensor,
        *,
        allowed: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        # Unimodal adaptive probes use the same eager path with steering inactive.
        if not self.active:
            return attention_logits
        layer_strengths = self.strengths_by_layer.get(layer)
        if not layer_strengths:
            return attention_logits
        if self.vision_mask is None or self.context_mask is None:
            raise RuntimeError("Steering token masks were not configured before attention")
        batch, _, _, key_length = attention_logits.shape
        if self.vision_mask.shape[0] != batch:
            raise RuntimeError("Steering mask batch size does not match attention batch size")

        positive = self.vision_mask if self.direction == "image" else self.context_mask
        negative = self.context_mask if self.direction == "image" else self.vision_mask
        mask_length = self.vision_mask.shape[1]
        query_length = attention_logits.shape[2]
        if key_length < mask_length:
            raise RuntimeError(
                f"Attention key length {key_length} is shorter than modality-mask length "
                f"{mask_length}"
            )
        if query_length == key_length and mask_length != key_length:
            raise RuntimeError(
                f"Prefill key length {key_length} does not equal modality-mask length "
                f"{mask_length}"
            )
        if key_length > mask_length:
            expected = torch.arange(
                key_length - query_length, key_length, device=query_positions.device
            ).view(1, -1)
            if query_positions.shape != (batch, query_length) or not torch.all(
                query_positions == expected
            ):
                raise RuntimeError(
                    "Modality masks may only be extended for appended decode tokens"
                )

        def align(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
            result = torch.zeros(
                (batch, key_length), dtype=dtype, device=attention_logits.device
            )
            result[:, :mask.shape[1]] = mask.to(
                device=attention_logits.device, dtype=dtype
            )
            return result

        heads = sorted(
            head for head, strength in layer_strengths.items()
            if self.alpha != math.inf or strength > 0
        )
        if not heads:
            return attention_logits
        head_indices = torch.tensor(heads, device=attention_logits.device)
        strengths = torch.tensor(
            [layer_strengths[head] for head in heads],
            device=attention_logits.device,
            dtype=attention_logits.dtype,
        ).view(1, -1, 1, 1)
        if self.apply_to == "generation":
            if self.prompt_lengths is None:
                raise RuntimeError("Generation-only steering requires prompt lengths")
            scope = query_positions >= self.prompt_lengths.to(query_positions.device).view(-1, 1)
        else:
            scope = torch.ones(
                (batch, query_length), dtype=torch.bool, device=attention_logits.device
            )

        if self.alpha == math.inf:
            positive_keys = align(positive, dtype=torch.bool).view(batch, 1, 1, key_length)
            positive_allowed = positive_keys & allowed
            has_positive = positive_allowed.any(dim=-1, keepdim=True)
            keep = positive_allowed | ~has_positive
            if self.suppress_opposite:
                negative_keys = align(negative, dtype=torch.bool).view(
                    batch, 1, 1, key_length
                )
                # If no target token is accessible, retain neutral keys while
                # applying the alpha->infinity suppression limit to opposite keys.
                keep_without_positive = ~negative_keys | ~allowed
                keep = torch.where(has_positive, positive_allowed, keep_without_positive)
            changed = attention_logits.clone()
            selected = changed[:, head_indices, :, :]
            hard_steered = selected.masked_fill(
                ~keep.expand(batch, len(heads), query_length, key_length),
                torch.finfo(attention_logits.dtype).min,
            )
            selected = torch.where(scope[:, None, :, None], hard_steered, selected)
            changed[:, head_indices, :, :] = selected
            return changed

        alpha = (
            self.sample_alphas.to(attention_logits.device).view(batch, 1, 1, 1)
            if self.sample_alphas is not None
            else self.alpha
        )
        modifier = (
            alpha
            * strengths.float()
            * align(positive, dtype=torch.float32).view(batch, 1, 1, key_length)
        )
        if self.suppress_opposite:
            modifier = modifier - (
                alpha
                * strengths.float()
                * align(negative, dtype=torch.float32).view(batch, 1, 1, key_length)
            )
        modifier = modifier * scope[:, None, :, None]
        changed = attention_logits.clone()
        # Avoid fp16/bf16 overflow before the final hard mask is applied.
        limit = torch.finfo(attention_logits.dtype).max
        modifier = torch.nan_to_num(modifier, nan=0.0, posinf=limit, neginf=-limit)
        steered = changed[:, head_indices, :, :].float() + modifier
        changed[:, head_indices, :, :] = steered.clamp(-limit, limit).to(
            attention_logits.dtype
        )
        return changed


def steering_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attention_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    additive_mask, allowed, query_positions = _normalized_additive_attention_mask(
        attention_mask, attention_logits, kwargs.get("cache_position")
    )
    controller: AttentionSteeringController = module._modality_steering_controller
    # Steering precedes masking so a hard-blocked key can never be revived.
    attention_logits = controller.modify_logits(
        module.layer_idx,
        attention_logits,
        allowed=allowed,
        query_positions=query_positions,
    )
    attention_logits = attention_logits + additive_mask
    attention_logits = attention_logits.masked_fill(
        ~allowed, torch.finfo(attention_logits.dtype).min
    )
    attention_weights = F.softmax(attention_logits, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_weights = F.dropout(attention_weights, p=dropout, training=module.training)
    output = torch.matmul(attention_weights, value_states).transpose(1, 2).contiguous()
    return output, attention_weights


def install_attention_steering(
    model: Any, controller: AttentionSteeringController
) -> Callable[[], None]:
    model_type = getattr(model.config, "model_type", None)
    expected_families = {
        "qwen3_vl": "qwen3-vl-8b",
        "qwen2_5_vl": "qwen2.5-vl-7b",
        "llava_onevision": "llava-ov-7b",
        "internvl": "internvl3-8b",
    }
    if model_type not in expected_families:
        raise ValueError(f"Resolution-head attention steering does not support {model_type!r}")
    expected_family = expected_families[model_type]
    if controller.model_family != expected_family:
        raise ValueError(
            f"Steering artifact is for {controller.model_family!r}, expected {expected_family!r}"
        )
    layer_paths = (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
    )
    layers = None
    for path in layer_paths:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        layers = value
        break
    if layers is None:
        raise ValueError(f"Could not locate decoder layers for {expected_family}")
    num_heads = int(model.config.text_config.num_attention_heads)
    targets: list[Any] = []
    for layer_index, head_strengths in controller.strengths_by_layer.items():
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"Steering layer {layer_index} is outside [0, {len(layers)})")
        if any(not 0 <= head < num_heads for head in head_strengths):
            raise ValueError(f"Steering artifact contains an invalid head for layer {layer_index}")
        attention = layers[layer_index].self_attn
        if hasattr(attention, "_modality_steering_controller"):
            raise RuntimeError(f"Attention layer {layer_index} is already patched for steering")
        targets.append(attention)

    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_STEERING_INTERFACE, steering_attention_forward)
    originals = [(attention, attention.config) for attention in targets]
    hook_handles = []

    def supply_cache_position(module: Any, args: Any, kwargs: dict[str, Any]):
        if kwargs.get("cache_position") is not None:
            return args, kwargs
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError("Steering attention requires hidden_states to locate queries")
        query_length = hidden_states.shape[1]
        cache = kwargs.get("past_key_values")
        past_length = 0 if cache is None else int(cache.get_seq_length(module.layer_idx))
        # Read the cache before the attention module appends the current queries.
        kwargs["cache_position"] = torch.arange(
            past_length,
            past_length + query_length,
            device=hidden_states.device,
        )
        return args, kwargs

    for attention in targets:
        attention.config = copy(attention.config)
        attention.config._attn_implementation = ATTENTION_STEERING_INTERFACE
        attention._modality_steering_controller = controller
        hook_handles.append(
            attention.register_forward_pre_hook(supply_cache_position, with_kwargs=True)
        )

    def uninstall() -> None:
        for handle in hook_handles:
            handle.remove()
        for attention, original_config in originals:
            attention.config = original_config
            if hasattr(attention, "_modality_steering_controller"):
                delattr(attention, "_modality_steering_controller")

    return uninstall


def context_token_mask(
    input_ids: torch.Tensor,
    tokenizer: Any,
    prompts: list[str],
    contexts: list[str | None],
) -> torch.Tensor:
    if input_ids.shape[0] != len(contexts) or len(prompts) != len(contexts):
        raise ValueError("Prompt/context count does not match encoded batch size")
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for batch_index, (prompt, context) in enumerate(zip(prompts, contexts)):
        if context is None:
            continue
        segment = f"Text context: {context}"
        segment_occurrences = prompt.count(segment)
        if segment_occurrences == 0:
            raise ValueError(f"Rendered prompt {batch_index} does not contain its text context")
        if segment_occurrences != 1:
            raise ValueError(f"Rendered prompt {batch_index} contains ambiguous text contexts")
        segment_start = prompt.find(segment)
        context_start = segment_start + len("Text context: ")
        context_end = segment_start + len(segment)
        tokenized = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True
        )
        prompt_ids = list(tokenized["input_ids"])
        offsets = list(tokenized["offset_mapping"])
        anchor_tokens = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > segment_start and start < context_end
        ]
        local_context_tokens = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > context_start and start < context_end
        ]
        if not anchor_tokens or not local_context_tokens:
            raise ValueError(f"Text context for batch item {batch_index} produced no tokens")
        anchor_start = anchor_tokens[0]
        anchor_end = anchor_tokens[-1] + 1
        anchor_ids = prompt_ids[anchor_start:anchor_end]
        row = input_ids[batch_index].tolist()
        matches = [
            start
            for start in range(len(row) - len(anchor_ids) + 1)
            if row[start : start + len(anchor_ids)] == anchor_ids
        ]
        if not matches:
            raise ValueError(
                f"Could not locate the anchored text-context segment for batch item {batch_index}"
            )
        if len(matches) != 1:
            raise ValueError(f"Ambiguous text-context segment for batch item {batch_index}")
        context_token_start = local_context_tokens[0] - anchor_start
        context_token_end = local_context_tokens[-1] + 1 - anchor_start
        match_start = matches[0]
        mask[
            batch_index,
            match_start + context_token_start : match_start + context_token_end,
        ] = True
    return mask
