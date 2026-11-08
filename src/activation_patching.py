from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from conditions import ConditionName
from data import MC2PairedExample
from metrics import contains_answer, normalize_answer


ANSWER_SCORE = "sum_teacher_forced_log_prob"
PATCH_POSITION = "last_prompt_token"
PATCH_SITE = "self_attn.o_proj input (concatenated per-head attention outputs)"
PATCHING_INSTRUCTION = "Answer the question directly with a short answer."


def activation_prompt(condition: ConditionName, example: MC2PairedExample) -> str:
    parts: list[str] = []
    if condition in (ConditionName.CONTEXT_ONLY, ConditionName.IMAGE_CONTEXT):
        parts.append(f"Text context: {example.context}")
    parts.extend((example.question, PATCHING_INSTRUCTION))
    return "\n".join(parts)


def modality_preference(correct_vision: bool, correct_text: bool) -> str:
    if correct_vision and correct_text:
        return "both"
    if correct_vision:
        return "vision"
    if correct_text:
        return "text"
    return "neither"


@dataclass(frozen=True)
class ScreeningResult:
    image_only_output: str
    text_only_output: str
    image_only_correct_vision: bool
    text_only_correct_text: bool

    @property
    def retained(self) -> bool:
        return self.image_only_correct_vision and self.text_only_correct_text


@dataclass(frozen=True)
class PreparedPrompt:
    encoded: dict[str, Any]
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None
    language_kwargs: dict[str, Any]
    prompt_token_count: int


@dataclass(frozen=True)
class CandidateInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None
    language_kwargs: dict[str, Any]
    answer_token_ids: list[int]
    prompt_token_count: int


class _LanguageInputsCaptured(RuntimeError):
    """Internal control flow used to stop before an expensive decoder pass."""


def _matrix_to_lists(matrix: torch.Tensor) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.detach().float().cpu().tolist()]


def _screening_record(example: MC2PairedExample, result: ScreeningResult) -> dict[str, Any]:
    return {
        "record_schema_version": "1.0",
        "dataset": "mc2",
        "dataset_index": example.index,
        "sample_id": f"mc2:{example.index}",
        "image_id": example.image_id,
        "question": example.question,
        "answer_vision": example.answer_vision,
        "answer_text": example.answer_text,
        "instruction": PATCHING_INSTRUCTION,
        "image_only_output": result.image_only_output,
        "text_only_output": result.text_only_output,
        "image_only_correct_vision": result.image_only_correct_vision,
        "text_only_correct_text": result.text_only_correct_text,
        "retained": result.retained,
    }


def load_jsonl_by_index(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["dataset_index"])
            if index in rows:
                raise ValueError(f"Duplicate dataset_index={index} in {path}:{line_number}")
            rows[index] = row
    return rows


def _prediction_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    context = row.get("text_context", row.get("context"))
    if context is None:
        raise ValueError("A screening prediction has no text context")
    return (
        str(row["image_id"]).zfill(12),
        str(row["question"]),
        str(context),
        str(row["answer_vision"]),
        str(row["answer_text"]),
    )


def _example_key(example: MC2PairedExample) -> tuple[str, str, str, str, str]:
    return (
        example.image_id,
        example.question,
        example.context,
        example.answer_vision,
        example.answer_text,
    )


def _load_prediction_index(path: Path, expected_condition: str) -> dict[tuple[str, ...], dict[str, Any]]:
    rows: dict[tuple[str, ...], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("condition")) != expected_condition:
                raise ValueError(
                    f"Expected condition={expected_condition!r} in {path}:{line_number}"
                )
            if bool(row.get("parser_failed")):
                raise ValueError(f"Screening parser failed in {path}:{line_number}")
            key = _prediction_key(row)
            if key in rows:
                raise ValueError(f"Duplicate sample identity in {path}:{line_number}")
            rows[key] = row
    return rows


def import_screening_predictions(
    examples: list[MC2PairedExample],
    image_only_predictions: Path,
    text_only_predictions: Path,
    output_file: Path,
    *,
    conflict_predictions: Path | None = None,
    resume: bool = False,
) -> dict[str, int]:
    """Join full-MC2 prediction runs to the one-token variant by sample identity."""
    existed_before = output_file.exists()
    image_rows = _load_prediction_index(image_only_predictions, "image_only")
    text_rows = _load_prediction_index(text_only_predictions, "context_only")
    conflict_rows = (
        None
        if conflict_predictions is None
        else _load_prediction_index(conflict_predictions, "image_context")
    )
    imported: list[dict[str, Any]] = []
    for example in examples:
        key = _example_key(example)
        if key not in image_rows or key not in text_rows:
            raise ValueError(
                "Could not match one-token MC2 sample to both screening runs: "
                f"dataset_index={example.index}"
            )
        image_row = image_rows[key]
        text_row = text_rows[key]
        conflict_row = None if conflict_rows is None else conflict_rows.get(key)
        if conflict_rows is not None and conflict_row is None:
            raise ValueError(
                "Could not match one-token MC2 sample to the conflict prediction run: "
                f"dataset_index={example.index}"
            )
        image_correct = bool(image_row["parser_correct_vision"])
        text_correct = bool(text_row["parser_correct_text"])
        expected_prompts = {
            "image_only": activation_prompt(ConditionName.IMAGE_ONLY, example),
            "context_only": activation_prompt(ConditionName.CONTEXT_ONLY, example),
            "image_context": activation_prompt(ConditionName.IMAGE_CONTEXT, example),
        }
        for condition, source_row in (
            ("image_only", image_row),
            ("context_only", text_row),
            ("image_context", conflict_row),
        ):
            if source_row is not None and str(source_row.get("prompt")) != expected_prompts[condition]:
                raise ValueError(
                    f"Imported {condition} prompt does not match diagnosis prompt for "
                    f"dataset_index={example.index}"
                )
        imported.append({
            "record_schema_version": "1.0",
            "dataset": "mc2",
            "dataset_index": example.index,
            "sample_id": f"mc2:{example.index}",
            "image_id": example.image_id,
            "question": example.question,
            "answer_vision": example.answer_vision,
            "answer_text": example.answer_text,
            "diagnosis_instruction": PATCHING_INSTRUCTION,
            "screening_source": "imported_predictions",
            "image_only_predictions_file": str(image_only_predictions),
            "text_only_predictions_file": str(text_only_predictions),
            "conflict_predictions_file": (
                None if conflict_predictions is None else str(conflict_predictions)
            ),
            "image_only_prompt": image_row.get("prompt"),
            "text_only_prompt": text_row.get("prompt"),
            "image_only_output": str(image_row["inference_result"]),
            "text_only_output": str(text_row["inference_result"]),
            "image_only_correct_vision": image_correct,
            "text_only_correct_text": text_correct,
            "image_only_parser": image_row.get("parser_type"),
            "text_only_parser": text_row.get("parser_type"),
            "retained": image_correct and text_correct,
            **(
                {}
                if conflict_row is None
                else {
                    "conflict_prompt": conflict_row.get("prompt"),
                    "conflict_output": str(conflict_row["inference_result"]),
                    "conflict_correct_vision": bool(conflict_row["parser_correct_vision"]),
                    "conflict_correct_text": bool(conflict_row["parser_correct_text"]),
                    "conflict_parser": conflict_row.get("parser_type"),
                }
            ),
        })
    if existed_before:
        if not resume:
            raise FileExistsError(output_file)
        existing = load_jsonl_by_index(output_file)
        if list(existing.values()) != imported:
            raise ValueError(f"Existing screening file does not match imported runs: {output_file}")
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("x", encoding="utf-8") as handle:
            for row in imported:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return {
        "total": len(imported),
        "written": 0 if existed_before else len(imported),
        "resumed": len(imported) if existed_before else 0,
        "retained": sum(bool(row["retained"]) for row in imported),
    }


def filtered_dataset_rows(
    examples: Iterable[MC2PairedExample],
    screening_rows: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        screen = screening_rows.get(example.index)
        if screen is None or not bool(screen["retained"]):
            continue
        rows.append({
            "dataset_index": example.index,
            "image_id": example.image_id,
            "task": example.task,
            "question": example.question,
            "context": example.context,
            "aligned_context": example.aligned_context,
            "answer_vision": example.answer_vision,
            "answer_text": example.answer_text,
        })
    return rows


def write_filtered_dataset(
    path: Path,
    examples: Iterable[MC2PairedExample],
    screening_rows: dict[int, dict[str, Any]],
    *,
    resume: bool = False,
) -> int:
    rows = filtered_dataset_rows(examples, screening_rows)
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != rows:
            raise ValueError(f"Existing filtered dataset does not match current screening: {path}")
        return len(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return len(rows)


class ActivationPatchingDiagnoser:
    """Head-output patching for Hugging Face VLMs with a decoder language model."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_new_tokens: int = 8,
        head_batch_size: int = 28,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if head_batch_size < 1:
            raise ValueError("head_batch_size must be at least 1")
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.head_batch_size = head_batch_size
        self._backbone, self._language_model, self._lm_head = self._model_components()
        self._attention_modules = self._language_attention_modules()
        self.num_layers = len(self._attention_modules)
        self.num_heads = int(self._attention_modules[0].config.num_attention_heads)
        self.head_dim = int(self._attention_modules[0].head_dim)
        for module in self._attention_modules:
            if int(module.config.num_attention_heads) != self.num_heads:
                raise RuntimeError("Decoder layers do not share one attention-head count")
            if int(module.head_dim) != self.head_dim:
                raise RuntimeError("Decoder layers do not share one attention-head dimension")
            if int(module.o_proj.in_features) != self.num_heads * self.head_dim:
                raise RuntimeError("o_proj input cannot be partitioned into attention heads")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        model_class: type,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 8,
        head_batch_size: int = 28,
        trust_remote_code: bool = False,
    ) -> "ActivationPatchingDiagnoser":
        if dtype == "auto":
            torch_dtype: Any = "auto"
        else:
            try:
                torch_dtype = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"Unknown torch dtype: {dtype}") from error
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
            trust_remote_code=trust_remote_code,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        return cls(
            model,
            processor,
            max_new_tokens=max_new_tokens,
            head_batch_size=head_batch_size,
        )

    def _model_components(self) -> tuple[Any, torch.nn.Module, torch.nn.Module]:
        backbone = getattr(self.model, "model", None)
        language_model = getattr(backbone, "language_model", None)
        lm_head = getattr(self.model, "lm_head", None)
        if backbone is None or language_model is None or lm_head is None:
            raise RuntimeError("Could not locate the VLM backbone, language model, and lm_head")
        if not hasattr(backbone, "get_input_embeddings"):
            raise RuntimeError("The VLM backbone does not expose input embeddings")
        return backbone, language_model, lm_head

    def _language_attention_modules(self) -> list[Any]:
        layers = getattr(self._language_model, "layers", None)
        if layers is None:
            raise RuntimeError("Could not locate language-model decoder layers")
        modules = [getattr(layer, "self_attn", None) for layer in layers]
        if not modules or any(module is None or not hasattr(module, "o_proj") for module in modules):
            raise RuntimeError("Could not locate every language self-attention o_proj")
        return modules

    def _prompt(self, condition: ConditionName, example: MC2PairedExample) -> str:
        return activation_prompt(condition, example)

    def _encode(
        self,
        image: Image.Image | None,
        prompt: str,
    ) -> PreparedPrompt:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [rendered],
            "return_tensors": "pt",
        }
        if image is not None:
            processor_kwargs["images"] = [image]
        batch = self.processor(**processor_kwargs).to(self.model.device)
        encoded = {key: value for key, value in batch.items()}
        input_ids = encoded["input_ids"]
        captured: dict[str, Any] = {}

        def capture_language_inputs(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            if args:
                raise RuntimeError("Expected the VLM backbone to call its language model by keyword")
            captured.update(kwargs)
            raise _LanguageInputsCaptured

        handle = self._language_model.register_forward_pre_hook(
            capture_language_inputs,
            with_kwargs=True,
        )
        try:
            with torch.inference_mode():
                self._backbone(**encoded, use_cache=False, return_dict=True)
        except _LanguageInputsCaptured:
            pass
        finally:
            handle.remove()
        inputs_embeds = captured.pop("inputs_embeds", None)
        attention_mask = captured.pop("attention_mask", None)
        position_ids = captured.pop("position_ids", None)
        captured.pop("input_ids", None)
        captured.pop("past_key_values", None)
        captured.pop("use_cache", None)
        captured.pop("return_dict", None)
        if inputs_embeds is None or attention_mask is None:
            raise RuntimeError("Failed to capture prepared language-model inputs from the VLM")
        return PreparedPrompt(
            encoded=encoded,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            language_kwargs=captured,
            prompt_token_count=int(input_ids.shape[1]),
        )

    def _generate(self, prepared: PreparedPrompt) -> str:
        with torch.inference_mode():
            generated = self.model.generate(
                **prepared.encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        generated_ids = generated[0, prepared.prompt_token_count :].tolist()
        return self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def screen(self, image: Image.Image, example: MC2PairedExample) -> ScreeningResult:
        image_only = self._encode(
            image,
            self._prompt(ConditionName.IMAGE_ONLY, example),
        )
        text_only = self._encode(
            None,
            self._prompt(ConditionName.CONTEXT_ONLY, example),
        )
        image_output = self._generate(image_only)
        text_output = self._generate(text_only)
        return ScreeningResult(
            image_only_output=image_output,
            text_only_output=text_output,
            image_only_correct_vision=contains_answer(image_output, example.answer_vision),
            text_only_correct_text=contains_answer(text_output, example.answer_text),
        )

    def _answer_token_ids(self, answer: str) -> list[int]:
        token_ids = self.processor.tokenizer.encode(answer, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Answer {answer!r} produced no tokenizer IDs")
        return [int(token_id) for token_id in token_ids]

    def _candidate_inputs(
        self,
        prepared: PreparedPrompt,
        answer_token_ids: list[int],
    ) -> CandidateInputs:
        token_ids = torch.tensor(
            [answer_token_ids],
            dtype=prepared.input_ids.dtype,
            device=prepared.input_ids.device,
        )
        answer_embeds = self._backbone.get_input_embeddings()(token_ids)
        inputs_embeds = torch.cat([prepared.inputs_embeds, answer_embeds], dim=1)
        answer_mask = torch.ones_like(token_ids, dtype=prepared.attention_mask.dtype)
        attention_mask = torch.cat([prepared.attention_mask, answer_mask], dim=1)
        position_ids = self._extend_position_ids(prepared.position_ids, len(answer_token_ids))
        language_kwargs = dict(prepared.language_kwargs)
        visual_pos_masks = language_kwargs.get("visual_pos_masks")
        if visual_pos_masks is not None:
            language_kwargs["visual_pos_masks"] = torch.cat(
                [
                    visual_pos_masks,
                    torch.zeros_like(token_ids, dtype=visual_pos_masks.dtype),
                ],
                dim=1,
            )
        return CandidateInputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            language_kwargs=language_kwargs,
            answer_token_ids=answer_token_ids,
            prompt_token_count=prepared.prompt_token_count,
        )

    @staticmethod
    def _extend_position_ids(
        position_ids: torch.Tensor | None,
        extra_tokens: int,
    ) -> torch.Tensor | None:
        if position_ids is None:
            return None
        offsets = torch.arange(
            1,
            extra_tokens + 1,
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        view_shape = [1] * position_ids.ndim
        view_shape[-1] = extra_tokens
        suffix = position_ids[..., -1:] + offsets.view(view_shape)
        return torch.cat([position_ids, suffix], dim=-1)

    @staticmethod
    def _repeat_candidate(candidate: CandidateInputs, batch_size: int) -> CandidateInputs:
        position_ids = candidate.position_ids
        if position_ids is not None:
            if position_ids.ndim == 3:
                position_ids = position_ids.repeat(1, batch_size, 1)
            elif position_ids.ndim == 2:
                position_ids = position_ids.repeat(batch_size, 1)
            else:
                raise RuntimeError(f"Unexpected position_ids rank: {position_ids.ndim}")
        language_kwargs = dict(candidate.language_kwargs)
        visual_pos_masks = language_kwargs.get("visual_pos_masks")
        if visual_pos_masks is not None:
            language_kwargs["visual_pos_masks"] = visual_pos_masks.repeat(batch_size, 1)
        deepstack_visual_embeds = language_kwargs.get("deepstack_visual_embeds")
        if deepstack_visual_embeds is not None:
            language_kwargs["deepstack_visual_embeds"] = [
                value.repeat(batch_size, 1) for value in deepstack_visual_embeds
            ]
        return CandidateInputs(
            inputs_embeds=candidate.inputs_embeds.repeat(batch_size, 1, 1),
            attention_mask=candidate.attention_mask.repeat(batch_size, 1),
            position_ids=position_ids,
            language_kwargs=language_kwargs,
            answer_token_ids=candidate.answer_token_ids,
            prompt_token_count=candidate.prompt_token_count,
        )

    def _language_forward(
        self,
        inputs: PreparedPrompt | CandidateInputs,
    ) -> Any:
        return self._language_model(
            input_ids=None,
            inputs_embeds=inputs.inputs_embeds,
            attention_mask=inputs.attention_mask,
            position_ids=inputs.position_ids,
            use_cache=False,
            return_dict=True,
            **inputs.language_kwargs,
        )

    def _candidate_log_probs(self, candidate: CandidateInputs) -> torch.Tensor:
        outputs = self._language_forward(candidate)
        start = candidate.prompt_token_count - 1
        stop = start + len(candidate.answer_token_ids)
        answer_hidden = outputs.last_hidden_state[:, start:stop, :]
        logits = self._lm_head(answer_hidden).float()
        targets = torch.tensor(
            candidate.answer_token_ids,
            dtype=torch.long,
            device=logits.device,
        ).view(1, -1, 1).expand(logits.shape[0], -1, -1)
        token_log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)
        return token_log_probs.sum(dim=-1)

    def _margin(
        self,
        prepared: PreparedPrompt,
        answer_vision_token_ids: list[int],
        answer_text_token_ids: list[int],
    ) -> float:
        vision = self._candidate_inputs(prepared, answer_vision_token_ids)
        text = self._candidate_inputs(prepared, answer_text_token_ids)
        with torch.inference_mode():
            score_vision = self._candidate_log_probs(vision)[0]
            score_text = self._candidate_log_probs(text)[0]
        return float((score_vision - score_text).detach().cpu())

    def _capture_head_outputs(self, prepared: PreparedPrompt) -> torch.Tensor:
        captured: list[torch.Tensor | None] = [None] * self.num_layers
        handles = []
        for layer_index, attention in enumerate(self._attention_modules):
            def capture(
                module: torch.nn.Module,
                args: tuple[torch.Tensor, ...],
                index: int = layer_index,
            ) -> None:
                value = args[0]
                captured[index] = value[0, prepared.prompt_token_count - 1].reshape(
                    self.num_heads,
                    self.head_dim,
                ).detach().float().cpu()

            handles.append(attention.o_proj.register_forward_pre_hook(capture))
        try:
            with torch.inference_mode():
                self._language_forward(prepared)
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in captured):
            raise RuntimeError("Failed to capture a head output at every decoder layer")
        return torch.stack([value for value in captured if value is not None], dim=0)

    def _patched_margins(
        self,
        prepared: PreparedPrompt,
        answer_vision_token_ids: list[int],
        answer_text_token_ids: list[int],
        source_head_outputs: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(source_head_outputs.shape) != (self.num_layers, self.num_heads, self.head_dim):
            raise ValueError("Source head-output tensor has an incompatible shape")
        vision_candidate = self._candidate_inputs(prepared, answer_vision_token_ids)
        text_candidate = self._candidate_inputs(prepared, answer_text_token_ids)
        margins = torch.empty((self.num_layers, self.num_heads), dtype=torch.float32)
        for layer_index, attention in enumerate(self._attention_modules):
            for head_start in range(0, self.num_heads, self.head_batch_size):
                head_indices = list(range(
                    head_start,
                    min(head_start + self.head_batch_size, self.num_heads),
                ))
                batch_size = len(head_indices)

                def patch_heads(
                    module: torch.nn.Module,
                    args: tuple[torch.Tensor, ...],
                ) -> tuple[torch.Tensor, ...]:
                    value = args[0].clone()
                    source = source_head_outputs[layer_index].to(
                        device=value.device,
                        dtype=value.dtype,
                    )
                    position = prepared.prompt_token_count - 1
                    for batch_index, head_index in enumerate(head_indices):
                        begin = head_index * self.head_dim
                        end = begin + self.head_dim
                        value[batch_index, position, begin:end] = source[head_index]
                    return (value, *args[1:])

                handle = attention.o_proj.register_forward_pre_hook(patch_heads)
                try:
                    with torch.inference_mode():
                        score_vision = self._candidate_log_probs(
                            self._repeat_candidate(vision_candidate, batch_size)
                        )
                        score_text = self._candidate_log_probs(
                            self._repeat_candidate(text_candidate, batch_size)
                        )
                finally:
                    handle.remove()
                margins[layer_index, head_indices] = (
                    score_vision - score_text
                ).detach().float().cpu()
        return margins

    def diagnose(
        self,
        image: Image.Image,
        example: MC2PairedExample,
        screening: dict[str, Any],
        conflict_margin_row: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_only = self._encode(
            image,
            self._prompt(ConditionName.IMAGE_ONLY, example),
        )
        text_only = self._encode(
            None,
            self._prompt(ConditionName.CONTEXT_ONLY, example),
        )
        conflict = self._encode(
            image,
            self._prompt(ConditionName.IMAGE_CONTEXT, example),
        )
        imported_conflict = "conflict_output" in screening
        conflict_output = str(screening.get("conflict_output", ""))
        if conflict_margin_row is not None:
            conflict_output = str(conflict_margin_row["conflict_output"])
        elif not imported_conflict:
            conflict_output = self._generate(conflict)
        answer_vision_token_ids = self._answer_token_ids(example.answer_vision)
        answer_text_token_ids = self._answer_token_ids(example.answer_text)

        image_head_outputs = self._capture_head_outputs(image_only)
        text_head_outputs = self._capture_head_outputs(text_only)
        margin_conflict = (
            float(conflict_margin_row["margin_conflict"])
            if conflict_margin_row is not None
            else self._margin(
                conflict,
                answer_vision_token_ids,
                answer_text_token_ids,
            )
        )
        margin_image_only = self._margin(
            image_only,
            answer_vision_token_ids,
            answer_text_token_ids,
        )
        margin_text_only = self._margin(
            text_only,
            answer_vision_token_ids,
            answer_text_token_ids,
        )
        patched_vision = self._patched_margins(
            conflict,
            answer_vision_token_ids,
            answer_text_token_ids,
            image_head_outputs,
        )
        patched_text = self._patched_margins(
            conflict,
            answer_vision_token_ids,
            answer_text_token_ids,
            text_head_outputs,
        )
        causal_vision = patched_vision - margin_conflict
        causal_text = patched_text - margin_conflict
        if conflict_margin_row is not None:
            conflict_correct_vision = bool(conflict_margin_row["conflict_correct_vision"])
            conflict_correct_text = bool(conflict_margin_row["conflict_correct_text"])
            preference = str(conflict_margin_row["conflict_preference"])
        else:
            conflict_correct_vision = (
                bool(screening["conflict_correct_vision"])
                if imported_conflict
                else contains_answer(conflict_output, example.answer_vision)
            )
            conflict_correct_text = (
                bool(screening["conflict_correct_text"])
                if imported_conflict
                else contains_answer(conflict_output, example.answer_text)
            )
            preference = modality_preference(conflict_correct_vision, conflict_correct_text)

        return {
            "record_schema_version": "1.0",
            "dataset": "mc2",
            "dataset_index": example.index,
            "sample_id": f"mc2:{example.index}",
            "image_id": example.image_id,
            "image_path": str(example.image_path),
            "task": example.task,
            "question": example.question,
            "context": example.context,
            "answer_vision": example.answer_vision,
            "answer_text": example.answer_text,
            "answer_vision_token_ids": answer_vision_token_ids,
            "answer_text_token_ids": answer_text_token_ids,
            "instruction": PATCHING_INSTRUCTION,
            "filter": {
                "image_only_output": screening["image_only_output"],
                "text_only_output": screening["text_only_output"],
                "image_only_correct_vision": bool(screening["image_only_correct_vision"]),
                "text_only_correct_text": bool(screening["text_only_correct_text"]),
                "retained": bool(screening["retained"]),
            },
            "conflict_output": conflict_output,
            "conflict_output_source": (
                "margin_phase"
                if conflict_margin_row is not None
                else ("imported_predictions" if imported_conflict else "native_generation")
            ),
            "conflict_preference": preference,
            "conflict_preference_matcher": (
                str(screening.get("conflict_parser", "hybrid"))
                if imported_conflict
                else "normalized_reference_contiguous_token_subset"
            ),
            "conflict_correct_vision": conflict_correct_vision,
            "conflict_correct_text": conflict_correct_text,
            "answer_score": ANSWER_SCORE,
            "margin_definition": "log_p(answer_vision) - log_p(answer_text)",
            "margin_conflict": margin_conflict,
            "margin_image_only": margin_image_only,
            "margin_text_only": margin_text_only,
            "patch_site": PATCH_SITE,
            "patch_position": PATCH_POSITION,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "c_vision_layer_head": _matrix_to_lists(causal_vision),
            "c_text_layer_head": _matrix_to_lists(causal_text),
        }


def run_screening(
    diagnoser: ActivationPatchingDiagnoser,
    examples: list[MC2PairedExample],
    output_file: Path,
    *,
    resume: bool = False,
) -> dict[str, int]:
    existing = load_jsonl_by_index(output_file) if resume else {}
    expected_indices = {example.index for example in examples}
    existing = {index: row for index, row in existing.items() if index in expected_indices}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "x"
    written = 0
    retained = sum(bool(row["retained"]) for row in existing.values())
    with output_file.open(mode, encoding="utf-8") as handle:
        pending = [example for example in examples if example.index not in existing]
        for example in tqdm(pending, desc="MC2 unimodal screening"):
            with ExitStack() as stack:
                source = stack.enter_context(Image.open(example.image_path))
                image = source.convert("RGB")
                stack.callback(image.close)
                result = diagnoser.screen(image, example)
            row = _screening_record(example, result)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            written += 1
            retained += int(result.retained)
    return {
        "total": len(existing) + written,
        "written": written,
        "resumed": len(existing),
        "retained": retained,
    }


def run_activation_patching(
    diagnoser: ActivationPatchingDiagnoser,
    examples: list[MC2PairedExample],
    screening_rows: dict[int, dict[str, Any]],
    output_file: Path,
    *,
    conflict_margin_rows: dict[int, dict[str, Any]] | None = None,
    resume: bool = False,
) -> dict[str, int]:
    retained_examples = [
        example
        for example in examples
        if example.index in screening_rows and bool(screening_rows[example.index]["retained"])
    ]
    existing = load_jsonl_by_index(output_file) if resume else {}
    expected_indices = {example.index for example in retained_examples}
    existing = {index: row for index, row in existing.items() if index in expected_indices}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "x"
    written = 0
    with output_file.open(mode, encoding="utf-8") as handle:
        pending = [example for example in retained_examples if example.index not in existing]
        for example in tqdm(pending, desc="MC2 activation patching"):
            with ExitStack() as stack:
                source = stack.enter_context(Image.open(example.image_path))
                image = source.convert("RGB")
                stack.callback(image.close)
                margin_row = (
                    None
                    if conflict_margin_rows is None
                    else conflict_margin_rows.get(example.index)
                )
                if margin_row is None:
                    row = diagnoser.diagnose(
                        image,
                        example,
                        screening_rows[example.index],
                    )
                else:
                    row = diagnoser.diagnose(
                        image,
                        example,
                        screening_rows[example.index],
                        margin_row,
                    )
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            written += 1
    return {
        "total": len(existing) + written,
        "written": written,
        "resumed": len(existing),
        "eligible": len(retained_examples),
    }


def diagnose_conflict_margin(
    diagnoser: ActivationPatchingDiagnoser,
    image: Image.Image,
    example: MC2PairedExample,
    screening: dict[str, Any],
) -> dict[str, Any]:
    conflict = diagnoser._encode(
        image,
        diagnoser._prompt(ConditionName.IMAGE_CONTEXT, example),
    )
    conflict_output = diagnoser._generate(conflict)
    answer_vision_token_ids = diagnoser._answer_token_ids(example.answer_vision)
    answer_text_token_ids = diagnoser._answer_token_ids(example.answer_text)
    margin = diagnoser._margin(
        conflict,
        answer_vision_token_ids,
        answer_text_token_ids,
    )

    imported_output = screening.get("conflict_output")
    reused_imported_labels = (
        imported_output is not None
        and normalize_answer(conflict_output) == normalize_answer(str(imported_output))
    )
    if reused_imported_labels:
        correct_vision = bool(screening["conflict_correct_vision"])
        correct_text = bool(screening["conflict_correct_text"])
        matcher = "imported_hybrid_parser_after_normalized_output_match"
    else:
        correct_vision = contains_answer(conflict_output, example.answer_vision)
        correct_text = contains_answer(conflict_output, example.answer_text)
        matcher = "normalized_reference_contiguous_token_subset"
    return {
        "record_schema_version": "1.0",
        "dataset": "mc2",
        "dataset_index": example.index,
        "sample_id": f"mc2:{example.index}",
        "image_id": example.image_id,
        "image_path": str(example.image_path),
        "task": example.task,
        "question": example.question,
        "context": example.context,
        "answer_vision": example.answer_vision,
        "answer_text": example.answer_text,
        "answer_vision_token_ids": answer_vision_token_ids,
        "answer_text_token_ids": answer_text_token_ids,
        "instruction": PATCHING_INSTRUCTION,
        "answer_score": ANSWER_SCORE,
        "margin_definition": "log_p(answer_vision) - log_p(answer_text)",
        "margin_conflict": margin,
        "conflict_output": conflict_output,
        "conflict_correct_vision": correct_vision,
        "conflict_correct_text": correct_text,
        "conflict_preference": modality_preference(correct_vision, correct_text),
        "conflict_preference_matcher": matcher,
        "imported_conflict_output": imported_output,
        "imported_labels_reused": reused_imported_labels,
    }


def run_conflict_margins(
    diagnoser: ActivationPatchingDiagnoser,
    examples: list[MC2PairedExample],
    screening_rows: dict[int, dict[str, Any]],
    output_file: Path,
    *,
    resume: bool = False,
) -> dict[str, int]:
    retained_examples = [
        example
        for example in examples
        if example.index in screening_rows and bool(screening_rows[example.index]["retained"])
    ]
    existing = load_jsonl_by_index(output_file) if resume else {}
    expected_indices = {example.index for example in retained_examples}
    existing = {index: row for index, row in existing.items() if index in expected_indices}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "x"
    written = 0
    with output_file.open(mode, encoding="utf-8") as handle:
        pending = [example for example in retained_examples if example.index not in existing]
        for example in tqdm(pending, desc="MC2 conflict margins"):
            with ExitStack() as stack:
                source = stack.enter_context(Image.open(example.image_path))
                image = source.convert("RGB")
                stack.callback(image.close)
                row = diagnose_conflict_margin(
                    diagnoser,
                    image,
                    example,
                    screening_rows[example.index],
                )
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            written += 1
    return {
        "total": len(existing) + written,
        "written": written,
        "resumed": len(existing),
        "eligible": len(retained_examples),
    }
