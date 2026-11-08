from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm

from conditions import Condition
from data import BenchmarkExample
from models.base import ModelAdapter, ModelInput
from methods import MethodName, parse_method, transform_prompt


def run_evaluation(
    model: ModelAdapter,
    examples: list[BenchmarkExample],
    conditions: list[Condition],
    output_file: Path,
    *,
    model_name: str,
    seed: int = 42,
    method: str = "base",
    method_config: dict | None = None,
    generation_config: dict | None = None,
    benchmark: str = "mc2",
) -> dict:
    method_name = parse_method(method)
    rows: list[dict] = []
    run_id = str(uuid4())
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8") as handle:
        for condition in conditions:
            progress = tqdm(total=len(examples), desc=condition.name.value)
            for batch_number, example in enumerate(examples):
                batch = [example]
                model_inputs = [
                    ModelInput(
                        prompt=transform_prompt(
                            condition.prompt(example.question, example.context),
                            method_name,
                            method_config,
                        ),
                        image_path=example.image_path if condition.use_image else None,
                        image_blur_radius=(example.image_blur_radius if condition.use_image else 0.0),
                        context=example.context if condition.use_context else None,
                        question=example.question,
                    )
                    for example in batch
                ]
                generated_batch = model.generate(model_inputs)
                predictions = generated_batch.texts
                if len(predictions) != len(batch):
                    raise RuntimeError("Model adapter returned the wrong number of predictions")
                for sample_offset, (example, prompt, prediction) in enumerate(
                    zip(batch, model_inputs, predictions)
                ):
                    row = {
                        "record_schema_version": "1.0",
                        "dataset": benchmark,
                        "dataset_index": example.index,
                        "sample_id": f"{benchmark}:{example.index}",
                        "run_id": run_id,
                        "run_timestamp_utc": run_timestamp_utc,
                        "random_seed": seed,
                        "method": method,
                        "method_config": method_config or {},
                        "generation_config": generation_config or {},
                        "batch_id": f"{run_id}:{condition.name.value}:{batch_number}",
                        "batch_metrics": generated_batch.metrics,
                        "image_id": example.image_id,
                        "image_filename": example.image_path.name,
                        "image_path": str(example.image_path),
                        "image_blur_level": example.image_blur_level,
                        "image_blur_radius": example.image_blur_radius,
                        "task": example.task,
                        "question": example.question,
                        "text_context": example.context,
                        "context_source": example.context_source,
                        "context_is_conflicting": example.context_source in {
                            "context",
                            "raw_question_description",
                            "phd-ica",
                            "phd-icc",
                            "caption",
                            "conflicting_caption",
                        },
                        "model": model_name,
                        "condition": condition.name.value,
                        "condition_uses_image": condition.use_image,
                        "condition_uses_context": condition.use_context,
                        "condition_instruction": condition.instruction,
                        "prompt": prompt.prompt,
                        "inference_result": prediction.strip(),
                    }
                    if example.answer_vision is not None:
                        row["answer_vision"] = example.answer_vision
                    if example.answer_text is not None:
                        row["answer_text"] = example.answer_text
                    if example.metadata:
                        row["benchmark_metadata"] = example.metadata
                    if generated_batch.sample_metrics:
                        row.update(generated_batch.sample_metrics[sample_offset])
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(row)
                progress.update(len(batch))
            progress.close()
    return {
        condition.name.value: {
            "count": sum(row["condition"] == condition.name.value for row in rows)
        }
        for condition in conditions
    }
