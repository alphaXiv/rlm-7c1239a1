from __future__ import annotations

import json
import os
import random
import re
import statistics
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
RESULT_DIR = Path("/tmp/rlm_repro_results")
CONFIG_PATH = Path(__file__).with_name("config.json")
CATEGORIES = ("amber", "blue", "coral", "green")
TRAIN_EXAMPLES = 2048
TRAIN_MIN_RECORDS = 4
TRAIN_MAX_RECORDS = 4
SHORT_RECORDS = 4
LONG_RECORDS = 131072
SHORT_EVAL_EXAMPLES = 4096
LONG_EVAL_EXAMPLES = 32
CHUNK_SIZE = 4
EPOCHS = 5
BATCH_SIZE = 8


def make_example(rng: random.Random, n_records: int) -> tuple[str, int, list[str]]:
    target = rng.choice(CATEGORIES)
    labels = [rng.choice(CATEGORIES) for _ in range(n_records)]
    records = [f"record {i + 1}: category={label}" for i, label in enumerate(labels)]
    return format_prompt(records, target), labels.count(target), records


def format_prompt(records: list[str], target: str) -> str:
    body = "\n".join(records)
    return (
        "Count how many records have the requested category. "
        "Reply with only the integer count.\n"
        f"requested category: {target}\nrecords:\n{body}\ncount:"
    )


def encode_training_example(tokenizer, prompt: str, answer: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    answer_ids = tokenizer(f" {answer}{tokenizer.eos_token}", add_special_tokens=False).input_ids
    input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
    labels = torch.tensor([-100] * len(prompt_ids) + answer_ids, dtype=torch.long)
    return input_ids, labels


def make_batch(examples, tokenizer, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = [encode_training_example(tokenizer, prompt, answer) for prompt, answer in examples]
    input_ids = pad_sequence(
        [item[0] for item in encoded],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels = pad_sequence([item[1] for item in encoded], batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def train_model(model, tokenizer, rng: random.Random, device: torch.device) -> list[float]:
    examples = [
        make_example(rng, rng.randint(TRAIN_MIN_RECORDS, TRAIN_MAX_RECORDS))[:2]
        for _ in range(TRAIN_EXAMPLES)
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    losses: list[float] = []
    model.train()
    for _ in range(EPOCHS):
        rng.shuffle(examples)
        for start in range(0, len(examples), BATCH_SIZE):
            batch = make_batch(examples[start : start + BATCH_SIZE], tokenizer, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return losses


def parse_count(text: str) -> int | None:
    match = re.search(r"-?\d+", text.strip())
    return int(match.group()) if match else None


@torch.inference_mode()
def predict(model, tokenizer, prompts: list[str], device: torch.device) -> list[int | None]:
    predictions: list[int | None] = []
    model.eval()
    for start in range(0, len(prompts), 32):
        batch_prompts = prompts[start : start + 32]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        suffixes = generated[:, encoded.input_ids.shape[1] :]
        predictions.extend(parse_count(text) for text in tokenizer.batch_decode(suffixes))
    return predictions


def evaluate_direct(model, tokenizer, examples, device: torch.device) -> dict[str, float]:
    predictions = predict(model, tokenizer, [item[0] for item in examples], device)
    targets = [item[1] for item in examples]
    exact = [prediction == target for prediction, target in zip(predictions, targets, strict=True)]
    absolute_errors = [
        abs(prediction - target) if prediction is not None else LONG_RECORDS
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    return {
        "accuracy": sum(exact) / len(exact),
        "mae": sum(absolute_errors) / len(absolute_errors),
        "parse_rate": sum(prediction is not None for prediction in predictions) / len(predictions),
    }


def evaluate_mapreduce(model, tokenizer, examples, device: torch.device) -> dict[str, float]:
    chunk_prompts: list[str] = []
    chunk_counts: list[int] = []
    targets: list[int] = []
    for _, target_count, records, target in examples:
        targets.append(target_count)
        chunks = [records[start : start + CHUNK_SIZE] for start in range(0, len(records), CHUNK_SIZE)]
        chunk_counts.append(len(chunks))
        chunk_prompts.extend(
            format_prompt(
                [
                    re.sub(r"^record \d+:", f"record {index + 1}:", record)
                    for index, record in enumerate(chunk)
                ],
                target,
            )
            for chunk in chunks
        )

    chunk_predictions = predict(model, tokenizer, chunk_prompts, device)
    predictions: list[int | None] = []
    offset = 0
    for count in chunk_counts:
        values = chunk_predictions[offset : offset + count]
        predictions.append(sum(values) if all(value is not None for value in values) else None)
        offset += count
    exact = [prediction == target for prediction, target in zip(predictions, targets, strict=True)]
    absolute_errors = [
        abs(prediction - target) if prediction is not None else LONG_RECORDS
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    return {
        "accuracy": sum(exact) / len(exact),
        "mae": sum(absolute_errors) / len(absolute_errors),
        "parse_rate": sum(prediction is not None for prediction in predictions) / len(predictions),
    }


def build_eval_examples(rng: random.Random, n_records: int, n_examples: int):
    examples = []
    for _ in range(n_examples):
        target = rng.choice(CATEGORIES)
        labels = [rng.choice(CATEGORIES) for _ in range(n_records)]
        records = [f"record {i + 1}: category={label}" for i, label in enumerate(labels)]
        examples.append((format_prompt(records, target), labels.count(target), records, target))
    return examples


def run_rank(rank: int, world_size: int) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text())
    harness = config["harness"]
    seed = 9001 + rank
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    started = time.time()
    losses = train_model(model, tokenizer, rng, device)
    short_examples = build_eval_examples(
        random.Random(seed + 10_000), SHORT_RECORDS, SHORT_EVAL_EXAMPLES
    )
    long_examples = build_eval_examples(
        random.Random(seed + 20_000), LONG_RECORDS, LONG_EVAL_EXAMPLES
    )
    short_metrics = evaluate_direct(model, tokenizer, short_examples, device)
    if harness == "direct":
        long_metrics = evaluate_direct(model, tokenizer, long_examples, device)
    elif harness == "mapreduce":
        long_metrics = evaluate_mapreduce(model, tokenizer, long_examples, device)
    else:
        raise ValueError(f"Unknown harness: {harness}")

    return {
        "rank": rank,
        "world_size": world_size,
        "seed": seed,
        "model": MODEL_NAME,
        "harness": harness,
        "train_examples": TRAIN_EXAMPLES,
        "train_length_range": [TRAIN_MIN_RECORDS, TRAIN_MAX_RECORDS],
        "short_eval_records": SHORT_RECORDS,
        "short_eval_examples": SHORT_EVAL_EXAMPLES,
        "long_eval_records": LONG_RECORDS,
        "long_eval_examples": LONG_EVAL_EXAMPLES,
        "length_ratio": LONG_RECORDS / SHORT_RECORDS,
        "final_train_loss": losses[-1],
        "mean_last_10_train_loss": statistics.mean(losses[-10:]),
        "short": short_metrics,
        "long": long_metrics,
        "generalization_ratio": long_metrics["accuracy"] / max(short_metrics["accuracy"], 1e-9),
        "elapsed_seconds": time.time() - started,
    }


def aggregate(world_size: int) -> None:
    paths = [RESULT_DIR / f"rank_{rank}.json" for rank in range(world_size)]
    deadline = time.time() + 900
    while not all(path.exists() for path in paths):
        if time.time() > deadline:
            missing = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(f"Timed out waiting for rank results: {missing}")
        time.sleep(2)

    results = [json.loads(path.read_text()) for path in paths]
    metric_paths = (
        ("final_train_loss",),
        ("short", "accuracy"),
        ("short", "mae"),
        ("long", "accuracy"),
        ("long", "mae"),
        ("generalization_ratio",),
        ("elapsed_seconds",),
    )
    summary: dict[str, object] = {
        "experiment": "short-to-32x-length compositional counting",
        "harness": results[0]["harness"],
        "model": MODEL_NAME,
        "seeds": world_size,
        "per_seed": results,
    }
    aggregate_metrics: dict[str, dict[str, float]] = {}
    for path in metric_paths:
        values = []
        for result in results:
            value = result
            for key in path:
                value = value[key]
            values.append(float(value))
        name = ".".join(path)
        aggregate_metrics[name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    summary["aggregate"] = aggregate_metrics
    print("ORX_REPRO_RESULT_BEGIN", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("ORX_REPRO_RESULT_END", flush=True)


def main() -> None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_rank(rank, world_size)
    result_path = RESULT_DIR / f"rank_{rank}.json"
    result_path.write_text(json.dumps(result))
    print(f"rank={rank} seed={result['seed']} result={json.dumps(result, sort_keys=True)}", flush=True)
    if rank == 0:
        aggregate(world_size)


if __name__ == "__main__":
    main()
