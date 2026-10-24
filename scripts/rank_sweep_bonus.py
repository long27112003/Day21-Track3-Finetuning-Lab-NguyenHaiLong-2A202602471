"""Bonus B4: train r=8, 16, 64 on the same local GPU and frozen settings.

Adapters stay local; results/rank_sweep.json is the small submission artifact.
Each completed point is reusable after interruption.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from labkit import data, evaluate, generate, modeling, train  # noqa: E402
from labkit.config import SPECS, get_tier, training_epochs  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def train_one(rank: int, tier, rows: list[dict]) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    key = f"rank_r{rank}"
    adapter_dir = ROOT / "adapters" / key
    if (adapter_dir / "adapter_model.safetensors").exists():
        print(f"skip training r={rank}: adapter already exists", flush=True)
        report_path = ROOT / "results" / "rank_sweep.json"
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            recorded = next((row for row in previous if row["rank"] == rank), None)
            if recorded and recorded.get("trainable_params"):
                return {**recorded, "adapter": f"adapters/{key}", "reused": True}
        return {"rank": rank, "adapter": f"adapters/{key}", "reused": True}

    spec = dataclasses.replace(SPECS["correct"], key=key, r=rank,
                               alpha=2 * rank, label=f"text-linear r={rank}")
    model, tokenizer = generate.load_base(tier)
    tokenized = data.to_training_dataset(
        tokenizer, rows, max_length=tier.max_length, mask_mode="assistant-only",
    )
    train_ds = Dataset.from_list(tokenized)
    targets = modeling.resolve_target_modules(model, "text-linear")
    trainable_params = modeling.count_lora_params(model, targets, rank)
    steps = train.planned_steps(len(train_ds), tier, training_epochs())
    if steps != 30:
        raise ValueError(f"Rank sweep must match the original 30-step budget, got {steps}")

    desired_sft = train.sft_config_kwargs(
        tier, spec, str(adapter_dir), max_steps=steps,
        num_train_epochs=training_epochs(), mask_mode="assistant-only",
    )
    sft_kwargs, _ = train.filter_kwargs(SFTConfig, desired_sft, label=f"SFTConfig[{key}]")
    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[{key}]",
    )
    trainer = SFTTrainer(
        model=model, args=SFTConfig(**sft_kwargs), train_dataset=train_ds,
        processing_class=tokenizer, peft_config=LoraConfig(**lora_kwargs),
    )
    train.align_trainable_precision(trainer.model)
    started = time.perf_counter()
    result = trainer.train()
    seconds = time.perf_counter() - started
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    measured = {
        "rank": rank, "adapter": f"adapters/{key}", "placement": "text-linear",
        "alpha": 2 * rank, "learning_rate": spec.lr, "max_steps": steps,
        "trainable_params": trainable_params,
        "final_train_loss": float(result.training_loss),
        "train_seconds": round(seconds, 1),
        "peak_vram_gb": generate.peak_vram_gb(),
    }
    print(json.dumps(measured, ensure_ascii=False), flush=True)
    del trainer, model
    generate.free_memory()
    return measured


def score_one(rank: int, tier, target: list[dict]) -> dict:
    from peft import PeftModel

    model, tokenizer = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / f"rank_r{rank}"))
    predictions, latency = generate.generate_batch(
        model, tokenizer, [row["input"] for row in target],
        system=generate.NAIVE_PROMPT, label=f"rank-r{rank}",
    )
    accuracy = sum(evaluate.triage_field_accuracy(pred, row["label"])
                   for pred, row in zip(predictions, target)) / len(target)
    del model
    generate.free_memory()
    return {"target": round(accuracy, 4), "latency_ms": round(latency, 1),
            "eval_n": len(target)}


def main() -> None:
    tier = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
    if tier.name != "T4":
        raise ValueError("Use the original T4 tier/model to compare ranks fairly")
    train_rows = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
    target = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    if len(target) != 50:
        raise ValueError("Rank sweep requires the original full 50-item target set")
    report_path = ROOT / "results" / "rank_sweep.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else []
    by_rank = {row["rank"]: row for row in previous}
    results = []
    for rank in (8, 16, 64):
        prior = by_rank.get(rank)
        if (prior and prior.get("trainable_params") and prior.get("eval_n") == 50
                and prior.get("target") is not None
                and (ROOT / "adapters" / f"rank_r{rank}" / "adapter_model.safetensors").exists()):
            results.append({**prior, "adapter": f"adapters/rank_r{rank}"})
            print(f"Reuse complete local r={rank} result", flush=True)
            continue
        measured = train_one(rank, tier, train_rows)
        measured.update(score_one(rank, tier, target))
        measured["adapter"] = f"adapters/rank_r{rank}"
        measured["hardware"] = "NVIDIA GeForce RTX 3060 12 GB"
        results.append(measured)
        report_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
