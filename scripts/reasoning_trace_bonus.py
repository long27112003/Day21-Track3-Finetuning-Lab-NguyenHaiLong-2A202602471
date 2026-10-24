"""Bonus B3: controlled reasoning-mask comparison on the same thinking base.

The core triage data contains only bare JSON responses, so switching MASK_MODE there
would be a no-op. This script derives explicit, short rationale spans from the *train*
labels and ticket text, then runs two otherwise identical LoRA trainings. The traces
are synthetic teaching examples, not human explanations or a correctness oracle.
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
from labkit.config import NAIVE_PROMPT, SPECS, get_tier, training_epochs  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def make_trace_rows(rows: list[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        label = row["label"]
        ticket = row["input"].strip()
        product = str(label["product"])
        # State observable evidence and the classification decision. Do not pretend
        # these templated rationales were written by people or produced by the model.
        rationale = (
            f"Ticket nhắc tới {product}. Nội dung: {ticket} "
            f"Từ yêu cầu đó, tôi chọn ý định {label['intent']}, "
            f"độ khẩn {label['urgency']} và sắc thái {label['sentiment']}."
        )
        prepared.append({
            "input": ticket,
            "output": f"<think>{rationale}</think>{row['output']}",
            "label": label,
        })
    return prepared


def train_mode(mode: str, tier, rows: list[dict]) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    key = f"trace_{mode.replace('-', '_')}"
    adapter_dir = ROOT / "adapters" / key
    if (adapter_dir / "adapter_model.safetensors").exists():
        print(f"Reuse saved adapter: {adapter_dir}", flush=True)
        report_path = ROOT / "results" / "reasoning_trace_bonus.json"
        if report_path.exists():
            prior = json.loads(report_path.read_text(encoding="utf-8"))
            recorded = next((r for r in prior.get("runs", []) if r["mode"] == mode), None)
            if recorded and recorded.get("train_loss") is not None:
                return {**recorded, "adapter": f"adapters/{key}", "reused": True}
        return {"mode": mode, "adapter": f"adapters/{key}", "reused": True}

    spec = dataclasses.replace(SPECS["correct"], key=key, label=f"trace {mode}")
    model, tokenizer = generate.load_base(tier)
    trace_check = data.thinking_survives(tokenizer)
    if not trace_check["ok"]:
        raise RuntimeError(f"Chat template removed reasoning: {trace_check['verdict']}")
    first = data.to_messages(rows[0])
    keep = data.build_example(tokenizer, first, tier.max_length,
                              mask_mode="assistant-only", enable_thinking=True)
    drop = data.build_example(tokenizer, first, tier.max_length,
                              mask_mode="response-only", enable_thinking=True)
    if keep.n_supervised <= drop.n_supervised:
        raise RuntimeError("Mask modes supervise the same tokens; experiment invalid")
    encoded = data.to_training_dataset(tokenizer, rows, max_length=tier.max_length,
                                       mask_mode=mode, enable_thinking=True)
    targets = modeling.resolve_target_modules(model, "text-linear")
    steps = train.planned_steps(len(encoded), tier, training_epochs())
    if steps != 30:
        raise ValueError(f"B3 must match original 30 optimizer steps, got {steps}")
    sft_kwargs, _ = train.filter_kwargs(
        SFTConfig,
        train.sft_config_kwargs(tier, spec, str(adapter_dir), max_steps=steps,
                                num_train_epochs=training_epochs(), mask_mode=mode),
        label=f"SFTConfig[{key}]",
    )
    lora_kwargs, _ = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label=f"LoraConfig[{key}]",
    )
    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                         train_dataset=Dataset.from_list(encoded),
                         processing_class=tokenizer, peft_config=LoraConfig(**lora_kwargs))
    train.align_trainable_precision(trainer.model)
    started = time.perf_counter()
    result = trainer.train()
    seconds = time.perf_counter() - started
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {
        "mode": mode, "adapter": f"adapters/{key}", "model": tier.model_id,
        "max_steps": steps, "train_rows": len(rows),
        "synthetic_trace": True, "train_loss": float(result.training_loss),
        "train_seconds": round(seconds, 1),
        "first_mask_supervised_tokens": keep.n_supervised if mode == "assistant-only"
        else drop.n_supervised,
        "first_mask_assistant_only_tokens": keep.n_supervised,
        "first_mask_response_only_tokens": drop.n_supervised,
    }
    del trainer, model
    generate.free_memory()
    return metrics


def score_mode(mode: str, tier, holdout: list[dict]) -> dict:
    from peft import PeftModel

    key = f"trace_{mode.replace('-', '_')}"
    prompts = [row["input"] for row in holdout]
    model, tokenizer = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / key))
    # The thinking chat template ends with an opening <think> in the prompt.
    # Prefix it to the decoded continuation for the rubric's trace validator.
    prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": NAIVE_PROMPT},
         {"role": "user", "content": prompts[0]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    if not prefix.rstrip().endswith("<think>"):
        raise RuntimeError("Thinking prompt does not end with <think>; adjust metric")
    outputs, latency = generate.generate_batch(
        model, tokenizer, prompts, system=NAIVE_PROMPT, max_new_tokens=220,
        enable_thinking=True, batch_size=1, label=key,
    )
    scored = ["<think>" + output for output in outputs]
    rate = sum(evaluate.valid_reasoning_trace(x) for x in scored) / len(scored)
    target_accuracy = sum(
        evaluate.triage_field_accuracy(output, row["label"])
        for output, row in zip(scored, holdout)
    ) / len(scored)
    result = {
        "mode": mode, "valid_trace_rate": round(rate, 4),
        "target_field_accuracy": round(target_accuracy, 4),
        "eval_n": len(prompts), "latency_ms": round(latency, 1),
        "outputs": scored,
    }
    del model
    generate.free_memory()
    return result


def main() -> None:
    tier = get_tier(os.environ.get("COMPUTE_TIER", "T4"))
    if tier.name != "T4":
        raise ValueError("B3 requires the original T4 tier and thinking base")
    train_rows = load_jsonl(ROOT / "data" / "split" / "train.jsonl")
    holdout = load_jsonl(ROOT / "data" / "eval_target.jsonl")
    if len(train_rows) != 225 or len(holdout) != 50:
        raise ValueError("Expected frozen original 225/50 data; refusing changed split")
    rows = make_trace_rows(train_rows)
    # Keep a fixed 20-item holdout for this slow, 220-token generation comparison.
    eval_rows = holdout[:20]
    report_path = ROOT / "results" / "reasoning_trace_bonus.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    by_mode = {r["mode"]: r for r in previous.get("runs", [])}
    results = []

    def persist() -> None:
        report_path.write_text(json.dumps({
            "design": "same base, data, text-linear r=16, seed, LR and 30 steps; mask only changes",
            "data_provenance": "templated synthetic rationales from the original train split",
            "eval": "first 20 frozen target prompts, thinking enabled, greedy decode",
            "runs": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    for mode in ("assistant-only", "response-only"):
        prior = by_mode.get(mode)
        key = f"trace_{mode.replace('-', '_')}"
        if (prior and prior.get("train_loss") is not None
                and prior.get("valid_trace_rate") is not None and prior.get("eval_n") == 20
                and (ROOT / "adapters" / key / "adapter_model.safetensors").exists()):
            results.append({**prior, "adapter": f"adapters/{key}"})
            print(f"Reuse complete {mode} result", flush=True)
            continue
        measured = train_mode(mode, tier, rows)
        results.append(measured)
        persist()  # The expensive training evidence survives an interrupted evaluation.
        measured.update(score_mode(mode, tier, eval_rows))
        persist()
        print(f"{mode}: valid_trace_rate={measured['valid_trace_rate']}", flush=True)


if __name__ == "__main__":
    main()
