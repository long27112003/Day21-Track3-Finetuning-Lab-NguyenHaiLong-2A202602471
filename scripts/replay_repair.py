"""Supplementary Day21 experiment: add 1–5% general replay and remeasure.

Run on the same GPU/tier used for NB3 after NB1 has created data/split/train.jsonl.
Writes only adapters/correct_replay/ and results/replay_verdict.json. The original
four-run comparison and frozen baseline remain untouched.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import data, device, evaluate as ev, generate, modeling, report, train
from labkit.config import SPECS, get_tier, training_epochs


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build_replay_rows(train_rows: list[dict], replay_rows: list[dict],
                      regression_rows: list[dict]) -> list[dict]:
    """Keep the evaluation set frozen and reject exact prompt leakage."""
    if not train_rows or not replay_rows:
        raise ValueError("Training and replay sets must both be nonempty")
    ratio = len(replay_rows) / (len(train_rows) + len(replay_rows))
    if not 0.01 <= ratio <= 0.05:
        raise ValueError(f"Replay fraction {ratio:.3%} must be between 1% and 5%")
    held_out = {r["instruction"].strip().casefold() for r in regression_rows}
    prompts = [r["instruction"].strip().casefold() for r in replay_rows]
    if len(prompts) != len(set(prompts)) or set(prompts) & held_out:
        raise ValueError("Replay prompts duplicate each other or the regression holdout")
    return train_rows + replay_rows


def main() -> None:
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer

    tier = get_tier()
    spec = SPECS["correct"]
    original = load_jsonl(ROOT / "data/split/train.jsonl")
    replay = load_jsonl(ROOT / "data/replay_seed.jsonl")
    regression = load_jsonl(ROOT / "data/eval_regression.jsonl")
    frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text(encoding="utf-8"))
    if frozen.get("tier") != tier.name or frozen.get("model") != tier.model_id:
        raise RuntimeError("Replay tier/model does not match the frozen baseline")
    if frozen.get("n_regression") != len(regression) or frozen.get("smoke_mode"):
        raise RuntimeError("Replay requires the full frozen regression holdout")
    combined = build_replay_rows(original, replay, regression)
    print(f"Replay: {len(replay)}/{len(combined)} ({len(replay)/len(combined):.2%})")

    model, tok = generate.load_base(tier, load_in_4bit=spec.load_in_4bit)
    targets = modeling.resolve_target_modules(model, spec.target)
    rows = data.to_training_dataset(tok, combined, max_length=tier.max_length,
                                    mask_mode="assistant-only")
    if len(rows) != len(combined):
        raise RuntimeError("Some replay examples lost supervised tokens")
    # Same optimizer-step budget as the original NB3. The replay run is explicitly
    # supplementary and never replaces the original matched NB3/NB4 comparison.
    steps = train.planned_steps(len(original), tier, training_epochs())
    desired = train.sft_config_kwargs(
        tier, spec, output_dir=str(ROOT / "adapters/correct_replay"),
        max_steps=steps, total_steps=steps, mask_mode="assistant-only")
    sft_kwargs, dropped = train.filter_kwargs(SFTConfig, desired, label="SFTConfig")
    if dropped:
        raise RuntimeError(f"Unsupported training options: {dropped}")
    lora_kwargs, dropped = train.filter_kwargs(
        LoraConfig, train.lora_config_kwargs(spec, targets), label="LoraConfig")
    if dropped:
        raise RuntimeError(f"Unsupported LoRA options: {dropped}")
    trainer = SFTTrainer(model=model, args=SFTConfig(**sft_kwargs),
                         train_dataset=Dataset.from_list(rows), processing_class=tok,
                         peft_config=LoraConfig(**lora_kwargs))
    train.align_trainable_precision(trainer.model)
    started = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - started
    adapter = ROOT / "adapters/correct_replay"
    trainer.model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    del trainer, model
    generate.free_memory()

    model, tok = generate.load_base(tier, load_in_4bit=spec.load_in_4bit)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    target = load_jsonl(ROOT / "data/eval_target.jsonl")
    if frozen.get("n_target") != len(target):
        raise RuntimeError("Frozen baseline uses a different target slice")
    base_b = ev.GroupScores(**{k: v for k, v in frozen["baseline_b"].items() if k != "extra"})
    predictions, latency = generate.generate_batch(
        model, tok, [r["input"] for r in target], system=generate.NAIVE_PROMPT,
        label="replay/target")
    regression_predictions, _ = generate.generate_batch(
        model, tok, [r["instruction"] for r in regression], system=None,
        max_new_tokens=96, label="replay/regression")
    scores = ev.GroupScores(
        target=sum(ev.triage_field_accuracy(p, r["label"])
                   for p, r in zip(predictions, target)) / len(target),
        regression=sum(ev.keyword_recall(p, r["keywords"])
                       for p, r in zip(regression_predictions, regression)) / len(regression),
        format=sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in predictions) / len(target),
        latency_ms=latency, n=len(target))
    verdict = ev.regression_gate(scores, base_b)
    report.write_json({
        "experiment": "1-5% held-out-safe general replay",
        "replay_count": len(replay), "train_count": len(original), "max_steps": steps,
        "train_seconds": round(elapsed, 1), "train_loss": round(result.training_loss, 4),
        "optimized_baseline": base_b.as_dict(), "replay_scores": scores.as_dict(),
        "verdict": verdict.as_dict(),
        "regression_predictions": regression_predictions,
    }, "replay_verdict.json", results_dir=ROOT / "results")
    print("Replay verdict:", verdict.as_dict())


if __name__ == "__main__":
    main()
