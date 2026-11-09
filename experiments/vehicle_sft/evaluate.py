"""评测车辆意图模型：意图准确率、槽位完全匹配率、JSON 合法率。

用法（GPU）：
    python evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter lora-out
    python evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct          # 基座对照
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_rows(path: Path) -> list[dict]:
    """读取测试集。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_output(text: str) -> dict | None:
    """从模型输出里抽取 JSON。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def evaluate(model_name: str, adapter: str | None, limit: int) -> None:
    """加载模型并评测。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    rows = load_rows(ROOT / "test.jsonl")[:limit]
    intent_ok = slot_ok = valid = 0
    started = time.perf_counter()
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        prediction = parse_output(text)
        if prediction is not None:
            valid += 1
        gold = json.loads(row["messages"][-1]["content"])
        if prediction and prediction.get("intent") == gold["intent"]:
            intent_ok += 1
        if prediction and prediction.get("slots") == gold["slots"]:
            slot_ok += 1
    total = len(rows)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "n": total,
                "json_valid": round(valid / total, 3),
                "intent_acc": round(intent_ok / total, 3),
                "slot_exact": round(slot_ok / total, 3),
                "avg_latency_ms": round(elapsed / total * 1000, 1),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=48)
    args = parser.parse_args()
    evaluate(args.model, args.adapter, args.limit)


if __name__ == "__main__":
    main()
