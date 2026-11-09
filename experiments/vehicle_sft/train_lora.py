"""LoRA 微调车辆控制意图模型（需 GPU；本机无卡时可只跑 --dry-run）。

用法（GPU 机器）：
    pip install transformers peft accelerate datasets
    python train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "你是车载语音助手，需要把用户指令解析成 JSON。"
    '只输出 {"intent": "...", "slots": {...}}，不要解释。'
)


def load_rows(path: Path) -> list[dict]:
    """读取 jsonl 数据集。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_example(tokenizer, row: dict, max_length: int = 512):
    """把一条样本转成 (input_ids, labels)：只在 assistant 片段上计算 loss。"""
    messages = row["messages"]
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    target_text = messages[-1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
    return input_ids, labels


def collate(batch, pad_token_id: int):
    """右侧 padding 的动态 batch。"""
    import torch

    max_len = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, (ids, label) in enumerate(batch):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        labels[row, : len(ids)] = torch.tensor(label)
        attention[row, : len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def dry_run() -> None:
    """不加载模型，只校验数据与输出格式。"""
    rows = load_rows(ROOT / "train.jsonl")
    assert rows and "messages" in rows[0], "train.jsonl 缺少 messages"
    intents = {row["intent"] for row in rows}
    print(f"train={len(rows)} intents={len(intents)}")
    print("example:", json.dumps(rows[0]["messages"][-1]["content"], ensure_ascii=False))
    for name in ("val.jsonl", "test.jsonl"):
        print(name, len(load_rows(ROOT / name)))


def train(args: argparse.Namespace) -> None:
    """执行 LoRA 训练。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    rows = load_rows(ROOT / "train.jsonl")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            batch = collate([build_example(tokenizer, row) for row in chunk], tokenizer.pad_token_id)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss.item())
        print(f"epoch={epoch + 1} loss={total / max(1, (len(rows) + args.batch_size - 1) // args.batch_size):.4f}")

    out = ROOT / "lora-out"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print("saved:", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
        return
    train(args)


if __name__ == "__main__":
    main()
