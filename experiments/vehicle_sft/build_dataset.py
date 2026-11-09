"""构造车辆控制意图数据集（SFT 用）。

任务定义：用户指令 → JSON `{"intent": "...", "slots": {...}}`
覆盖 9 类意图 + 槽位（位置 / 数值 / 设备 / 媒体 / 目的地），用于 LoRA 微调
Qwen2.5-0.5B-Instruct 后评测"意图准确率 + 槽位完全匹配率"。

生成方式：模板 + 槽位枚举 + 扰动，保证可复现、可检查、不依赖任何外部数据。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POSITIONS = ["主驾", "副驾", "后排左侧", "后排右侧", "全车"]
TEMPERATURES = ["18", "20", "22", "24", "26", "28", "30"]
FAN_LEVELS = ["1", "2", "3", "4", "5", "6", "7"]
SEAT_MODES = ["加热", "通风"]
WINDOW_TARGETS = ["主驾车窗", "副驾车窗", "后排车窗", "天窗"]
MUSIC = ["周杰伦的歌", "轻音乐", "新闻广播", "上一首", "下一首"]
DESTINATIONS = ["西湖", "杭州东站", "最近的服务区", "公司", "家", "机场"]


def sample(rng: random.Random) -> dict[str, object] | None:
    """按意图模板生成一条样本。"""
    intent = rng.choice(
        ["ac_set_temperature", "ac_toggle", "ac_set_fan", "seat_control", "window_control", "navigation", "media", "phone_call"]
    )
    position = rng.choice(POSITIONS)
    if intent == "ac_set_temperature":
        temp = rng.choice(TEMPERATURES)
        text = rng.choice(
            [
                f"把{position}空调调到{temp}度",
                f"{position}空调设成{temp}摄氏度",
                f"温度调到{temp}度，{position}",
            ]
        )
        return {"instruction": text, "intent": intent, "slots": {"position": position, "temperature": temp}}
    if intent == "ac_toggle":
        text = rng.choice([f"打开{position}空调", f"把{position}空调关掉", f"关闭空调，{position}"])
        action = "打开" if "打开" in text else "关闭"
        return {"instruction": text, "intent": intent, "slots": {"position": position, "action": action}}
    if intent == "ac_set_fan":
        level = rng.choice(FAN_LEVELS)
        text = rng.choice([f"风量调到{level}档", f"{position}风速设为{level}", f"把风量调大一点，设成{level}档"])
        return {"instruction": text, "intent": intent, "slots": {"position": position, "fan_level": level}}
    if intent == "seat_control":
        mode = rng.choice(SEAT_MODES)
        text = rng.choice([f"打开{position}座椅{mode}", f"{position}座椅{mode}关掉", f"把{position}座椅{mode}开一下"])
        action = "打开" if "打开" in text or "开一下" in text else "关闭"
        return {"instruction": text, "intent": intent, "slots": {"position": position, "mode": mode, "action": action}}
    if intent == "window_control":
        target = rng.choice(WINDOW_TARGETS)
        text = rng.choice([f"打开{target}", f"关闭{target}", f"把{target}开一半"])
        action = "打开" if "打开" in text else ("关闭" if "关闭" in text else "半开")
        return {"instruction": text, "intent": intent, "slots": {"target": target, "action": action}}
    if intent == "navigation":
        destination = rng.choice(DESTINATIONS)
        text = rng.choice([f"导航到{destination}", f"带我去{destination}", f"去{destination}怎么走"])
        return {"instruction": text, "intent": intent, "slots": {"destination": destination}}
    if intent == "media":
        item = rng.choice(MUSIC)
        text = rng.choice([f"播放{item}", f"来点{item}", f"切到{item}"])
        return {"instruction": text, "intent": intent, "slots": {"media": item}}
    if intent == "phone_call":
        who = rng.choice(["妈妈", "爸爸", "张三", "李四", "公司前台"])
        text = rng.choice([f"给{who}打电话", f"拨打{who}的电话", f"呼叫{who}"])
        return {"instruction": text, "intent": intent, "slots": {"contact": who}}
    return None


SYSTEM_PROMPT = (
    "你是车载语音助手，需要把用户指令解析成 JSON。"
    '只输出 {"intent": "...", "slots": {...}}，不要解释。'
    "intent 取值：" + "、".join(
        [
            "ac_set_temperature",
            "ac_toggle",
            "ac_set_fan",
            "seat_control",
            "window_control",
            "navigation",
            "media",
            "phone_call",
        ]
    )
)


def to_messages(item: dict[str, object]) -> list[dict[str, str]]:
    """转成 chat 格式（训练与评测共用）。"""
    target = json.dumps({"intent": item["intent"], "slots": item["slots"]}, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(item["instruction"])},
        {"role": "assistant", "content": target},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    items = []
    seen = set()
    attempts = 0
    # 模板×槽位的唯一组合有限；达到上限后退化为止损，而不是死循环
    while len(items) < args.count and attempts < args.count * 50:
        attempts += 1
        item = sample(rng)
        if item is None or item["instruction"] in seen:
            continue
        seen.add(item["instruction"])
        items.append(item)
    if len(items) < args.count:
        print(f"warning: 唯一指令只有 {len(items)} 条，低于目标 {args.count}")
    rng.shuffle(items)
    train_end = int(len(items) * 0.8)
    val_end = int(len(items) * 0.9)
    splits = {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }
    for name, rows in splits.items():
        path = ROOT / f"{name}.jsonl"
        path.write_text(
            "\n".join(json.dumps({**row, "messages": to_messages(row)}, ensure_ascii=False) for row in rows)
            + "\n",
            encoding="utf-8",
        )
        print(name, len(rows), "->", path.name)


if __name__ == "__main__":
    main()
