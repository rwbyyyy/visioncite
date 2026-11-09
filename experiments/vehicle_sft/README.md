# 车辆控制意图 SFT（LoRA）

对应车载岗位"车辆控制（HVAC/座椅/车窗）+ 大模型微调"的要求：把用户指令解析成结构化 JSON。

## 任务

```
用户：把主驾空调调到 24 度
模型：{"intent": "ac_set_temperature", "slots": {"position": "主驾", "temperature": "24"}}
```

9 类意图：`ac_set_temperature` / `ac_toggle` / `ac_set_fan` / `seat_control` / `window_control` / `navigation` / `media` / `phone_call`（模板化生成，共 8 类 + 负例扩展位）。

## 数据

```bash
python experiments/vehicle_sft/build_dataset.py --count 320 --seed 42
```

- 模板 × 槽位枚举生成，按 8:1:1 切分 train/val/test；
- 唯一指令上限约 259 条（模板组合有限），脚本会打印实际数量，不会死循环；
- 不含任何真实用户数据，可复现。

## 训练（需要 GPU）

```bash
pip install transformers peft accelerate
python experiments/vehicle_sft/train_lora.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --batch-size 4 --lr 2e-4
```

- LoRA：r=8、alpha=16、target=`q_proj`,`v_proj`；
- 只在 assistant 片段上计算 loss（prompt 部分 label=-100）；
- 产物在 `lora-out/`。

## 评测

```bash
# 基座对照
python experiments/vehicle_sft/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct
# LoRA 结果
python experiments/vehicle_sft/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter lora-out
```

输出：`json_valid`（JSON 合法率）、`intent_acc`（意图准确率）、`slot_exact`（槽位完全匹配）、`avg_latency_ms`。

## 结果（A800-80GB，2026-11-09）

训练：207 条训练样本、3 epochs、batch 4、lr 2e-4，loss 0.199 → 0.017 → 0.006（约 36 秒）。
评测：26 条测试样本，贪心解码。

| 模型 | JSON 合法率 | 意图准确率 | 槽位完全匹配 | 平均延迟 |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct（基座） | 1.00 | 0.231 | 0.000 | 1764 ms |
| + LoRA（r=8, α=16, 3 epochs） | 1.00 | **1.000** | **0.846** | 2726 ms |

**怎么读**：

- 基座模型能输出合法 JSON（格式对），但意图只有 23.1%、槽位 0%，说明"会格式 ≠ 会解析指令"；
- LoRA 把意图准确率拉到 100%、槽位完全匹配 84.6%，剩余错误集中在多槽位语序与"关掉/开一下"这类同义表达；
- 延迟上升是 adapter 前向与首次显存分配导致，未做推理优化，不代表线上性能；
- 样本量小（26 条）、模板化数据，不能外推到真实车载语音场景。

## 诚实的边界

- 本仓库的运行环境是 **CPU**，没有在这里跑过真实训练，因此**不提供训练后的准确率数字**；
- 数据是模板生成，覆盖的是"标准说法"，对口语/方言/多轮省略的泛化需要真实数据再测；
- 0.5B 模型只用于验证 SFT 流程；生产应换更大模型并做安全约束（车辆控制指令必须二次确认）。
