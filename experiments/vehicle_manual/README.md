# 车辆用户手册 RAG 实验

把 VisionCite 的 RAG 能力搬到"车辆用户手册"场景，验证"知识库构建 → 混合检索 → 带引用作答 → 评测"这条链路。

## 数据

- `generate.py` 生成 **12 页**合成车辆手册（空调、座椅、车窗、灯光、导航、多媒体、驾驶辅助、充电、保养、故障码），
  答案与证据框由绘制坐标精确得到。
- `items.jsonl`：**28 题**（24 可答 + 4 不可答），覆盖 12 页。

```bash
python experiments/vehicle_manual/generate.py
```

## 运行

```bash
# 需要 PYTHONPATH=src 与 VISIONCITE_CACHE
python experiments/vehicle_manual/run_eval.py          # BM25 文本检索
python experiments/vehicle_manual/run_eval.py --visual # 加 CLIP 视觉召回
```

## 结果（BM25 模式，2026-11-09）

| 答题率 | AnswerAcc | Cite-P | Cite-R | 弃答准确率 | F1 | p50 |
|---|---|---|---|---|---|---|
| 0.89 | 0.96 | 0.94 | 0.96 | 0.96 | 0.95 | 0.4 ms |

逐样本记录在 `reports/*.json`，失败样本列在 `reports/*.md`。

## 已知问题

- 手册是模板化生成的，句式规整、无手写/倾斜，真实车主手册更难；
- 问题只覆盖单页单句，没有跨页汇总类问题（"首保和常规保养有什么区别"）；
- 未接 VLM，图表类内容（如仪表指示灯示意图）只能靠文字。
