# VisionCite（证源）

多模态文档问答，答案带**可核对的位置证据**。上传 PDF 或图片，用自然语言提问，
系统要么给出答案并高亮它在页面上的位置，要么明确说"证据不足，不作答"。

英文说明见 [README.md](README.md)。

## 它解决什么

大部分"多模态 RAG"演示止步于"模型说了句像样的话"。这个项目补的是落地必需的三件事：

1. **证据是坐标，不是文字**。每条引用带 bbox，前端画在页面上，评测用 IoU 对齐标准框；
2. **答案先被检查再展示**。引用必须存在、答案里的数字必须出现在被引文本里、词面支撑度达标，
   否则弃答 —— 宁可少答，不可答错；
3. **检索过程可解释**。中文按字 bigram 走 BM25，CLIP 做视觉召回，RRF 融合（不需要分数尺度一致），
   MMR 去冗余。

缺依赖时逐级降级：没有 OCR 就只读 PDF 文本层；没有 CLIP 就纯文本检索；没有 VLM 就走抽取式。
当前能力由 `/health` 如实报告。

**还能回答视觉属性**："这段字是什么颜色 / 底色是什么 / 印章什么色" 走的是引用区域的像素统计
（HSV 分桶），不是靠 OCR 文本猜；印章问题会先定位红色区域再统计。

## 快速开始

```bash
pip install -e ".[dev]"           # 核心 + 测试
pip install -e ".[ocr,visual]"    # 可选：离线 OCR 与 CLIP 检索

pytest -q                         # 82 个用例，完全离线
uvicorn visioncite.api.app:app --port 8080
```

命令行：

```bash
visioncite ingest 文档.pdf
visioncite ask 文档.pdf "合计金额是多少？"
visioncite serve --port 8080
```

## 实测指标（自建合成基准）

36 道题 / 8 张渲染页面（发票、表格、图表、幻灯片），CPU-only，`python -m visioncite.eval.run --mode all`：

| 模式 | 答题率 | AnswerAcc | Cite-P | Cite-R | 弃答准确 | ECE | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| page（不切版面） | 0.83 | 0.62 | 0.00 | 0.00 | 0.83 | 0.34 | 1 ms | 1 ms |
| layout（XY-cut + BM25） | 0.78 | 1.00 | 0.85 | 1.00 | 0.89 | 0.17 | 0 ms | 2 ms |
| hybrid（+ CLIP） | 0.78 | 1.00 | 0.85 | 1.00 | 0.89 | 0.17 | 30 ms | 33 ms |

**数值型问题答"值"而不是"整行"**：问"成绩多少"时返回 `88`，靠表头把"百分制/绩点/学分"
对齐到列；编码类数字（`INV-2026-100`）不会被误当成答案。

**怎么读这些数字**：

- `AnswerAcc` 只统计"可答且实际作答"的题；答题率 0.78 意味着 22% 的题被主动弃答；
- `page` 模式没有引用能力（Cite 全 0），这一列就是版面切分的价值；
- 这个语料上 CLIP **只增加延迟、没提升质量**：查询以文字为主，CLIP 读不了小字。
  图表类问题它本该有用，但样本太少不足以证明；
- 95% bootstrap 区间退化成 `[1.00, 1.00]`（24 个已答可答题全对），**不要把它当成稳定性**。

语料是程序合成的：干净背景、端正页面、无手写。它是回归基准，不代表真实扫描件难度。
详见 [docs/LIMITATIONS.md](docs/LIMITATIONS.md)。

## 学习材料

| 文件 | 内容 |
|---|---|
| [docs/CURRICULUM.md](docs/CURRICULUM.md) | 12 讲：原理 → 代码 → 手算例子 → 自测 |
| [docs/CODE_TOUR.md](docs/CODE_TOUR.md) | 8 站代码阅读路线 |
| [docs/EXERCISES.md](docs/EXERCISES.md) | 12 个动手练习（含验收标准） |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层、数据流、上生产要改什么 |
| [docs/EVALUATION.md](docs/EVALUATION.md) | 指标定义与复现方式 |
| [docs/INTERVIEW.md](docs/INTERVIEW.md) | 30 个面试深挖问答（带代码位置） |
| [docs/RESUME.md](docs/RESUME.md) | 简历 bullet 与口径说明 |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | 已知边界 |

## 不在范围内

模型训练、多租户、鉴权、爬虫、视频时间轴检索。见 [ROADMAP.md](ROADMAP.md)。

## 许可

MIT，见 [LICENSE](LICENSE)。
