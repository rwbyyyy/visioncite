# 从零开始学 VisionCite（7 天路线）

> 目标：7 天后你能**独立讲清**这个项目的每个设计决定，并至少完成 3 个动手改动。
> 每天 1.5–2 小时，按顺序做，不要跳步。

## 第 0 天（30 分钟）：先跑起来，别急着读代码

**动作**

```powershell
cd C:\Users\VRVillenial\Documents\deepseek-harness\default-workspace\visioncite
$env:PYTHONPATH="src"; $env:VISIONCITE_CACHE="E:\visioncite-cache"

# 1) 测试：看到 119 passed，确认环境和项目是好的
E:\venvs\visioncite\Scripts\python.exe -m pytest -q

# 2) 起服务，打开 http://127.0.0.1:8080
E:\venvs\visioncite\Scripts\python.exe -m uvicorn visioncite.api.app:app --port 8080

# 3) 命令行问一次（观察"答案 + 证据框"）
E:\venvs\visioncite\Scripts\python.exe -m visioncite.cli ask corpus\images\invoice-0.png "合计金额是多少？"

# 4) Agent 跑一次（观察工具轨迹）
E:\venvs\visioncite\Scripts\python.exe -m visioncite.cli agent corpus\images\invoice-0.png "合计金额是多少？" --fake
```

**验收**：你能说出"一次问答经过哪几步、每步产出什么"（哪怕只是照着输出念）。

---

## 第 1 天：跑通主线 + 读懂数据契约（2 小时）

**读**（按顺序，共约 600 行）：

1. `README.zh-CN.md`（只看"它能做什么"和"实测指标"两节）
2. `src/visioncite/types.py` — `BBox` / `Region` / `Chunk` / `Citation` / `AnswerResult`
3. `src/visioncite/pipeline.py` — `build_document()` 的 7 个步骤
4. `docs/CODE_TOUR.md` 第 1–2 站

**做**：在 Python 里手动建一条 `BBox(0,0,10,10)` 与 `BBox(5,0,15,10)`，算 IoU，看是不是 1/3。

**验收**：能回答"为什么 bbox 用左闭右开""Region 和 Chunk 有什么区别"。

---

## 第 2 天：版面分析（2 小时）

**读**：`layout/binarize.py` → `layout/xycut.py` → `layout/table.py` → `layout/regions.py`
**配套**：`docs/CURRICULUM.md` 第 2、3、4 讲（有手算例子）

**做**：在 `tests/test_layout.py` 里加一个用例——自己画一张"三行文字 + 一条表格线"的图，断言 XY-cut 切出 ≥3 块、表格线被识别。

**验收**：能画图解释 XY-cut 的递归过程，并说出它在什么页面上会失效。

---

## 第 3 天：检索（2 小时）

**读**：`index/tokenize.py` → `index/chunking.py` → `index/bm25.py` → `index/fuse.py` → `index/retrieve.py`
**配套**：`docs/CURRICULUM.md` 第 6、7、9 讲

**做**：

```python
from visioncite.index.bm25 import BM25Index
idx = BM25Index()
idx.fit([("d1", "合计金额：1286.40 元"), ("d2", "开票日期：2026-10-25")])
print(idx.search("金额"))
```

再手算一遍这个分数，和代码输出对一下。

**验收**：能解释 BM25 的 k1/b 各自控制什么、RRF 为什么不用分数加权。

---

## 第 4 天：作答、校验、弃答（2 小时）

**读**：`answer/extractive.py` → `answer/verify.py` → `answer/abstain.py` → `answer/qa.py`
**配套**：`docs/CURRICULUM.md` 第 10、11 讲

**做**：构造一个"数字被改过"的答案，看 `verify_answer` 返回什么 reasons；再把它改回去。

**验收**：能说清三道校验分别挡什么、为什么顺序不能换。

---

## 第 5 天：评测（2 小时）

**读**：`eval/run.py`、`eval/metrics.py`、`eval/bootstrap.py`，以及 `reports/` 里最新的 md + json
**配套**：`docs/CURRICULUM.md` 第 12 讲

**做**：跑一次 `--mode all`，然后从 `.json` 里挑 3 条记录，手工复算 `contains` / `f1` / `cite_hit`。

**验收**：能解释"为什么 AnswerAcc 1.00 而 EM 可能很低""为什么 CI 会退化成 [1,1]"。

---

## 第 6 天：Agent 与 MCP（第 13 讲，2 小时）

**读**：`llm/fake.py` → `tools/base.py` → `agent/loop.py` → `mcp/server.py`
**配套**：`docs/AGENT_MCP.md`、`docs/INTERVIEW.md` Q31–Q33

**做**：给 `run_agent` 的 fake 脚本加一次 `get_region` 调用，观察轨迹与引用是否变化。

**验收**：能回答"引用为什么不可伪造""同一套工具怎么同时给 Agent/HTTP/MCP 用"。

---

## 第 7 天：动手改 + 自测（2 小时）

**做**（从 `docs/EXERCISES.md` 里挑 2 个，推荐 13 和 1）：

- 练习 13：加一个 `page_text` 工具，验证三处复用
- 练习 1：Otsu 换自适应阈值，跑评测对比

**自测清单**（全部能答出来才算学完）：

1. 一次 `answer_question` 的完整决策树
2. XY-cut 的终止条件与复杂度
3. BM25 公式里每一项的作用
4. 中文为什么用字 bigram，代价是什么
5. RRF 与 MMR 各自解决什么问题
6. 三道校验 + 三级弃答的顺序与理由
7. 三个数字口径：AnswerAcc、coverage、abstain_accuracy
8. 为什么 CI 会退化，这说明什么
9. Agent 的引用如何做到不可伪造
10. MCP 实现了哪几个方法，为什么不做全

---

## 每天的时间分配建议

| 时间 | 内容 |
|---|---|
| 20% | 读文档（CURRICULUM / CODE_TOUR 对应章节） |
| 40% | 读代码（只读路线里指定的文件，不要发散） |
| 30% | 动手（跑命令、加用例、改参数） |
| 10% | 写三句话笔记：今天懂了什么、哪里还糊、明天补什么 |

## 卡住时的三个排查顺序

1. **先跑测试**：`pytest -q tests/test_xxx.py`，看是理解错还是代码错
2. **再打日志**：`pipeline` / `retrieve` / `answer` 三个模块都有 logger，临时加一行比猜快
3. **最后看报告**：`reports/*.json` 的逐样本记录能回答"为什么这题错了"

## 不要做的事

- 不要一上来读 `scheduler`/`eval/templates` 这种边缘文件
- 不要先学 Agent 再学检索——Agent 的工具就是检索/颜色，基础不牢会看不懂
- 不要试图一次读完 12 讲；按天拆分，每讲配套的代码当天读
