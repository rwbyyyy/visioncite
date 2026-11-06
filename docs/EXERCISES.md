# VisionCite 动手练习

12 个练习，每个 20–40 分钟，都和课程的一讲对应。练习的目标不是"改出更好的系统"，而是**让你必须读代码、必须跑评测、必须解释数字为什么变了**。改坏了不算失败，改完不知道数字为什么动才是。

## 开始之前

**环境**：

```bash
uv sync --extra dev
python -m pytest -q          # 基线必须全绿，否则先修环境
```

`pypdfium2` 是核心依赖：`ingest.loader` 在模块顶层导入 `ingest.pdf`，所以即使只处理图片也需要它。`rapidocr-onnxruntime`（OCR）与 `torch`/`transformers`（CLIP）是可选的，没装时系统会静默降级。如果可编辑安装因为缺 `README.md`（`pyproject.toml` 里声明了但仓库没有这个文件）失败，直接用 `PYTHONPATH=src` 跑模块即可。

**评测基线**：做练习前先跑一次 `layout` 模式并记下数字，作为你自己的对照组（`reports/` 为空时尤其要跑）：

```bash
python -m visioncite.eval.run --mode layout --per 2
```

仓库里一次参考运行（36 题 / 8 图）的结果是：

| 指标 | 值 | 说明 |
|---|---|---|
| coverage | 0.72 | 26/36 作答 |
| AnswerAcc | 1.00 | 24 个"可答且已作答"全对 |
| F1 | 0.40 | 答案是整句、gold 是片段，P 被拉低 |
| Cite-P | 0.83 | 每题最多两条引用，第二条常是另一区域 |
| Cite-R | 1.00 | 每题至少一条引用命中 |
| abstain_acc | 0.83 | 30/36 |
| ECE | 0.19 | 明显过自信，练习 4 就是修它 |
| p50 / p95 | 0.30 / 0.64 ms | 纯 CPU、无模型 |

数字会随依赖版本与字体轻微波动，比较时**只比同一台机器上你自己跑出来的两次**。

**一次不要改两件事**：每个练习只动一个变量，否则数字变了也解释不清。建议每次实验前复制一份报告，文件名写清改动（例如 `reports/eval-layout-otsu-adaptive.md`）。

**通用验收模板**：

```bash
python -m pytest -q                                  # 全部回归
python -m pytest -q tests/test_<相关>.py -k <名字>    # 定向验证
python -m visioncite.eval.run --mode layout --per 2   # 指标对比
ruff check src tests                                 # 风格（行宽 100）
```

---

## 练习 1 把 Otsu 换成自适应阈值

**对应**：第 2 讲。**目标**：让版面分析在"光照不均"（一侧偏暗、一侧偏亮）的页面上也能正确二值化，并能量化它相对全局 Otsu 的收益与代价。

**要改的文件**

- `src/visioncite/layout/binarize.py`：新增 `adaptive_threshold(gray, block=31, c=10)`（积分图算局部均值，`ink = gray < local_mean - c`），让 `ink_mask` 支持 `method="otsu"|"adaptive"` 或新增一个 `ink_mask_adaptive`。
- `src/visioncite/layout/regions.py`：`build_regions()` 目前直接 `ink_mask(to_gray(image))`，把方法参数透传进去（或在 `CutConfig` 旁边加一个配置对象，避免函数签名膨胀）。
- `tests/test_layout.py`：新增一个"左半 200、右半 250 且文字为 80"的合成图用例，断言自适应能找回右侧文字、Otsu 找不回。

**验收方式**

1. `python -m pytest -q tests/test_layout.py` 全绿（现有 14 个用例——`test_otsu_separates_bimodal_histogram` 与 `test_ink_mask_marks_dark_pixels` 不能因为新分支而退化）。
2. 新增用例先在你改动**前**跑一次，确认它失败（否则用例没测到东西）。
3. 评测：`python -m visioncite.eval.run --mode layout --per 2`，`Cite-P` 不应下降超过 0.03，`pack/report` 里的块数（日志 `chunks=`）变化要能解释。
4. 写三句话结论：在什么图上有收益、在什么图上变差、为什么（局部窗太小会把字内部当背景）。

**预计耗时**：35–40 分钟（20 分钟写积分图，10 分钟造用例，10 分钟跑评测）。

---

## 练习 2 给 BM25 加字段权重

**对应**：第 7 讲。**目标**：让"表格里的数字"比"正文里顺带提到的数字"更容易被检索到；体会字段权重如何改变排序，而不是无脑调大分数。

**要改的文件**

- `src/visioncite/index/bm25.py`：`fit()` 目前只收 `(chunk_id, text)`。改成可选 `fit(documents, weights=None)`（同一个词在不同文档上乘不同权重），或新增 `fit_weighted(documents, field_weights)`；`search()` 里把权重乘到 `tf_component` 上。注意 `postings` 的结构要能承载"按文档的权重"，最简单的做法是 `tf * weight` 直接写进 `postings`，但要记住这会让 `doc_lengths` 与实际词频脱节，需要一并考虑（这正是练习要你想清楚的地方）。
- `src/visioncite/pipeline.py`：`build_document()` 里 `bm25.fit(...)` 传入每个 chunk 的权重（`chunk.kind == "table"` 给 1.5，其余 1.0）。
- `src/visioncite/eval/run.py`：`_flatten_to_page()` 里同样要重设权重（否则 `page` 模式的行为和 `layout` 模式不可比）。

**验收方式**

1. `python -m pytest -q tests/test_index.py` 全绿；`test_bm25_scores_are_sorted`、`test_bm25_ranks_matching_doc_first` 必须仍然通过。
2. 新增一条用例：两篇文档文字相同、权重不同，断言高分那篇排前。
3. 评测 `--mode layout --per 2`：看报告"分类别 AnswerAcc"里 `table` 与 `invoice` 两列，以及整体 `Cite-P`；如果 `table` 升而 `invoice` 降，说明表格权重开始"抢"正文证据，这本身就是合格结论。
4. 用 `python -c` 打印同一个查询在有/无权重下的 `search()` 前三条，附在笔记里。

**预计耗时**：30–40 分钟。

---

## 练习 3 把 RRF 换成加权分数融合，做 A/B

**对应**：第 9 讲。**目标**：亲手验证"为什么 RRF 更稳"——把分数直接加权求和，看它需要多难调、以及它对离群分数多敏感。

**要改的文件**

- `src/visioncite/index/fuse.py`：新增 `weighted_score_fusion(text_scores, visual_scores, alpha=0.5)`，先把两路分数各自 min-max 归一化到 0~1，再 `alpha*text + (1-alpha)*visual`；保留 `reciprocal_rank_fusion` 不动，方便回退。
- `src/visioncite/index/retrieve.py`：`retrieve()` 里加一个开关（读 `Settings` 的新字段 `fusion_mode: Literal["rrf", "weighted"] = "rrf"`），两种模式共用后面的 `_diversify` 与归一化。
- `src/visioncite/config.py`：新增 `fusion_mode` 与（可选）`fusion_alpha`，并在 `Settings.from_env()` 里留一个 `VISIONCITE_FUSION` 环境变量，方便 A/B 时不用改代码。

**验收方式**

1. `python -m pytest -q tests/test_index.py` 全绿；`test_rrf_boosts_items_in_both_rankings` 必须仍通过（说明旧路径没被破坏）。
2. 给 `weighted_score_fusion` 写 2 条用例：一路全 0、一路全 1 时结果是否合理；某一路出现离群大分数（1e6）时 min-max 是否把它压回 0~1。
3. 只在 `layout` 模式做 A/B 意义不大（视觉路为空）；真正的 A/B 要 `--mode hybrid`，需要装 `torch`/`transformers`（CPU 可跑，首轮会下载 CLIP 权重，属于 `slow` 范畴）。跑不动就退而求其次：构造两路假排名，用 `python -c` 对比两种融合的前三条与分数分布。
4. 结论里必须回答：`alpha` 改 0.1 与 0.9 时排序变化大不大？如果很大，说明这种融合"需要调参"，正是 RRF 想避免的。

**预计耗时**：35–40 分钟（含跑 hybrid 的等待）。

---

## 练习 4 给弃答加温度缩放

**对应**：第 11 讲。**目标**：把"过自信"的置信度校准到与真实准确率一致，用 ECE 而不是"感觉"来验收。

**要改的文件**

- `src/visioncite/answer/extractive.py`：`_confidence(coverage, margin, score)` 现在是线性加权。加一层单调变换：把 `value` 先转 logit（`log(p/(1-p))`，注意 `p` 截断到 `[1e-6, 1-1e-6]`），除以温度 `T`，再 sigmoid 回来；`T` 从 `Settings` 读（新增 `confidence_temperature: float = 1.0`）。
- `src/visioncite/config.py`：新增 `confidence_temperature`，并在 `from_env()` 里给 `VISIONCITE_CONF_TEMPERATURE`。
- `src/visioncite/eval/run.py`（可选）：加一个 `--temperature-sweep` 分支，对 `T in [0.5, 1, 1.5, 2, 3]` 各跑一遍，报告每个 T 的 ECE 与 coverage。

**验收方式**

1. `python -m pytest -q tests/test_answer.py` 全绿；`test_no_abstain_when_confident_and_verified`（阈值 0.3、置信度 0.8）在 `T=1` 时必须仍通过；`T` 很大时这条可能失败——这正是"校准会改变弃答率"的证据，请把它写进结论而不是删用例。
2. 评测前后对比 `ECE`：目标从 0.19 降到 0.12 以下；同时看 `coverage`，如果掉了超过 0.1，说明你把置信度整体压太低、开始误弃答。
3. 用 `python -c "from visioncite.eval.metrics import expected_calibration_error; ..."` 对同一批 `(confidence, correct)` 手工扫几个 T，确认脚本与报告一致。

**预计耗时**：30–40 分钟。

---

## 练习 5 增加一种文件类型（TIFF）

**对应**：第 3 站 ingest。**目标**：走完"新增格式"的完整改动面，理解为什么魔数判断要集中在两张表里。

**要改的文件**

- `src/visioncite/ingest/loader.py`：`_MAGIC_PREFIXES` 加 TIFF 的两种字节序魔数（`b"II*\x00"` 小端、`b"MM\x00*"` 大端），`_SUFFIXES` 加 `.tif` / `.tiff`，`FileKind` 不需要变（TIFF 归 `image`）。
- `src/visioncite/ingest/image.py`：确认 `load_image()` 能处理 TIFF 的多帧——`Image.open()` 默认只读第一帧，如果需要多页 TIFF，得用 `ImageSequence` 逐帧转成多个 `Page`；不想扩到多页，就在 docstring 里写清"只取第一帧"，并在 `load_pages()` 里保持"图片固定 1 页"的约定。
- `src/visioncite/ingest/loader.py`：`load_pages()` 里图片分支不需要改（已按 1 页返回），但要在 docstring 里补一行。
- `tests/test_pipeline.py`：新增 `test_detect_kind_recognizes_tiff`，用 8 字节魔数 + 填充字节造文件，断言 `detect_kind()` 返回 `"image"`。

**验收方式**

1. `python -m pytest -q tests/test_pipeline.py` 全绿；`test_detect_kind_by_magic_bytes`、`test_detect_kind_rejects_unknown` 仍然通过（说明没有把 `.txt` 误判）。
2. 手工验证：用 PIL 生成一张 TIFF（`Image.new("RGB", (200,100), "white").save(p, format="TIFF")`），`load_pages(p, Settings())` 返回 1 个 `Page`，`image.shape == (100, 200, 3)`。
3. 反向验证：把一个 `.tif` 文件内容换成 PNG 字节，`detect_kind()` 应返回 `"image"`（魔数优先），说明扩展名兜底没有抢优先级。

**预计耗时**：25–30 分钟。

---

## 练习 6 把表格区域再切成单元格

**对应**：第 4 讲。**目标**：让表格文本从"整块多行"变成"逐单元格结构"，直接改善 `table` 类样本的 F1（基线只有 0.10，因为整块文本 vs 单个数字）。

**要改的文件**

- `src/visioncite/layout/table.py`：新增 `table_cells(mask, bbox, horizontal, vertical) -> list[list[BBox]]`，用横线/竖线的坐标把表格框切成 `rows × cols` 个单元格。关键点：`find_tables()` 只返回 `TableCandidate(bbox, rows, cols)`，**不返回线的坐标**，所以你要么让它返回 `horizontal`/`vertical` 两个线带列表，要么在 `table_cells()` 内对 `bbox` 里裁剪出的子掩码重新调 `rule_bands()`。注意对整页求占比与对表格局部求占比结果不同（局部裁剪后横线占比接近 1.0，更容易过 0.25 阈值）。
- `src/visioncite/layout/regions.py`：`classify()` 判为 `table` 后，`build_text()` 里按单元格中心点把词分配到格子里，输出仍然保持 `"a | b"` 的行格式（下游检索与 `answer/verify.py` 都吃这个格式，不要改成 JSON）。
- `src/visioncite/eval/templates.py` 不用改：`table` 模板的证据框是整张表（`[70, 200, 930, 550]`），单元格切分不影响 gold。

**验收方式**

1. `python -m pytest -q tests/test_layout.py` 全绿；`test_find_tables_detects_grid` 仍通过。
2. 新增用例：用 `tests/test_layout.py::test_find_tables_detects_grid` 那张 300x300 网格（横线 4 条、竖线 3 条），断言 `table_cells()` 返回 **3 行 × 2 列**共 6 个格子，且每个格子的 `area > 0`、格子之间不重叠。如果你选择"把每条线当成格子的边"，先用 (rows+1) 与 (cols+1) 个边坐标生成网格，再断言数量——把口径写进用例名（例如 `test_table_cells_count_for_grid`）。
3. 评测 `--mode layout --per 2`：`分类别 AnswerAcc` 里 `table` 保持 1.00，`F1` 整体应从 0.40 上升（表格题贡献的 token 更集中）；`Cite-P` 不应下降。
4. 如果 F1 没动，检查 `build_text()` 是否真的被调用了——`_flatten_to_page()`（`page` 模式）会绕过它。

**预计耗时**：35–40 分钟（切网格容易，把词分进格子容易出错，留时间调试）。

---

## 练习 7 给切块加重叠

**对应**：第 6 讲。**目标**：验证"块边界切断答案"的损失有多大，并用数字决定重叠窗口该给多少。

**要改的文件**

- `src/visioncite/index/chunking.py`：`_pack(sentences, max_chars)` 增加 `overlap_chars=0`，装箱时把上一块末尾的若干字符带到下一块开头；`chunk_region()` 暴露参数；`chunks_from_pages()` 透传。
- `src/visioncite/pipeline.py`：`chunks_from_pages(doc_id, pages)` 调用处传参（或从 `Settings` 新增 `chunk_overlap` 读，推荐后者，保持单一配置源）。
- `src/visioncite/config.py`：新增 `chunk_max_chars: int = 240`、`chunk_overlap: int = 0`，并让 `build_document()` 使用它们（目前 `max_chars=240` 硬编码在 `chunking.py` 默认参数里）。

**验收方式**

1. `python -m pytest -q tests/test_index.py` 全绿；`test_chunk_region_splits_long_text` 与 `test_chunk_region_produces_stable_ids`（id 仍为 `"{doc}:{region}:{index}"`）必须通过。
2. 新增用例：同一段长文本，`overlap=0` 与 `overlap=40` 的块数不同、且相邻块有公共子串。
3. 评测扫 3 个点（0 / 40 / 80 字），记录 `F1`、`Cite-P` 与日志里的 `chunks=`：块数增加会抬高 BM25 的 `avgdl` 变化与召回量，你要能解释"为什么 F1 先升后降"（重复证据挤占 `top_k`）。

**预计耗时**：30–40 分钟（含三次评测）。

---

## 练习 8 用 dHash 给上传做近似去重

**对应**：第 3 站 ingest。**目标**：把仓库里已有但没接线的 `perceptual_hash()` 用起来，让"同一张图重新导出一次"也能命中去重；同时亲手量出它在"同模板不同数据"的文档上会误伤到什么程度，从而知道它该做"提示"还是"替代"。

**要改的文件**

- `src/visioncite/ingest/image.py`：`perceptual_hash()` 与 `hash_distance()` 已经实现，不用改；只确认 `size=8`、汉明距离阈值（文档建议 `<= 5`）。
- `src/visioncite/api/app.py`：真正的落盘与解析在 `_store_document(app, data, filename)` 里（`upload()` 只做大小校验然后转调它）。在 `doc_id = document_id_from_bytes(data)` 之后、命中 `documents` 缓存之前，增加"近似重复"检查：解析出页面图 → 算 dHash → 与 `app.state` 里已存文档的哈希比较，命中则返回已有文档并在响应里标注（`DocumentOut` 没有这个字段；可以先只记日志，或用 `HTTPException(409)` 让用户确认——加字段属于契约变更）。
- `src/visioncite/api/schemas.py`：如果加字段，记得同步 `DocumentOut` 与 `app._to_document_out()`。
- `tests/test_api.py`：新增用例，断言"同图不同编码"返回同一 `doc_id` 或标记 `deduped=True`。

**验收方式**

1. `python -m pytest -q tests/test_api.py` 全绿；415 与 413 两条错误用例不受影响。
2. 手工验证（下面的数字是在本仓库 `corpus/images/` 上实测的，可直接复现）：
   - `table-0.png` 与 `table-1.png` 的 dHash 距离是 **0**——模板里表格内容不随 seed 变化，只有坐标固定的一批数字；
   - `invoice-0.png` 与 `invoice-1.png` 距离也是 **0**——只差发票号 `INV-2026-100` / `101`，8x8 的哈希看不见；
   - `invoice-0.png` 与 `table-0.png` 距离是 **17**——不同版式才分得开；
   - 同一张图另存为 JPEG（`quality=90`）再读回，距离 **0**（PNG 是无损的，必须走一次有损编码才测得出鲁棒性）。
3. 用这些数字写结论：在这批"同模板不同数据"的文档上，任何 `<= 5` 的阈值都会把两份不同文档判成同一份。因此 dHash 只能做"疑似重复"的提示（例如返回 `409` 让用户确认），不能像 `document_id_from_bytes()` 的内容哈希那样直接替代解析。宁可漏判（继续走完整解析）也不要误判（把不同文档当成同一份）。

**预计耗时**：30–40 分钟。

---

## 练习 9 修 MMR 的量纲，并做 A/B

**对应**：第 9 讲。**目标**：修掉已知的量纲不匹配（`_diversify` 传的是 0.03 量级的 RRF 分，而多样性项是 0~1 的余弦），看它到底影响多少。

**要改的文件**

- `src/visioncite/index/retrieve.py`：`_diversify()` 里 `relevance` 先归一化（例如除以最大值，或做 min-max），再传给 `mmr_select`；保持 `lambda_` 语义不变。
- `src/visioncite/index/fuse.py`：`mmr_select()` 的文档字符串补一句"relevance 应与余弦同一量纲（0~1）"，避免下一个人踩同一个坑。
- `src/visioncite/config.py`：可选新增 `mmr_normalize: bool = True`，方便回退做 A/B。

**验收方式**

1. `python -m pytest -q tests/test_index.py` 全绿；`test_mmr_prefers_diverse_candidates`（λ=0.5）必须仍通过——注意这条用例的 relevance 已经是 0~1，归一化不改变它，这是"修复不破坏契约"的证据。
2. 构造一个反例证明修复有效：`relevance=[0.03, 0.029, 0.01]`、`embeddings` 中前两个几乎相同，断言"修复后第一个仍被选中、第二个被多样性挤下去"（修复前 `0.7*0.03 - 0.3*0.99` 为负，第三个会被选中）。
3. 评测 `--mode layout --per 2`（无向量时 MMR 走截断分支，数字不会变——这本身是个结论）；要真正验证需 `hybrid`。记录 `Cite-P` 变化与 `top_k` 条证据的 `region_id` 重复率。

**预计耗时**：30–35 分钟。

---

## 练习 10 扫弃答阈值，画 coverage–accuracy 曲线

**对应**：第 11、12 讲。**目标**：不靠拍脑袋定 `abstain_threshold=0.35`，而是给出"阈值 → 答题率 / 准确率 / 弃答准确率"的整条曲线，并带 bootstrap 区间。

**要改的文件**

- `src/visioncite/eval/run.py`：新增 `--sweep` 参数与 `sweep_abstain(corpus, thresholds)`：对每个阈值重建 `Settings(abstain_threshold=t)`（`Settings` 是 frozen，用 `dataclasses.replace` 或直接构造），只跑 `layout` 模式，复用 `evaluate()` 与 `summarize()`。
- `src/visioncite/eval/metrics.py`：不需要改（`summarize()` 已经能算 coverage / accuracy / abstain_accuracy）。
- `src/visioncite/eval/bootstrap.py`：对每个点的 accuracy 调 `bootstrap_mean_ci()`，写进报告。

**验收方式**

1. `python -m pytest -q tests/test_eval.py` 全绿；`test_summarize_counts_abstention` 是口径的守门人。
2. 跑 `python -m visioncite.eval.run --mode layout --per 2 --sweep 0.2,0.35,0.5,0.7`，输出一张表：每个阈值的 `coverage`、`AnswerAcc`、`abstain_accuracy`、`CI95`。
3. 结论必须回答：曲线是否单调？在哪个阈值出现"多弃答换来的准确率提升已经不值"的拐点？给出你选的阈值和理由。
4. 注意样本只有 36 条，任何两个点差 0.03 以内都不该当成差异。

**预计耗时**：35–40 分钟。

---

## 练习 11 给校验加"单位与日期一致性"规则

**对应**：第 10 讲。**目标**：现在的数字校验只比"数字本身"，答"1286.40 元"和"1286.40 美元"都会通过。补一条单位/日期校验，让这类错误被挡住。

**要改的文件**

- `src/visioncite/answer/verify.py`：新增 `units_in(text)`（抓 `元/美元/%/台/万次/日` 等紧跟数字的单位）与 `numbers_with_units_supported(answer, evidence_texts)`；在 `verify_answer()` 里加第 4 条检查，并把结果写进 `checks` 字典（键名用 `units_grounded`）。
- `src/visioncite/answer/abstain.py`：不需要改（它只消费 `verification.ok` 与 `reasons`）。
- `tests/test_answer.py`：新增"数字对但单位错"的失败用例，以及"单位相同但格式不同（`1286.40元` vs `1286.40 元`）"的通过用例——注意 `normalize` 会压缩空白，你的实现要对齐这个行为。
- 注意别破坏 `test_verify_answer_passes_for_grounded_answer`：证据里没有单位时，答案也不该带单位，否则会被新规则判失败。

**验收方式**

1. `python -m pytest -q tests/test_answer.py` 全绿（现有 14 条 + 新增）。
2. `python -m pytest -q tests/test_pipeline.py` 全绿：`test_answer_question_cites_region` 的答案含"1286.40"，若你的规则误伤它，会立刻暴露。
3. 评测 `--mode layout --per 2`：`coverage` 不应下降超过 0.05；如果明显下降，说明规则太严（例如把"万次"和"次"当成不同单位），需要先归一化单位再比。
4. 手工构造一条"数字幻觉"答案，确认 `verify_answer()` 返回的 `reasons` 里同时出现数字与单位两个原因（或明确只有一个，取决于短路顺序）。

**预计耗时**：35–40 分钟。

---

## 练习 12 新增一种消融模式，让报告自带一行

**对应**：第 12 讲。**目标**：把"某个组件到底有没有用"变成评测里的一等公民：新增 `layout_no_mmr`（有向量但跳过 MMR），报告自动多一行、多一份 json 记录。

**要改的文件**

- `src/visioncite/eval/run.py`：`MODES` 加 `"layout_no_mmr"`；`build_documents()` 里给它 `create_visual(settings)`（需要向量才能测 MMR），但把 `settings.mmr_lambda` 设为 1.0（等价于纯相关性，不做多样性）；或更干净的做法：在 `run_mode()` 前 `replace(settings, mmr_lambda=1.0)`。
- `src/visioncite/config.py`：不需要新增字段（复用 `mmr_lambda`）。
- `src/visioncite/eval/run.py` 的 `write_report()`：表格行是遍历 `results` 生成的，理论上不用改；检查"分类别"表头的类别列表（现在写死 `invoice/table/chart/slide`）是否需要跟随。
- `tests/test_eval.py`：新增一条"`MODES` 包含全部四种模式"的轻量断言（不要真的跑 CLIP）。

**验收方式**

1. `python -m pytest -q tests/test_eval.py` 全绿。
2. `python -m visioncite.eval.run --mode layout_no_mmr --per 2` 能单独跑通，报告里有该模式的一行；`--mode all` 会把 4 种模式都跑一遍（需要 CLIP 权重，CPU 上慢，注意时间）。
3. 对比 `hybrid` 与 `layout_no_mmr` 的 `Cite-P` 与 `cite_recall`：如果 `λ=1.0` 让引用更集中在同一区域，`Cite-P` 可能升而证据多样性降——这就是"MMR 的代价"，把它写进结论。
4. 检查 `reports/*.json` 里 `records` 的 `mode` 字段是否正确带上了新模式名。

**预计耗时**：30–40 分钟（跑 `all` 的时间另算）。

---

## 附：练习与讲次的对应

| 练习 | 讲 | 主要文件 | 难度 |
|---|---|---|---|
| 1 自适应阈值 | 2 | `layout/binarize.py`、`layout/regions.py` | 中 |
| 2 BM25 字段权重 | 7 | `index/bm25.py`、`pipeline.py` | 中 |
| 3 加权融合 A/B | 9 | `index/fuse.py`、`index/retrieve.py` | 中高 |
| 4 温度缩放校准 | 11 | `answer/extractive.py`、`config.py` | 中 |
| 5 增加 TIFF | ingest | `ingest/loader.py` | 低 |
| 6 表格切单元格 | 4 | `layout/table.py`、`layout/regions.py` | 高 |
| 7 切块重叠 | 6 | `index/chunking.py`、`config.py` | 中 |
| 8 dHash 去重 | ingest | `api/app.py`、`api/schemas.py` | 中 |
| 9 MMR 量纲修复 | 9 | `index/retrieve.py` | 中 |
| 10 阈值扫描 | 11、12 | `eval/run.py`、`eval/bootstrap.py` | 中高 |
| 11 单位校验 | 10 | `answer/verify.py` | 中 |
| 12 新消融模式 | 12 | `eval/run.py` | 低中 |

建议顺序：5 → 2 → 7 → 1 → 4 → 9 → 10 → 3 → 11 → 6 → 8 → 12。先做改动面小的（5、2、7）建立信心，再做需要跑三次评测的（1、10），最后做需要 CLIP 的（3、12）。
