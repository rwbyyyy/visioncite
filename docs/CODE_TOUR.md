# VisionCite 代码阅读路线

这是一份"带路"而不是"讲解"的文档：告诉你从哪进、读完本站应该能回答哪三个问题、以及这一段代码里真正值得停下来看的几个设计决定。公式与手算例子在 `docs/CURRICULUM.md`，这里不重复。

全篇 8 站，按依赖方向由下往上走。建议每站 30–60 分钟，读完在纸上写出三个问题的答案，再进下一站。总时长约 5–7 小时。

```text
第 1 站 config      → 第 2 站 types      → 第 3 站 ingest
      → 第 4 站 layout → 第 5 站 ocr      → 第 6 站 index
      → 第 7 站 answer → 第 8 站 api/eval
```

读之前先跑一次 `python -m pytest -q`。这个仓库的测试全部离线（`tests/conftest.py` 明确写了"不联网、不下载模型、不依赖本机 OCR"），跑通它你就有了"改动是否破坏契约"的安全网。

本文对照 2026-11-06 的仓库状态。函数名与文件名都核对过；若你读到的实现与描述有细节出入（项目当时仍在迭代），以源码和 `tests/` 为准。

---

## 第 1 站 config：参数只有一个来源

### 读哪些文件

- `src/visioncite/config.py`（全部）
- `src/visioncite/errors.py`（全部，很短）
- `src/visioncite/logging.py`（全部）

配套：`src/visioncite/__init__.py` 只导出 `Settings`、`VisionCiteError` 与 8 个数据类，是"公共 API 面很小"的示范。

### 读完应能回答

1. **哪些参数可以被环境变量覆盖，哪些不能？**
   `from_env()` 只接 `VISIONCITE_CACHE`、`VISIONCITE_RENDER_DPI`、`VISIONCITE_MAX_IMAGE_SIDE`、`VISIONCITE_MAX_UPLOAD_MB`、`VISIONCITE_ABSTAIN_THRESHOLD`、`VISIONCITE_OCR`、`VISIONCITE_VISUAL`、`VISIONCITE_CLIP_MODEL`、`VISIONCITE_HF_ENDPOINT`、`VISIONCITE_VLM_BASE_URL`、`VISIONCITE_VLM_MODEL`。**BM25 的 `k1`/`b`、`top_k`、`rrf_k`、`mmr_lambda` 没有环境变量入口**，只能在代码里构造 `Settings(bm25_k1=...)`（测试就是这么做的），或者调用 `Settings.from_env()` 后再 `dataclasses.replace`。想清楚"为什么不给它们开环境变量"（答案：它们不该在运行时被随意改，否则评测与线上两套数值）。

2. **`Settings` 为什么是 `frozen=True`？派生目录为什么用 `@property` 而不是字段？**
   冻结后不可能出现"运行到一半 cache_dir 被改"的竞态，也天然可哈希、可当缓存键。`model_dir`、`work_dir` 由 `cache_dir` 派生，避免"三个目录各写一次、迁移时漏一个"；`ensure_dirs()` 幂等创建。

3. **`/health` 会暴露什么、为什么不含密钥？**
   `Settings.describe()` 返回 `cache_dir`、`ocr_mode`、`visual_mode`、`clip_model`、`vlm_enabled`、`abstain_threshold`、`top_k`。`VISIONCITE_VLM_API_KEY` 不被 `Settings` 持有（它只在 `answer/vlm.py` 里从环境变量读），所以不可能从这个出口泄漏。

### 关键设计决定

- **单一配置源**：所有魔数（0.35、60、0.7、1.5、0.75）都在 `Settings` 里有一个名字、默认值、一句 docstring。面试里被问"这个 0.35 是什么"时，能翻到定义。
- **非法值安全回退**：`_env_int` / `_env_float` / `_env_str` 在解析失败时返回默认值，而不是让服务起不来。这是"配置错误不应打败可用性"的工程选择；代价是拼错的环境变量会被静默忽略（`_env_int` 不告警）。
- **能力开关而不是功能删除**：`OcrMode` / `VisualMode` 是 `Literal["auto", "off"]`，`vlm_enabled` 是派生属性。三个开关让"没有 OCR / 没有 CLIP / 没有 VLM"成为三种合法部署形态，而不是三种降级事故。
- **异常树按"谁能处理"分层**：`UnsupportedFileError` → HTTP 415，`IngestError` → HTTP 400，`VisualUnavailableError` → 内部降级，`AnswerBackendError` → 回退 extractive。每个子类带 `default_user_message`，路由层可以直接展示。

### 顺带确认的事实

`pyproject.toml` 里有两处声明与仓库实物不一致：`[project.scripts]` 声明了 `visioncite = "visioncite.cli:main"`，但源码里**没有 `cli.py`**；`readme = "README.md"` 指向的文件也不在仓库里。读到这一站时把 `pyproject.toml` 和目录树对一遍，是很好的"读项目先验契约"练习——**文档里的承诺要以实际存在的文件为准**。

---

## 第 2 站 types：全项目的数据契约

### 读哪些文件

- `src/visioncite/types.py`（全部）
- `tests/test_types.py`（作为契约的"可执行文档"）

### 读完应能回答

1. **`BBox` 为什么是左闭右开？代价是什么？**
   文档字符串给了理由：面积可以直接写 `(x1-x0)*(y1-y0)`，拼接相邻块不重复计像素。代价是"最后一个像素"不在框内，用 PIL/NumPy 裁剪时要小心是否要 `+1`；本项目统一用 NumPy 切片 `image[y0:y1, x0:x1]`，正好与左闭右开一致，所以不需要 `+1`。

2. **`frozen=True` 的 `BBox` 怎么在 `__post_init__` 里纠正反向坐标？**
   先把两端的值取成局部变量 `low`/`high`，再用 `object.__setattr__` 写回字段——冻结只挡普通赋值，挡不住这个后门。这样上游把 `(100, 80, 20, 10)` 传反了，也不会在 `iou()` 里算出负面积——`tests/test_types.py::test_bbox_normalizes_reversed_coordinates` 钉住了这个行为。

3. **`Region` 和 `Chunk` 为什么要分成两个类型？**
   `Region` 是"版面单元"（`page.regions`，可原地填充 `text`/`words`/`confidence`），`Chunk` 是"检索单元"（不可变，带 `chunk_id` 与 `doc_id`，一个 Region 可能被切成多个 Chunk）。分开之后，切块逻辑不必改 `Page` 的结构，引用又能通过 `Chunk.region_id` 指回原区域去裁图。

### 关键设计决定

- **冻结/可变分工**：`BBox`、`Word`、`Chunk`、`Citation`、`AnswerResult`、`DocumentMeta` 全部 `frozen=True`；`Region`、`Page` 是普通 dataclass，因为它们在 `build_regions()` 里被逐字段填充。这是"构建期可变、对外只读"的折中。
- **`Chunk` 同时带 `bbox` 和 `region_id`**：`bbox` 让引用能直接画框（不必回查 page），`region_id` 让裁图端点能按区域取图。冗余一点，但让下游少一次 join。
- **`AnswerResult` 把"弃答原因"做成一等公民**：`reasons: tuple[str, ...]` 不展示给终端用户，但评测与排查全靠它。`backend` 字段记录实际用的 `extractive`/`vlm`，消融时能按后端切片。
- **`confidence` 与 `score` 分开**：`Citation.score` 是融合检索分（0~1 归一化），`AnswerResult.confidence` 是作答置信度（由覆盖率和领先幅度算出）。两者来源不同，混用会让校准失真。
- **元组而不是列表**：`citations`、`reasons` 用 `tuple`，配合 `frozen=True` 让结果不可变、可哈希，避免调用方原地 `append` 后责任不清。

---

## 第 3 站 ingest：把文件变成页面图

### 读哪些文件

- `src/visioncite/ingest/loader.py`（全部）
- `src/visioncite/ingest/image.py`（全部）
- `src/visioncite/ingest/pdf.py`（全部）
- `tests/test_pipeline.py` 的 `test_detect_kind_by_magic_bytes`、`test_detect_kind_rejects_unknown`

### 读完应能回答

1. **为什么按魔数判断类型，而不是看扩展名？**
   扩展名可以随便改（`tests/test_pipeline.py::test_detect_kind_by_magic_bytes` 把 PNG 字节存成 `.pdf`，仍然识别为 `image`）。`_MAGIC_PREFIXES` 只列真正支持的格式，避免"能识别不能处理"的假承诺；`RIFF` 是 WebP/WAV 共用魔数，所以额外要求后缀在 `_SUFFIXES` 里。

2. **图片读取处理了哪三个坑？**
   EXIF 旋转（`ImageOps.exif_transpose`）、色彩模式（`convert("RGB")`，统一调色板/灰度/CMYK）、超大图（`_shrink_to_fit` 用 LANCZOS 按最长边 `max_image_side=2400` 等比缩小）。`crop_image()` 还会把越界 bbox 夹回图像内，完全在图外时返回 1x1 空白图而不是抛错。

3. **PDF 为什么以"渲染成图"为主，而不是直接抽文本？**
   多模态问答的证据是"页面上的一块区域"，而 `textpage.get_text_range()` 没有可靠坐标；统一走图片后，扫描件与电子版用同一条流水线，失败模式也一致。文本层只作降级来源（第 5 站 `_text_layer_region` 会用到）。`max_pages=40` 防止一个超大 PDF 拖垮服务。

### 关键设计决定

- **`detect_kind()` 是唯一分叉点**：新增格式只改 `_MAGIC_PREFIXES` 与 `_SUFFIXES` 两张表 + `load_pages()` 里一个分支。
- **`load_pages()` 不做 OCR、不做版面**：只产出 `Page`（`index`、`width`、`height`、`image`、`text_layer`），单一职责，测试可以用一张 3x3 的数组代替真实文件。
- **导入即依赖**：`loader.py` 顶层 `from visioncite.ingest.pdf import render_pages`，而 `pdf.py` 顶层 `import pypdfium2`。结果是"只处理图片也要装 pypdfium2"。这是速度与简单性的权衡——想做懒加载，就把 `render_pages` 的 import 挪进 `if kind == "pdf"` 分支（练习可做）。
- **错误包装**：PIL 解码失败 → `IngestError`；PDFium 抛出的异常类型不稳定，`_render_one()` 用 `except Exception` 统一包成 `IngestError`。异常信息里带页号（`page.get_index() + 1`），排障时能直接定位。
- **未被使用的导出**：`perceptual_hash()` / `hash_distance()`（dHash 判重）与 `count_pages()` 已实现但不在主链路里。读到时不要以为漏看调用点。

---

## 第 4 站 layout：从墨迹到带类型的区域

### 读哪些文件

- `src/visioncite/layout/binarize.py`（全部）
- `src/visioncite/layout/xycut.py`（全部）
- `src/visioncite/layout/table.py`（全部）
- `src/visioncite/layout/regions.py`（全部，本站重点）
- `tests/test_layout.py`（14 个用例，全部读完）

### 读完应能回答

1. **`build_regions()` 里的执行顺序是什么，为什么先找表格再切块？**
   `ink_mask(to_gray(image))` → `find_tables(mask)` → `xy_cut(mask, cut_config)` → `merge_overlapping()` → `_drop_tiny(min_area=900)` → `assign_words()` → 逐块 `classify()` / `build_text()` / `tighten_bbox()` → `reading_order()`。先找表格是因为表格候选要在分类时用到；先切块再合并是为了修 XY-cut 偶发的"一段被切成两块"。

2. **`region_id` 是怎么来的？引用为什么能回到页面上？**
   `region_id = f"p{page_index}-r{index}"`，`index` 是排序**前**的枚举序号（注意：`reading_order()` 在最后才重排列表，但 id 已经按原始顺序生成，所以 `r0` 不一定是最上面那块）。`Chunk.region_id` 保留这个 id，`api/app.py` 的 `/regions/{region_id}.crop` 靠遍历 `document.pages[*].regions` 找回 bbox 来裁图。

3. **`tighten_bbox()` 为什么存在？不收紧会怎样？**
   XY-cut 返回的是"整条内容带"，可能包含大段空白；直接当证据框会让前端高亮很空，评测 IoU 也偏低。收紧到词的外接矩形再 `pad(4)`，Cite-P 明显更好。副作用：无词的块（纯图）原样返回，所以图的框仍然是几何块。

### 关键设计决定

- **用几何方法而不是深度学习版面模型**：无参数、CPU 毫秒级、每一刀都能画出来解释；失效场景写在模块注释里（多栏无留白、背景噪点、倾斜扫描），不藏。
- **常量都带名字和理由**：`_MIN_INK = 0.002`（低于它的叶子当噪点）、`CutConfig(min_gap=12, min_size=24, max_depth=6)`（144dpi 下的经验值）、`_drop_tiny(min_area=900)`、`merge_overlapping(iou_threshold=0.3)`、`classify` 的 `> 0.5` 与 `> 0.05`。调参时改一处、解释得清。
- **分类只有三条规则**：表格覆盖过半 → table；无词但有墨 → figure；其余 text。不做页眉页脚/公式/多栏的过度工程，交给文档里的 LIMITATIONS 说明。
- **`_true_runs` 与 `_binary_runs` 各写一份**：`xycut.py` 和 `table.py` 各有一个"找连续 True"的函数，注释明说"保持模块独立"。读的时候不要以为是重复代码忘了重构——这是刻意的零耦合选择（代价是两处维护）。
- **`reading_order` 用 40px 分带的启发式**：`band[y0 // 40]` 再带内按 `x0` 排。简单、可解释，但对跨带的大图会排错位置。知道它"会错在哪"比知道它"怎么实现"更重要。

---

## 第 5 站 ocr：把会失败的外部能力变成插件

### 读哪些文件

- `src/visioncite/ocr/base.py`（全部）
- `src/visioncite/ocr/factory.py`（全部）
- `src/visioncite/ocr/rapidocr_backend.py`（全部）
- `src/visioncite/ocr/null_backend.py`（全部）
- `tests/conftest.py` 的 `FakeOcr`
- `src/visioncite/pipeline.py` 的 `_analyze_page()` 与 `_text_layer_region()`

### 读完应能回答

1. **接口为什么只有三个成员？**
   `name`、`available()`、`recognize(image) -> list[Word]`。再多一个方法，换后端（PaddleOCR、云服务、自研）的成本就上去了；`name` 是为了 `/health` 与 `Document.ocr_name` 能报告"实际用了谁"，这是可观测性的最小要求。

2. **降级路径一共几条？**
   三条：① `ocr_mode="off"` 直接返回 `NullOcrBackend`；② `rapidocr_onnxruntime` 未安装，`available()` 为 False，工厂 warning 后返回 `NullOcrBackend`；③ 引擎加载失败或单页推理抛异常，`_get_engine()` 置 `_load_failed`、`recognize()` 返回 `[]`（只记日志，不打断链路）。另有一条页级降级：无词但有 `page.text_layer` → 整页一个文本区域。

3. **置信度在哪一层被过滤？上层还会再过滤吗？**
   在 backend 内部：`_to_words()` 最后一步 `filter_confidence(words, self._min_confidence)`，默认 0.5。上层（`pipeline`、`layout`）不再过滤，只做加权平均（`regions._mean_confidence`）。如果以后要按区域动态调阈值，改的是 `RapidOcrBackend(min_confidence=...)` 的构造参数，不是上层。

### 关键设计决定

- **`Protocol` + `@runtime_checkable`**：结构化子类型，换后端不需要继承本项目；`tests/conftest.py` 的 `FakeOcr` 可以选择继承（它继承了）也可以不继承，都能通过。
- **懒加载 + 记忆化**：`_engine`、`_load_failed` 两个字段，避免"每次调用都尝试 import 并失败"的开销，也避免冷启动时阻塞。
- **失败返回空列表，不抛异常**：OCR 是可选能力，抛异常会迫使每个调用点写 `try/except`；返回 `[]` 让上层自然走"无文字"分支。
- **`OcrUnavailableError` 定义了但没抛**：它是给"调用方显式要求 OCR，而环境确实不可用"那类场景准备的契约。当前实现选择"静默降级 + 日志 + 后端名上报"，你要能说出这两种策略各自的适用场景。
- **`polygon_to_bbox` 放在 `base.py` 而不是 backend 里**：几何归一化是跨后端共用的，放在契约层可以复用、可以单测。

---

## 第 6 站 index：分词、切块、BM25、两路召回

### 读哪些文件

- `src/visioncite/index/tokenize.py`（全部）
- `src/visioncite/index/chunking.py`（全部）
- `src/visioncite/index/bm25.py`（全部）
- `src/visioncite/index/fuse.py`（全部）
- `src/visioncite/index/retrieve.py`（全部）
- `src/visioncite/index/visual.py`（`ClipVisualEncoder.load/encode_images/encode_texts`、`NullVisualEncoder`、`_l2_normalize`、`create_visual`）
- `tests/test_index.py`（全部）

这一站文件最多，但主链路只有一条：**Region 文本 → tokenize → BM25/向量 → RRF → MMR → 归一化分数**。

### 读完应能回答

1. **一段文本从 Region 到 BM25 的 token 流是什么？**
   `region.text` →（`chunking.split_sentences` + `_pack(max_chars=240)`）→ 若干 `Chunk` → `BM25Index.fit([(chunk_id, text)])` → 每个 chunk `tokenize()` → `Counter` 词频 → `postings[term][doc_index] = tf`。检索时 `tokenize(query)` → `set(terms)` 去重 → 对每个词的 posting 累加 `idf * tf_component`。

2. **两路召回在哪里合并？合并前后分数分别是什么量纲？**
   合并点是 `retrieve()` 里的 `reciprocal_rank_fusion(rankings, k=settings.rrf_k)`：先 `bm25.search(question, top_k*3)` 拿文本排名，再 `_visual_search()` 拿视觉排名（余弦相似度），两路只保留 `chunk_id` 顺序，丢掉原始分数。融合后才是 `min(1.0, raw/top)` 的 0~1 标度。

3. **MMR 的输入量纲对结果有什么影响？**
   `_diversify()` 传进去的 `relevance` 是 RRF 原始分（约 0.03 量级），而冗余度项是 `[0,1]` 的余弦；`λ=0.7` 时多样性项的量级反而更大，MMR 实际更"偏好多样"而不是"偏好相关"。这不是隐藏 bug，而是当前实现的真实行为，练习 9 会让你修它并做 A/B。

### 关键设计决定

- **bigram 替代分词器**：无词典、抗未登录词、可解释；代价是索引规模与"没有停用词表"。选择理由写在 `tokenize.py` 模块注释里，读一遍能直接用来回答面试题。
- **倒排是最朴素的结构**：`postings: dict[str, dict[int, int]]` + `doc_lengths: list[int]` + `chunk_ids: list[str]`。没有跳表、没有压缩、没有 SIMD；换来的是每一步都能单测、能在 20 行内讲清。
- **BM25 只依赖 `tokenize`**，不知道 `Chunk`、`Page`、`Document`；`fit()` 收的是 `(id, text)` 二元组。这让"给别的语料建索引"不需要构造整个 Document。
- **RRF 零参数、MMR 可退化**：`weights=None` 时等权；`vectors is None` 时 `_diversify` 走截断分支。每个降级都有明确出口，异常也有 `try/except` + `logger.exception` 兜底（`mmr_select` 失败就退回 RRF 顺序）。
- **`retrieve()` 只依赖 `Document` 的 4 个字段**（`chunks`、`bm25`、`vectors`、`encoder`）+ `chunk_by_id()`。这是它能被 `eval/run.py` 在 `page` 模式下"改完 chunks 后重跑"的前提（`_flatten_to_page()` 会重设 `chunks`、`bm25`、`vectors`）。
- **唯一的反向 import**：`from visioncite.pipeline import Document` 只为类型注解。它没有造成循环，但把 `pypdfium2` 拖进了任何 `import visioncite.answer` 的进程。

---

## 第 7 站 answer：检索之外的判断题

### 读哪些文件

- `src/visioncite/answer/qa.py`（全部，先读这个，它是用例层的地图）
- `src/visioncite/answer/extractive.py`（全部）
- `src/visioncite/answer/verify.py`（全部）
- `src/visioncite/answer/abstain.py`（全部）
- `src/visioncite/answer/vlm.py`（`SYSTEM_PROMPT`、`VlmAnswerer.answer/_build_payload/_parse`）
- `tests/test_answer.py`（全部）

### 读完应能回答

1. **一次 `answer_question()` 的完整决策树是什么？**
   `question.strip()` 为空 → 弃答；`retrieve()` 为空 → 弃答；`_compose()` 走 VLM（`settings.vlm_enabled` 且 `backend in {"auto","vlm"}`，失败/无有效引用则回退 extractive）或 extractive；`_build_citations()` 把块转引用；`verify_answer()` 校验；`should_abstain()` 三档判断；弃答时把 `answer` 置空。所有路径都返回 `AnswerResult`，不抛业务异常。

2. **置信度由哪些信号构成，分别管什么？**
   `0.55*coverage + 0.25*margin + 0.20*min(1, score)`：覆盖率管"答在点上"，margin 管"有没有歧义"（第一与第二候选的相对差距），句子分只作弱信号。VLM 路径直接用模型自报的 `confidence`（被 `_parse` 截断到 0~1）。

3. **VLM 失败时会发生什么，用户能看到什么？**
   `_compose()` 捕获所有异常并 `logger.exception`，回退 `compose_answer()`；`_map_citations()` 只保留"白名单内"的 id，模型编的 id 被丢弃、不报错；如果 VLM 返回空答案或全部引用无效，同样回退。用户只会看到 `backend="extractive"` 的结果，不会看到 500。

### 关键设计决定

- **用例层只依赖"接口 + 数据"**：`qa.py` 依赖 `retrieve`、`compose_answer`、`verify_answer`、`should_abstain` 和 `Document`，不 import `layout`/`ocr`/`bm25`。所以换检索器、换答案器都不影响编排。
- **校验用返回值，弃答用返回值**：`Verification(ok, reasons, checks)` 与 `should_abstain()` 的 `(bool, reasons)` 都是数据而不是异常；`errors.py` 里的 `VerificationError` 目前无人抛出。
- **VLM 的结构约束先于提示词**：`SYSTEM_PROMPT` 规定只输出 JSON、数字必须逐字出现在证据中、证据不足就置空；`_build_payload()` 用 `temperature=0` 与 `response_format={"type": "json_object"}`；`_parse()` 用 `_JSON_BLOCK` 正则兜底，字段类型不对就归零。**模型越自由，校验层越重要**，这是整个设计的主线。
- **引用白名单 `_map_citations()`**：模型只能引用传进去的 `chunk_id`，这是"引用不可伪造"的第一道闸；第二道是 `citation_ids_exist()`。
- **`snippet=chunk.text[:160]`**：引用摘要固定截断，避免响应体被长表格撑爆；完整文本仍可通过 `page`/`region_id` 取图。
- **外部依赖一律可降级**：`httpx` 超时、非 JSON、字段缺失 → `AnswerBackendError` → 上层回退。`vlm.py` 的模块注释也明说"本文件没有在真实厂商端点上验证过"，不夸大。

---

## 第 8 站 api / eval：出口与度量

### 读哪些文件

- `src/visioncite/api/schemas.py`（全部）
- `src/visioncite/api/app.py`（全部）
- `src/visioncite/eval/run.py`（全部）
- `src/visioncite/eval/metrics.py`（全部）
- `src/visioncite/eval/bootstrap.py`（全部）
- `src/visioncite/eval/dataset.py`、`src/visioncite/eval/corpus_gen.py`、`src/visioncite/eval/templates.py`（按需）
- `tests/test_api.py`、`tests/test_eval.py`

### 读完应能回答

1. **HTTP 层怎么把异常翻译成状态码？**
   上传超过 `max_upload_mb` → 413；`detect_kind()` 抛 `UnsupportedFileError` → 415；其他 `VisionCiteError` → 400，`detail` 用 `exc.user_message`；未上传就提问 / 文档不存在 → 404。业务逻辑全在 `pipeline`/`answer`，路由函数只做"校验输入 → 调用用例 → 翻译异常"。

2. **文档存在进程内字典里，有什么后果？**
   `app.state.documents` 是 `dict[str, Document]`，`doc_id` 由 `document_id_from_bytes()` 内容寻址，重复上传直接返回已有文档（`PdfiumError` 之类的解析失败也就不会重复发生）。后果：重启丢文档、多副本不共享、内存随文档数线性增长；上生产要换 Redis/对象存储（模块注释里写了）。另外 `Document` 里存着所有页面图（RGB 数组），一个 40 页 PDF 的内存占用要按 `页数 × 高 × 宽 × 3` 估算。

3. **三种消融模式差在哪，报告口径是什么？**
   `page` 模式在 `build_documents()` 之后调 `_flatten_to_page()`，把每页压成"整页一个区域"并重建 `chunks`/`bm25`/`vectors=None`；`layout` 用完整版面 + BM25、视觉用 `NullVisualEncoder()`；`hybrid` 额外用 `create_visual(settings)` 并保留 `vectors`。报告同时写 `.md`（人看）与 `.json`（逐样本记录 + 汇总），accuracy 只统计"可答且已作答"样本，coverage 与 abstain_accuracy 单列。

### 关键设计决定

- **`create_app(settings=None)` 工厂**：测试注入 `Settings(max_upload_mb=0)` 就能验 413（`tests/test_api.py::test_upload_rejects_too_large`），不必改环境变量。
- **路由按用途拆成四个注册函数**：`_register_health_routes` / `_register_document_routes` / `_register_asset_routes` / `_register_ui_routes`，每个函数只注册一类路由。好处是单个函数的圈复杂度不超标（`pyproject.toml` 里 `max-complexity = 10`），读起来也按"健康检查 / 文档与问答 / 资源 / 页面"分块。真正的上传逻辑抽到 `_store_document(app, data, filename)`，`upload()` 只剩"读字节 + 大小校验 + 转调"三行。
- **去重发生在 `_store_document()` 里**：先算 `doc_id = document_id_from_bytes(data)`，若已在 `app.state.documents` 就直接返回响应，连落盘都省了。
- **Pydantic 模型独立成 `schemas.py`**：自动生成 OpenAPI、字段改名立刻在类型检查暴露、前端照着写。`AskRequest.question` 有 `min_length=1, max_length=500`，`top_k` 有 `ge=1, le=20`（注意：这个 `top_k` 目前只做校验，`ask()` 并没有把它传给 `answer_question`，实际用的是 `settings.top_k`）。
- **页图与裁图端点**：`/api/documents/{doc_id}/pages/{page}.png` 与 `/regions/{region_id}.crop` 让前端能"页面叠加高亮框"，这是"可视化引用"的最后一公里；裁图统一 `pad(8, (width, height))` 留边，避免框贴边。
- **评测是可复现的实验装置而不是一次性脚本**：`load_or_generate()` 复用已生成语料，`write_report()` 落 md+json 双份，`EvalRecord` 逐样本保存（`item_id`、`category`、`answerable`、`abstained`、`contains`、`em`、`f1`、`cite_hit`、`cite_precision`）。任何一个汇总数字都能下钻到样本。
- **置信区间是默认输出**：`run_mode()` 里 `bootstrap_mean_ci(accuracy_values)` 一定会写进 `accuracy_ci95`，报告表格里单列一列。这是"不要只报点估计"的工程化。
- **合成语料用绘制坐标当真值**：`templates.py` 的 `draw_line()` 用 `draw.textbbox` 返回文本框，直接写进 `QAItem.citation`；`find_cjk_font()` 找不到字体就抛错，不生成豆腐块。代价与收益都写在报告"说明"里。

---

## 读完之后的三个自检动作

1. 不看代码，画出 `build_document()` → `answer_question()` 的完整调用链，标注每步的输入输出类型。
2. 挑一条 `reports/` 里的逐样本记录，用 `eval/metrics.py` 的函数手工复算它的 `contains`、`f1`、`cite_hit`，看能不能对上。
3. 把 `docs/EXERCISES.md` 里任意一个练习做掉，并用 `python -m pytest -q` 与一次 `python -m visioncite.eval.run --mode layout --per 2` 验证"没有把别的东西弄坏"。
