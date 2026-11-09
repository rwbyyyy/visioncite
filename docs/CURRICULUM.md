# VisionCite 十二讲：从像素到可验证引用

这份材料面向"能写 Python、但没做过文档 AI"的读者。目标不是让你背下 20 个名词，而是让你能沿着真实代码把一条链路走通：

```
文件 → 页面图 → 墨迹 → 版面区域 → OCR 词 → 检索块 → 两路召回 → 融合/去冗余
     → 抽取答案 → 校验 → 弃答 → 引用框 → 评测报告
```

每一讲固定五段：要解决的问题、核心概念与公式、对应代码、一个手算小例子、三道自测题。文中的函数名、常量名、文件路径全部来自本仓库源码；数值例子都可以用仓库里的函数复现（第四段会说明怎么复现）。

---

## 使用说明与运行前提

**代码位置约定**：所有路径相对于项目根目录 `visioncite/`，例如 `src/visioncite/layout/binarize.py`。

**安装**：

```bash
uv sync --extra dev
# 或
python -m pip install -e ".[dev]"
```

`dev` 额外依赖包含 `pytest`、`httpx`（`fastapi.testclient` 需要它）、`ruff`。`pypdfium2` 是核心依赖而不是可选项：`src/visioncite/ingest/loader.py` 在模块顶层 `from visioncite.ingest.pdf import render_pages`，所以即使你只处理 PNG，导入 `visioncite.pipeline` 也需要它。

**常用命令**：

```bash
python -m pytest -q                          # 全部离线测试（不下载模型）
python -m pytest -q tests/test_layout.py     # 只跑版面
python -m visioncite.eval.run --mode layout --per 2     # 一种消融模式
python -m visioncite.eval.run --mode all --per 2        # 三种模式，需 CLIP
ruff check src tests                         # 静态检查（行宽 100，见 pyproject）
```

**关于 OCR 与 CLIP**：默认配置 `ocr_mode="auto"`、`visual_mode="auto"`，但两者都是"尝试加载，失败即降级"：

- 没装 `rapidocr-onnxruntime` → `create_ocr()` 返回 `NullOcrBackend`，OCR 结果为空；
- 没装 `torch`/`transformers` → `create_visual()` 返回 `NullVisualEncoder`，视觉检索关闭。

因此评测在没有 GPU、没有网络的机器上也能跑通，但 `page`/`layout` 模式下文字来自哪里？看第五讲：`pipeline._analyze_page()` 在"OCR 无词但 PDF 有文本层"时会退化成整页一个文本区域；而合成语料是 PNG（没有文本层），没有 OCR 时文本区域会是空的。这也是为什么 `tests/` 用 `FakeOcr` 注入词而不是真的跑 OCR。

**每讲第四段怎么复现**：进入项目根目录，设置 `PYTHONPATH=src`（Windows PowerShell 用 `$env:PYTHONPATH="src"`），然后用 `python -c "..."` 调用文中的函数。例子里的数字都是这样算出来的。

**代码快照**：本材料对照的是 2026-11-06 的仓库状态。项目当时仍在迭代，如果你读到的函数行为与本文有出入，以源码与 `tests/` 为准；本文所有"实测"数字都可以用文中给的 `python -c` 命令重新算一遍。

---

## 课程地图

| 讲 | 主题 | 主要文件 | 前置 |
|---|---|---|---|
| 1 | 项目分层与依赖方向 | `pipeline.py`、`types.py`、`errors.py`、`logging.py` | 无 |
| 2 | 灰度与 Otsu 二值化 | `layout/binarize.py` | 1 |
| 3 | 投影与递归 XY-cut | `layout/xycut.py` | 2 |
| 4 | 表格线检测与区域分类 | `layout/table.py`、`layout/regions.py` | 3 |
| 5 | OCR 适配器与降级 | `ocr/*`、`pipeline.py` | 1 |
| 6 | 中文分词与字 bigram | `index/tokenize.py`、`index/chunking.py` | 1 |
| 7 | BM25 | `index/bm25.py` | 6 |
| 8 | CLIP 与对比学习 | `index/visual.py` | 6 |
| 9 | RRF 与 MMR | `index/fuse.py`、`index/retrieve.py` | 7、8 |
| 10 | 引用校验 | `answer/verify.py`、`types.py` | 6 |
| 11 | 弃答与 ECE | `answer/extractive.py`、`answer/abstain.py` | 10 |
| 12 | 评测方法与 bootstrap | `eval/*` | 全部 |

依赖关系是单向的：能力层（`ingest`/`layout`/`ocr`/`index`）不 import `answer`/`api`；`types.py` 不 import 任何内部模块。唯一的"反向边"是 `src/visioncite/index/retrieve.py` 里的 `from visioncite.pipeline import Document`，第一讲会解释它为什么没有形成真正的循环。

---

## 第 1 讲 项目分层与依赖方向

### ① 这一讲解决什么问题

打开仓库会看到 8 个子包，`pipeline.py` 又几乎 import 了所有模块，很容易误以为"到处都是耦合"。这一讲给出这个项目的分层规则、依赖方向，以及"哪些模块可以随便换、哪些不能"。读完你应该能判断：要加一个新能力，代码应该落在哪一层。

### ② 核心概念与公式

分层（自下而上，箭头表示"可以被 import"）：

```text
契约层   types.py  errors.py  logging.py  config.py     ← 谁都可以用，它不用任何人
能力层   ingest/  layout/  ocr/  index/                 ← 只依赖契约层（index.retrieve 例外）
编排层   pipeline.py                                    ← 唯一知道"步骤顺序"的地方
用例层   answer/                                        ← 检索→作答→校验→弃答
入口层   api/  eval/                                    ← 只做输入输出与评测编排
```

四条被代码强制的规则：

1. **单一配置源**：所有可调参数集中在 `Settings`（frozen dataclass），默认值只写一次。
2. **依赖倒置**：`OcrBackend` 是 `typing.Protocol`（`@runtime_checkable`），上层只认 `available()` 与 `recognize()` 两个方法，不认具体类。
3. **降级是显式能力开关**：`Settings.ocr_mode` / `Settings.visual_mode` 取 `"auto"` 或 `"off"`；`Settings.vlm_enabled` 由 `vlm_base_url` 与 `vlm_model` 是否同时非空决定。
4. **错误分两类**：可降级错误（`OcrUnavailableError`、`VisualUnavailableError`）与不可降级错误（`IngestError`、`UnsupportedFileError`），后者带 `user_message`，可以直接展示给用户。

### ③ 对应代码

- `src/visioncite/types.py` — `BBox`、`Word`、`Region`、`Page`、`Chunk`、`Citation`、`AnswerResult`、`DocumentMeta`。全项目共享的数据契约。
- `src/visioncite/config.py` — `Settings`、`Settings.from_env()`、`Settings.ensure_dirs()`、`Settings.describe()`、属性 `model_dir` / `work_dir` / `vlm_enabled`。
- `src/visioncite/errors.py` — `VisionCiteError` 及 6 个子类。
- `src/visioncite/logging.py` — `setup_logging()`、`get_logger(name)`。
- `src/visioncite/pipeline.py` — `build_document()`、`Document`、`_analyze_page()`、`_encode_chunks()`、`document_id_from_bytes()`、`_fingerprint()`、`make_word()`。
- `src/visioncite/ocr/base.py` — `OcrBackend`（Protocol）。
- `src/visioncite/index/visual.py` — `create_visual()`、`ClipVisualEncoder`、`NullVisualEncoder`。

关于那条"反向边"：`index/retrieve.py` 只为了类型注解写上 `from visioncite.pipeline import Document`。它不构成运行时循环，因为 `pipeline.py` 只 import `index.bm25`、`index.chunking`、`index.visual`，而 `index/__init__.py` 不 import `retrieve`。代价是：**任何人只要 `import visioncite.answer`，就会连带把 `pypdfium2` 拉进内存**（`answer.qa` → `index.retrieve` → `pipeline` → `ingest.loader` → `ingest.pdf`）。这就是第 3 站要讲的"导入即依赖"。

### ④ 一个手算小例子

拿一页图，数一遍 `build_document()` 的调用次数与对象数量。

假设：1 张 PNG 宽 640 高 400；`FakeOcr` 返回 6 个词；XY-cut 得到 2 个区域；区域 A 的文本 300 字、区域 B 的文本 120 字；`chunks_from_pages` 的 `max_chars=240`。

```text
load_pages(path, settings)            → 1 次，返回 1 个 Page
create_ocr(settings)                  → 1 次（ocr 参数为 None 时才创建）
create_visual(settings)               → 1 次
_analyze_page(page, backend, settings) → 1 次/页 = 1 次
  └ ocr.recognize(page.image)          → 1 次，返回 6 个 Word
  └ build_regions(...)                 → 1 次，产出 2 个 Region
chunks_from_pages(doc_id, pages)      → 1 次 → 300 字按 240 打包成 2 块，120 字成 1 块，共 3 个 Chunk
BM25Index.fit([(chunk_id, text)…])    → 1 次，输入 3 条
_encode_chunks(pages, chunks, enc)    → 1 次
  └ crop_image(...)                    → 2 次，不是 3 次：按 (page, region_id) 去重
  └ encoder.encode_images(crops)       → 1 次，输出 2 行向量
  └ np.stack([vectors[i] for i in chunk_to_vector]) → 3 行（区域 A 的两块共用同一向量）
```

关键数字：**3 个 chunk、2 张裁图、2 行向量、3 行 chunk 向量**。如果 `crop_image` 被调用 3 次，说明去重键写错了（`_encode_chunks` 里是 `key = (chunk.page, chunk.region_id)`）。

### ⑤ 自测 3 问

1. **为什么 `build_document()` 放在 `pipeline.py` 而不是 `api/app.py`？**
   要点：编排涉及 6 个模块和一个固定顺序；写进路由会让 HTTP 层与算法耦合，`eval/run.py` 与命令行都无法复用；`pipeline` 允许注入 `ocr=`/`visual=`，测试不必跑真模型（`tests/test_pipeline.py` 的 `document` fixture 就是这么做的）。

2. **`OcrBackend` 为什么用 `Protocol` 而不是抽象基类？**
   要点：结构化子类型——任何类只要实现 `name`、`available()`、`recognize()` 就满足接口，不必继承本项目；`@runtime_checkable` 让 `isinstance(backend, OcrBackend)` 可用；替换成 PaddleOCR 时不需要改上层。

3. **要做"同一张页面图 OCR 结果缓存"，应该改哪一层？**
   要点：编排层（`pipeline.py`）或 `ocr/` 里加装饰器；`layout` 只消费 `words`，不应该知道缓存、IO、哈希。判断标准是"这段代码是否知道步骤顺序或外部资源"。

---

## 第 2 讲 灰度与 Otsu 二值化

### ① 这一讲解决什么问题

版面分析的第一刀不是切分，而是决定"哪些像素算墨"。彩色页要先降成灰度，再用一个阈值分成前景/背景。阈值选得不好，浅色文字被当成背景（漏检），或者纸张纹理被当成墨（整页糊成一团），后面所有几何计算都会错。Otsu 给了"无参数、可解释、对扫描件稳定"的默认答案。

### ② 核心概念与公式

灰度化用 ITU-R BT.601 亮度权重（人眼对绿最敏感）：

```text
Y = 0.299*R + 0.587*G + 0.114*B
```

Otsu：在所有 256 个候选阈值 `t` 上，把像素分成 `<=t`（背景）与 `>t`（前景）两类，取"类间方差"最大的那个：

```text
w0(t) = Σ_{i<=t} p(i)          w1(t) = 1 - w0(t)
mu0(t) = Σ_{i<=t} i*p(i) / w0   mu1(t) = (mu_T - mu0*w0) / w1
sigma_b^2(t) = w0 * w1 * (mu0 - mu1)^2
t* = argmax_t sigma_b^2(t)
```

实现里用的是等价的累积和形式（避免二重循环、避免除零）：

```text
sigma_b^2(t) = (mu_T * w0(t) - mu0_unnorm(t))^2 / (w0(t) * (1 - w0(t)))
其中 mu0_unnorm(t) = Σ_{i<=t} i*p(i)，mu_T = mu0_unnorm(255)
```

三个必须记住的实现细节：

- 阈值遍历 `0..255`，`np.argmax` 返回**最小**的极大值下标，所以当方差在 `t=100..254` 上并列时返回 100；
- 前景判定用 `<=`：`ink = gray <= t`。这不是笔误——纯黑白扫描件的最优阈值常常就是 0，用 `<` 会把所有纯黑像素漏掉（mask 直接变空）；
- 全空图（直方图和为 0，例如 0x0 数组）返回 `128`。

区域墨迹占比：

```text
ink_density(mask, bbox) = mean(mask[y0:y1, x0:x1])
```

### ③ 对应代码

- `src/visioncite/layout/binarize.py`
  - `to_gray(image)` — 已经是二维就原样返回；否则 float32 加权求和，`np.clip(0,255).astype(np.uint8)`（截断而非四舍五入）。
  - `otsu_threshold(gray)` — `np.bincount(gray.ravel(), minlength=256)` 建直方图，`np.cumsum` 求 `omega` 与 `mu`，`np.errstate(divide="ignore", invalid="ignore")` 下算方差，`np.argmax` 取阈值。
  - `ink_mask(gray, threshold=None)` — `threshold=None` 时调用 Otsu，返回 `gray <= cut`。
  - `ink_density(mask, bbox)` — 先把 bbox 夹到图像内，再 `patch.mean()`。
- 调用方：`src/visioncite/layout/regions.py` 的 `build_regions()` 第一行就是 `mask = ink_mask(to_gray(image))`；`xycut.xy_cut()` 与 `table.find_tables()` 都只吃这个 bool 矩阵。

### ④ 一个手算小例子

设一张 2x3 的灰度图，值分别是 `[0, 0, 100; 0, 100, 255]`，即 3 个 0、2 个 100、1 个 255，总像素 `N=6`，概率 `p(0)=0.5, p(100)=1/3, p(255)=1/6`，`mu_T = 0*0.5 + 100/3 + 255/6 = 33.3333 + 42.5 = 75.8333`。

候选 `t=0`（背景 = 3 个 0）：

```text
w0 = 0.5,  mu0_unnorm = 0
sigma_b^2 = (75.8333*0.5 - 0)^2 / (0.5*0.5) = 1437.67 / 0.25 = 5750.69
```

候选 `t=100`（背景 = 3 个 0 + 2 个 100）：

```text
w0 = 0.83333,  mu0_unnorm = 100/3 = 33.3333
sigma_b^2 = (75.8333*0.83333 - 33.3333)^2 / (0.83333*0.16667)
          = (63.1944 - 33.3333)^2 / 0.13889
          = 891.69 / 0.13889 = 6420.14
```

6420.14 > 5750.69，所以 `t* = 100`（`t=100..254` 方差相同，取到 100）。于是 `ink_mask` 得到 `[[T,T,T],[T,T,F]]`：3 个 0 与 2 个 100 被判为墨迹（`<=100`），只有 1 个 255 是背景。注意最后那个 `255` 之所以是背景，是因为 `255 <= 100` 为假，而不是因为"严格小于"。

顺带算一遍灰度化：`RGB=(200,100,50)` → `0.299*200 + 0.587*100 + 0.114*50 = 59.8 + 58.7 + 5.7 = 124.2` → `uint8` 截断为 **124**。

复现：

```bash
python -c "import numpy as np; from visioncite.layout.binarize import otsu_threshold, ink_mask; g=np.array([[0,0,100],[0,100,255]],dtype=np.uint8); print(otsu_threshold(g), ink_mask(g).astype(int))"
# 100 [[1 1 1] [1 1 0]]
```

### ⑤ 自测 3 问

1. **`ink_mask` 为什么用 `gray <= cut` 而不是 `<`？这个选择会带来什么副作用？**
   要点：纯黑白扫描件（只有 0 与 255）的类间方差在最优点上并列，`argmax` 返回 0，用 `<` 会让"所有黑像素"全部漏掉，mask 变成空的；用 `<=` 才能把纯黑识别成墨。副作用是"恰好等于阈值的像素一律算墨"，在灰度渐变（照片、阴影）上会把一部分中间调算进来——所以它更适合扫描件而不是自然图像。

2. **拍照阴影下 Otsu 为什么会失效？正确做法是什么？**
   要点：全局阈值要同时满足亮区和暗区，阴影中的纸面灰度可能低于亮区文字，任何单一 t 都会二选一地犯错；应改用局部自适应阈值（Sauvola、或积分图算局部均值/标准差），见 `docs/EXERCISES.md` 练习 1。

3. **空白页与全黑页分别得到什么？**
   要点：两者都是"单峰直方图"，所有候选阈值的类间方差都因分母为 0 被置 0，`argmax` 返回 0。空白页（全 255）`255 <= 0` 全 False，mask 为空，`xy_cut` 返回空列表；全黑页（全 0）`0 <= 0` 全 True，整页一个墨迹块，`_MIN_INK` 会保留它，最终整页一个 Region。

---

## 第 3 讲 投影与递归 XY-cut

### ① 这一讲解决什么问题

把整页像素变成"若干内容块"，每个块对应视觉上的一段文字、一张图或一个表格。这一步决定了证据框的粒度：块太粗，引用高亮一大片空白；块太碎，一句话被拆成三块，检索分数全被稀释。XY-cut 用"投影找空白带"这个最朴素的信号完成切分，无参数、毫秒级、每一刀都能画给你看。

### ② 核心概念与公式

```text
水平投影（按行）：profile[y] = Σ_x mask[y, x]        axis=0
垂直投影（按列）：profile[x] = Σ_y mask[y, x]        axis=1
内容段 = profile > 0 的连续区间
若 下一段start - 上一段end < min_gap，则合并（行距 < 阈值视为同一块）
```

递归规则（`_cut`）：

1. 深度到顶、或块边长小于 `min_size`、或面积过小 → 记为叶子；
2. 当前方向投影能切出 >1 段 → 对每段递归，**换另一个方向**；
3. 当前方向切不动 → 换另一个方向再试一次（多栏文档常见：横切失败、竖切有效）；
4. 两个方向都切不动 → 叶子；
5. 叶子还要过一道墨迹门槛（`_MIN_INK = 0.002`），否则丢弃。

复杂度：每层递归对区域像素做一次投影，最深 `D` 层，总计 `O(D × 页面像素)`。`CutConfig` 默认 `min_gap=12, min_size=24, max_depth=6`。

`_true_runs()` 的小技巧：布尔序列首尾各补一个 `False`，做差分，`+1` 处是起点、`-1` 处是终点（左闭右开），一次 `np.diff` 代替逐元素循环。

### ③ 对应代码

- `src/visioncite/layout/xycut.py`
  - `CutConfig` — `min_gap=12`、`min_size=24`、`max_depth=6`。
  - `xy_cut(mask, config=None)` — 入口，从 `BBox(0, 0, width, height)` 开始，`axis=0`。
  - `_cut(...)` — 递归主体，包含"换轴重试"。
  - `_split_segments(piece, axis, min_gap)` — 投影 + 合并。
  - `_true_runs(flags)` — 差分找连续 True。
  - `_child_box(parent, start, end, axis)` — 相对坐标换算成子区域。
  - `_append_leaf(mask, box, out)` — 面积与墨迹占比过滤。
- 调用方：`src/visioncite/layout/regions.py` 的 `build_regions()`：`boxes = xy_cut(mask, cut_config)`，随后 `merge_overlapping()`、`_drop_tiny()`（面积 < 900 丢弃）。

### ④ 一个手算小例子

取一张 30 行 x 20 列的墨迹矩阵，只有两段文字：行 2–4 与行 20–22 整行有墨。

```text
profile(y) = [0,0,20,20,20,0,...,0,20,20,20,0,...]
occupied   = [F,F,T,T,T,F,...,F,T,T,T,F,...]
```

`_true_runs` 得到两段：`(2, 5)` 与 `(20, 23)`。

合并判断：`20 - 5 = 15`。

- `min_gap = 12` → `15 >= 12`，两段分开，产出 2 个块；
- `min_gap = 16` → `15 < 16`，合并成 `(2, 23)`，只产出 1 个块。

换轴后每个块再按列投影切。若某块内部还有左右两栏（列 1–8 与列 12–19 有墨），`axis=1` 的 `_split_segments` 会再切出 2 个子块。

复现：

```bash
python -c "import numpy as np; from visioncite.layout.xycut import _split_segments; m=np.zeros((30,20),bool); m[2:5,:]=True; m[20:23,:]=True; print(_split_segments(m,0,12)); print(_split_segments(m,0,16))"
# [(2, 5), (20, 23)]
# [(2, 23)]
```

### ⑤ 自测 3 问

1. **为什么第一刀按行（`axis=0`）而不是按列？**
   要点：文档的主结构是水平文本行，行间留白最稳定；先横后竖配合"切不动就换轴"，能自然处理单栏、多栏、图注混排。

2. **"切不动时换轴重试"解决什么具体问题？**
   要点：多栏页面每一行都横跨两栏（横切切不开），但栏间有竖向空白；不换轴就会直接退化成整页一块，证据框失去意义。见 `_cut()` 里 `alt_axis = 1 - axis` 那一段。

3. **`min_gap` 从 12 调到 40、再调到 0，分别会发生什么？**
   要点：调大 → 相邻行被合并，块变大变少，引用高亮更粗、Cite-P 下降；调 0 → 只要有一行空白就切，块数暴涨，一句话被拆散，BM25 长度归一反而让碎块占便宜。默认 12 是 144dpi 下"行距 vs 段距"的经验折中。

---

## 第 4 讲 表格线检测与区域分类

### ① 这一讲解决什么问题

表格是数字最密集的区域，但 XY-cut 会把它切成一条条横带，列关系全丢。这一讲先用"贯穿全宽的长直线"把整张表框出来，再决定每个块的类型（正文/表格/图/未知），并按类型拼接文本（表格用 `a | b` 保留列边界）。分类规则刻意保持三条、可解释，方便你在评测里一眼看出错在哪一类。

### ② 核心概念与公式

线带检测（`rule_bands`）：

```text
line_ratio = mask.mean(axis=1)        # axis=0 → 每一行的墨迹占比
strong = line_ratio >= min_ratio      # 默认 0.25
band   = 连续 strong 区间，且 (end - start) <= 12   # 太厚的深色块不是线
```

表格判定（`find_tables`）：

```text
if len(horizontal) < 3 or len(vertical) < 2: return []
bbox = (vertical[0].start, horizontal[0].start, vertical[-1].end, horizontal[-1].end)
if (x1-x0) < 40 or (y1-y0) < 30: return []          # 太小的多半是误判
rows = len(horizontal) - 1
cols = len(vertical) - 1
```

区域分类（`regions.classify`）：

```text
1. 与某表格候选的交集 / 块面积 > 0.5   → table
2. 块内没有词，但 ink_density > 0.05   → figure
3. 其余                                → text
（无词且墨迹也很少 → unknown）
```

两个配套规则：

- 行聚类 `group_rows`：以**中位字高**乘以 `y_tolerance=0.6` 作为"同一行"的容差（比直接按 y 排序稳，上下标不会被打散）；
- 阅读顺序 `reading_order`：按 `y0 // 40` 分带，带内按 `x0` 排序（简单分带，不做复杂版面树）。

### ③ 对应代码

- `src/visioncite/layout/table.py` — `TableCandidate`、`rule_bands(mask, axis, min_ratio=0.25)`、`find_tables(mask, min_ratio=0.25)`、`_binary_runs(flags)`。
- `src/visioncite/layout/regions.py` — `build_regions()`、`classify()`、`build_text()`、`group_rows()`、`assign_words()`、`_owner_index()`、`tighten_bbox()`、`merge_overlapping()`、`_first_overlapping()`、`_drop_tiny()`、`_mean_confidence()`、`reading_order()`。

### ④ 一个手算小例子

一张 300x300 的墨迹矩阵：横线在 y=40、120、200、280，每条约 2px 厚，从 x=40 画到 x=260（跨度 220）；竖线在 x=40、150、259，从 y=40 到 y=282（跨度 242）。

横线占比：`220 / 300 = 0.733 >= 0.25` → 4 条横线带。
竖线占比：`242 / 300 = 0.807 >= 0.25` → 3 条竖线带。
厚度都是 2px（≤12），不会被"太厚"过滤。

于是 `find_tables` 返回一个候选项：

```text
bbox = (40, 40, 261, 282)      面积 = 221 * 242 = 53482 > 40*30
rows = 4 - 1 = 3
cols = 3 - 1 = 2
```

注意 **`cols` 只是"竖线条数 - 1"，是估计值而不是真值**：这张图有 3 条竖线（左边框、1 条内部分隔、右边框），`cols = 2` 恰好等于 2 个单元格宽；`rows = 3` 也恰好等于 3 行。公式在"边框完整、无合并单元格"时是对的。但 `find_tables` 只返回外接框与两个计数，**不返回线的坐标**，所以想真的是把表格切成单元格，必须回到 `rule_bands` 重新取线（见练习 6）。真正会低估列数的场景：表格没有左右边框（少 2 条线）、合并单元格（少内部线）、竖线太短（不满足整列占比阈值）。

再算一次 `classify` 的比例：假设某块 `BBox(60, 120, 240, 200)`，面积 `180*80=14400`，与表格框交集 `180*80=14400`（完全在内），`14400/14400 = 1.0 > 0.5` → 判为 `table`。若块只有一半压在表上（交集 7000），`7000/14400 = 0.486` → 不判为表格。

复现：

```bash
python -c "import numpy as np; from visioncite.layout.table import find_tables; m=np.zeros((300,300),bool); [m.__setitem__((slice(y,y+2), slice(40,260)), True) for y in (40,120,200,280)]; [m.__setitem__((slice(40,282), slice(x,x+2)), True) for x in (40,150,259)]; print([(t.bbox.to_list(), t.rows, t.cols) for t in find_tables(m)])"
# [([40, 40, 261, 282], 3, 2)]
```

### ⑤ 自测 3 问

1. **三线表为什么检测不到？**
   要点：三线表只有表头上下两条横线（甚至没有竖线），`len(horizontal) < 3 or len(vertical) < 2` 直接返回空；要处理它得换成线段检测（Hough）+ 文字对齐，或者接受"表格是 text 区域"的降级。

2. **`cols` 为什么会低估真实列数？**
   要点：`cols = 竖线数 - 1`，而竖线必须满足"整列墨迹占比 >= 0.25 且厚度 <= 12"；没有左右边框（少 2 条线）、合并单元格（少内部线）、竖线太短或不连续、表宽小于 40px（整体被拒），都会让计数偏小。它是估计值，不要当布局真值用。

3. **`classify` 用 `inter / box.area > 0.5` 而不是 IoU，为什么？**
   要点：XY-cut 的块常常比表格小，用"块被表格覆盖的比例"判断归属更稳；IoU 会被两侧面积差异拉低，导致明明整块都在表格里却判成 text。

---

## 第 5 讲 OCR 适配器与降级

### ① 这一讲解决什么问题

OCR 是全项目唯一的重外部依赖：要装 onnxruntime、要加载模型、可能因为网络或权重损坏失败、单页还可能推理崩溃。这一讲解决"如何让一个会失败的外部能力变成系统的可选插件"——接口只有两个方法，工厂负责选实现，失败一律降级为"没有文字"，并让 `/health` 如实报告当前用的是哪个后端。

### ② 核心概念与公式

接口（`Protocol`，结构化子类型）：

```python
class OcrBackend(Protocol):
    name: str
    def available(self) -> bool: ...
    def recognize(self, image: NDArray[np.uint8]) -> list[Word]: ...
```

几何归一：OCR 常见输出是四边形（4 个点，可能倾斜），项目统一转成轴对齐框：

```text
bbox = (floor(min x), floor(min y), ceil(max x), ceil(max y))
```

置信度过滤：`filter_confidence(words, min_confidence)` 保留 `confidence >= min_confidence`，默认 `min_confidence = 0.5`。

工厂决策顺序（`create_ocr`）：

```text
ocr_mode == "off"                  → NullOcrBackend
RapidOcrBackend().available() 为真 → RapidOcrBackend
否则                               → logger.warning + NullOcrBackend
```

`available()` 只做 `import rapidocr_onnxruntime`，**不加载模型**；真正的加载发生在首次 `recognize()` 里的 `_get_engine()`（懒加载 + `_load_failed` 记忆化）。

页级降级（`pipeline._analyze_page`）：

```text
words = ocr.recognize(page.image)
if not words and page.text_layer:
    page.regions = [_text_layer_region(page)]   # 整页一个 text 区域
else:
    page.regions = build_regions(page.index, page.image, words)
```

### ③ 对应代码

- `src/visioncite/ocr/base.py` — `OcrBackend`、`polygon_to_bbox(points)`、`filter_confidence(words, min_confidence)`。
- `src/visioncite/ocr/factory.py` — `create_ocr(settings)`。
- `src/visioncite/ocr/rapidocr_backend.py` — `RapidOcrBackend(min_confidence=0.5)`、`available()`、`recognize()`、`_get_engine()`、`_to_words(raw)`、`create_rapidocr()`。
- `src/visioncite/ocr/null_backend.py` — `NullOcrBackend`（`available()` 恒 True，`recognize()` 恒空）。
- `src/visioncite/pipeline.py` — `_analyze_page()`、`_text_layer_region()`。
- `src/visioncite/errors.py` — `OcrUnavailableError`（已定义；当前实现里没有地方抛出它，属于"契约先行"的预留）。

### ④ 一个手算小例子

RapidOCR 返回一条 `[[[10.5, 20.2], [90.7, 20.4], [91.0, 48.9], [10.2, 48.6]], "合计金额：1,286.40 元", 0.93]`。

```text
min x = 10.2 → floor = 10      max x = 91.0 → ceil = 91
min y = 20.2 → floor = 20      max y = 48.9 → ceil = 49
bbox = BBox(10, 20, 91, 49)    宽 81，高 29，面积 2349
```

置信度 0.93 ≥ 0.5 → 保留。若另有一条 score=0.42 的噪声词，`filter_confidence` 会丢掉它（`0.42 < 0.5`），最终 `_to_words` 只返回 1 个 `Word`。

换成 PDF 且没有 OCR 的情况：`recognize()` 返回 `[]`，页面有文本层（例如 36 个字符），则 `_text_layer_region()` 造出 `Region(region_id="p0-r0", bbox=BBox(0, 0, 640, 400), kind="text", text=文本层, confidence=1.0)`——整页一个框，引用只能精确到页。

复现：

```bash
python -c "from visioncite.ocr.base import polygon_to_bbox, filter_confidence; from visioncite.types import Word, BBox; print(polygon_to_bbox([[10.5,20.2],[90.7,20.4],[91.0,48.9],[10.2,48.6]])); print(len(filter_confidence([Word('a',BBox(0,0,1,1),0.9), Word('b',BBox(0,0,1,1),0.42)], 0.5)))"
```

### ⑤ 自测 3 问

1. **`recognize()` 推理失败为什么返回 `[]` 而不抛异常？**
   要点：OCR 是可选能力，单页失败不该让整份文档失败；降级后仍有文本层或空文本，链路可继续；`logger.exception` 留下可排查的痕迹。`OcrUnavailableError` 是留给"显式要求 OCR 但环境不可用"那类调用方的，当前代码没有抛。

2. **`available()` 只检查 import，有什么意义？**
   要点：不触发模型下载与初始化，`create_ocr` 与 `/health` 可以廉价判断能力；真正的重活推迟到首次识别（懒加载），冷启动不会被拖住。

3. **四边形转轴对齐框丢了什么？什么时候会咬人？**
   要点：丢旋转角与透视。倾斜 5 度的文本行，轴对齐框会明显变高，`tighten_bbox` 后的 IoU 下降、裁图带进多余背景；CLIP 裁图和引用高亮都会受影响。

---

## 第 6 讲 中文分词与字 bigram

### ① 这一讲解决什么问题

中文没有空格，BM25 需要 token。引入 jieba 之类的分词器会带来词典依赖、冷启动成本，而且"总金额"被切成什么词完全取决于词典版本。这一讲用"字 bigram"换掉分词器：无词典、对未登录词稳、可解释。代价是索引膨胀与虚词没有停用词表，需要靠 IDF 自己压下去。

### ② 核心概念与公式

归一化四步（`normalize`）：

```text
1. unicodedata.normalize("NFKC", text)   全角 → 半角（１ → 1，： → :，？ → ?）
2. re.sub(r"(?<=\d),(?=\d{3}\b)", "")    去千分位逗号
3. text.lower()
4. re.sub(r"\s+", " ").strip()           压缩空白
```

切分模式（`_TOKEN_PATTERN`）：

```text
[\u4e00-\u9fff]+         连续中文 → bigram 滑窗
|[a-z]+                  连续英文 → 整体一个 token
|[0-9]+(?:\.[0-9]+)?     数字（含小数）→ 整体一个 token
```

中文 bigram：

```text
len(piece) == 1 → [piece]                     单字保留
否则            → [piece[i:i+2] for i in range(len(piece)-1)]
```

切块（`chunking.py`）：按句末标点 `(?<=[。！？!?；;])|(?<=\n)` 切句，再贪心装箱到 `max_chars=240`，不重叠；块 id 为 `f"{doc_id}:{region.region_id}:{index}"`。一个例外必须知道：如果**单句本身就超过 240 字**（OCR 常把整页连成一段、没有句号），`_pack()` 会走 `_hard_split()` 按字符硬切，否则会产出一个"包住整页"的块，检索分数被 `avgdl` 稀释、答案也会过长。

### ③ 对应代码

- `src/visioncite/index/tokenize.py` — `normalize()`、`tokenize()`、`_is_cjk()`、`_cjk_tokens()`、常量 `_CJK`、`_TOKEN_PATTERN`、`_THOUSANDS`。
- `src/visioncite/index/chunking.py` — `split_sentences()`、`chunk_region(doc_id, page, region, max_chars=240)`、`_pack(sentences, max_chars)`、`_hard_split(text, max_chars)`、`chunks_from_pages()`、`_SENTENCE_END`。
- 复用方：`index/bm25.py`、`answer/extractive.py`、`answer/verify.py`、`eval/metrics.py` 都调用同一个 `tokenize`，保证"检索、作答、评测"三处口径一致。

### ④ 一个手算小例子

输入 OCR 文本 `"合计金额：1,286.40元"`。

```text
normalize → "合计金额:1286.40元"        （全角冒号变半角，千分位逗号被去掉）
finditer → "合计金额" / "1286.40" / "元"
tokenize → ["合计", "计金", "金额", "1286.40", "元"]     共 5 个 token
```

注意 `"1286.40"` 是一个 token（小数被 `[0-9]+(?:\.[0-9]+)?` 整体捕获），而 `"1,286.40"` 里的逗号在归一化阶段已经删掉。

再算两个容易错的：

```text
"INV-2026-100" → "inv" "2026" "100"        （连字符把字母与数字断开，3 个 token）
"ECE0.08"      → "ece" "0.08"              （字母数字粘连被拆开，OCR 常见）
"BM25"         → "bm" "25"                 （同上；想整体匹配 BM25 必须改模式）
```

切块计数：区域文本 300 字，句子边界让贪心装箱装出 `[238 字, 62 字]` 两块（`len(current)+len(sentence) > 240` 时落板），所以 `chunk_region` 返回 2 个 Chunk，id 为 `"{doc}:p0-r1:0"` 与 `"{doc}:p0-r1:1"`。

复现：

```bash
python -c "from visioncite.index.tokenize import normalize, tokenize; print(normalize('合计金额：1,286.40元')); print(tokenize('合计金额：1,286.40元')); print(tokenize('ECE0.08'), tokenize('BM25'))"
```

### ⑤ 自测 3 问

1. **为什么不用 jieba？bigram 的代价是什么？**
   要点：无词典依赖、冷启动快、对专名与未登录词更稳、行为可解释；代价是索引规模约等于字数（bigram 数 ≈ 字数）、虚词（的/了）没有停用词表只能靠 IDF 降权、语义粒度比词粗。

2. **查询"金额"如何命中原文"总金额"？**
   要点："总金额" → `总金`、`金额`，bigram 里保留了 `金额`，所以即使分词器把"总金额"切成别的词，检索仍能命中。这就是 bigram 的"抗分词错误"特性。

3. **`normalize` 为什么必须 NFKC 在前、去千分位在后？**
   要点：OCR 常输出全角 `１，２８６.４０`，`，` 不是半角逗号，去千分位正则 `(?<=\d),(?=\d{3}\b)` 匹配不到；NFKC 先把全角转半角，正则才生效。`tests/test_index.py::test_normalize_fullwidth_and_thousands` 就是钉这个顺序的。

---

## 第 7 讲 BM25

### ① 这一讲解决什么问题

拿到一堆检索块后，要回答"哪一块最可能包含问题的答案"。BM25 用词频饱和 + 文档长度归一 + IDF 三个机制，让"精确词命中"排前面，又不让长块靠体量取胜。它是稀疏检索的工业默认值，也是这个项目里唯一"分数有明确公式"的一路召回。

### ② 核心概念与公式

```text
score(q, d) = Σ_{t in q} IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))

IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

length_norm(d) = 1 - b + b·|d|/avgdl
tf_component   = tf·(k1+1) / (tf + k1·length_norm)
```

参数默认值来自 `Settings`：`bm25_k1=1.5`、`bm25_b=0.75`。

实现里的四个选择：

1. 查询词去重（`for term in set(terms)`）——同一个查询词只累加一次 IDF，不做 query tf 饱和（查询通常很短）；
2. `avgdl = sum(doc_lengths) / size or 1.0`——空索引时避免除零；
3. 倒排是 `dict[str, dict[int, int]]`（词 → {文档下标: 词频}），检索时对查询词取并集累加；
4. 只给有命中的文档打分（`scores` 是 `defaultdict(float)`，没命中的不会出现），最后 `sorted(...)[:top_k]`。

### ③ 对应代码

- `src/visioncite/index/bm25.py` — `BM25Index(k1=1.5, b=0.75)`、`fit(documents)`、`size` 属性、`search(query, top_k=8)`、`_idf(df)`、`_tf_component(tf, doc_length, avgdl)`。
- 建索引：`src/visioncite/pipeline.py` 的 `build_document()` 里 `bm25.fit([(chunk.chunk_id, chunk.text) for chunk in chunks])`。
- 检索：`src/visioncite/index/retrieve.py` 的 `retrieve()` 里 `document.bm25.search(question, top_k=settings.top_k * 3)`（先多召回 3 倍，再融合、去冗余）。

### ④ 一个手算小例子

三篇文档（用英文是为了让数字干净；中文 token 数会翻倍但公式相同）：

```text
d1 = "clip retrieval clip"   → tokens: clip, retrieval, clip   |d1| = 3
d2 = "retrieval fusion"      → tokens: retrieval, fusion       |d2| = 2
d3 = "table layout"          → tokens: table, layout           |d3| = 2
查询 q = "clip retrieval"
N = 3,  avgdl = (3+2+2)/3 = 7/3 = 2.3333
df(clip) = 1,  df(retrieval) = 2
```

IDF：

```text
IDF(clip)      = ln(1 + (3-1+0.5)/(1+0.5)) = ln(1 + 1.6667) = ln(2.6667) = 0.9808
IDF(retrieval) = ln(1 + (3-2+0.5)/(2+0.5)) = ln(1 + 0.6)    = ln(1.6)    = 0.4700
```

d1（`|d|=3`，`length_norm = 0.25 + 0.75*(3/2.3333) = 0.25 + 0.9643 = 1.2143`）：

```text
tf(clip)=2: 2*2.5/(2 + 1.5*1.2143) = 5/3.8214 = 1.3084  → 0.9808*1.3084 = 1.2833
tf(retrieval)=1: 2.5/(1 + 1.8214)  = 0.8861             → 0.4700*0.8861 = 0.4165
score(d1) = 1.6998
```

d2（`|d|=2`，`length_norm = 0.25 + 0.75*(2/2.3333) = 0.8929`）：

```text
tf(retrieval)=1: 2.5/(1 + 1.5*0.8929) = 2.5/2.3393 = 1.0687 → 0.4700*1.0687 = 0.5023
score(d2) = 0.5023
```

d3 无命中，`score(d3) = 0`。排序 `d1 > d2 > d3`。

两个值得注意的数字：`clip` 出现 2 次只贡献 1.2833，而不是 1 次时的 0.6417 的两倍——这就是 `k1` 的饱和效应；d1 更长（3 vs 2），`length_norm` 更大，tf 项被额外压了一点。

复现：

```bash
python -c "from visioncite.index.bm25 import BM25Index; b=BM25Index(1.5,0.75); b.fit([('d1','clip retrieval clip'),('d2','retrieval fusion'),('d3','table layout')]); print(b.search('clip retrieval', top_k=3)); print(b._idf(1), b._idf(2))"
# [('d1', 1.6997869065846745), ('d2', 0.5022939549191068)]
# 0.9808292530117262 0.4700036292457356
```

### ⑤ 自测 3 问

1. **IDF 为什么写成 `ln(1 + x)` 而不是 `ln(N/df)`？**
   要点：保证非负。经典形式在 `df > N/2` 且带 0.5 平滑时可能出负分，导致"命中反被扣分"；`ln(1+·)` 把下界抬到 0，代价是常见词与罕见词的区分度略降。

2. **`b=0` 与 `b=1` 分别是什么行为？**
   要点：`b=0` 完全忽略长度，长块靠体量占便宜；`b=1` 完全按 `|d|/avgdl` 归一，短块占便宜；0.75 是折中，也是第 4 讲里"表格块很长导致分数被压"的直接原因。

3. **为什么检索时用 `set(terms)`？**
   要点：查询里重复出现的词只应贡献一次 IDF，否则"金额金额"会把分数翻倍；短查询的 query tf 饱和意义不大，直接去重更简单可测。

---

## 第 8 讲 CLIP 与对比学习

### ① 这一讲解决什么问题

BM25 只认字面：问"这张图展示了什么趋势"，图里一个字都没有，文本检索必然为空。CLIP 把图片和文字映射到同一个向量空间，于是"用文字查图片"变成一个余弦相似度排序问题。这一讲说明它为什么有效、在本项目里只做召回不做依据、以及不可用时链路怎么优雅退化。

### ② 核心概念与公式

对比学习（InfoNCE，对称形式）。一个 batch 有 `N` 对（图片, 文本），对角线是正例，其余 `N^2 - N` 个是负例，温度 `τ` 控制分布的尖锐程度：

```text
s_ij = cosine(z_i^img, z_j^txt) / τ
L_i  = -log( exp(s_ii) / Σ_j exp(s_ij) )
L    = (1/N) Σ_i L_i          # 对称版本再对文本方向做一次并平均
```

关键点：

- 分母包含**整个 batch**，所以 batch 越大负例越多，学到的表征越好（CLIP 用 4 亿对、batch 数万）；
- `τ` 越小分布越尖，损失越容易被"已经分开的正例"压到 0；`τ` 越大越接近均匀分布，损失趋近 `ln N`；
- 推理时只需要单塔输出向量：`z = normalize(f(x))`，相似度 = 点积。

L2 归一化让余弦退化为点积：

```text
z ← z / ||z||_2        →  cos(a,b) = a·b / (||a||·||b||) = a·b
```

本项目里 CLIP 的定位（`index/visual.py` 模块注释写得很直白）：擅长"这张图大概是什么"，不擅长"这张发票的金额是多少"，所以**只做视觉召回的一路，不做最终答案依据**。

### ③ 对应代码

- `src/visioncite/index/visual.py`
  - `ClipVisualEncoder(model_id, cache_dir, batch_size=8)`、`available()`（只查 `torch`/`transformers` 是否可导入）、`load()`（首次下载权重，失败置 `_load_failed`）、`encode_images(images)`、`encode_texts(texts)`。
  - 取向量的两条路径：`_image_features(pixel_values)` / `_text_features(inputs)` 优先直接走 `vision_model`+`visual_projection`、`text_model`+`text_projection`（transformers 5.x 的 `CLIPModel.forward` 要求同时给文本输入，这样写跨版本更稳）；拿不到时退回 `_pick_embeds(output, kind)`（兼容 `image_embeds`/`text_embeds`/`pooler_output`/`last_hidden_state` 的字段名差异）。
  - 辅助：`_to_numpy(tensor)`、`_l2_normalize(matrix)`。
  - `NullVisualEncoder` — `available()` 恒 False，编码函数抛 `VisualUnavailableError`。
  - `create_visual(settings)` — `visual_mode == "off"` 或依赖缺失时返回 `NullVisualEncoder`。
- `src/visioncite/pipeline.py` — `_encode_chunks()`：按 `(page, region_id)` 去重裁图、`bbox.pad(4, (width, height))` 留边、`np.stack` 对齐到 chunk 顺序、任何异常都返回 `None`。
- `src/visioncite/index/retrieve.py` — `_visual_search()`：`document.vectors @ query_vector`（已归一化，所以就是余弦），`np.argsort(-similarities)[: top_k*3]`。

### ④ 一个手算小例子

设一个 2 对的 batch，相似度矩阵（已除过温度之前的原始余弦）为：

```text
        t1     t2
i1     0.9    0.3
i2     0.2    0.8
```

取 `τ = 0.5`：

```text
第 1 行 logits = [0.9/0.5, 0.3/0.5] = [1.8, 0.6]
exp = [6.0496, 1.8221],  Σ = 7.8717
p_11 = 6.0496/7.8717 = 0.7685
L_1  = -ln(0.7685) = 0.2632

第 2 行 logits = [0.4, 1.6]
exp = [1.4918, 4.9530],  Σ = 6.4448
p_22 = 4.9530/6.4448 = 0.7685
L_2  = 0.2632

L = (0.2632 + 0.2632)/2 = 0.2632
```

再换温度感受一下：`τ → 0.05` 时 logits 变成 `[18, 6]`，`p_11 ≈ 0.999996`，损失 ≈ 0.000004（太容易，梯度消失）；`τ = 10` 时 logits 变成 `[0.09, 0.03]`，`p_11 ≈ 0.515`，损失 ≈ 0.664，接近 `ln 2 = 0.693`（完全分不开）。

余弦手算：`a = [3, 4]`（模 5），`b = [1, 0]`，`cos = (3*1 + 4*0)/(5*1) = 0.6`；归一化后 `a' = [0.6, 0.8]`，`a'·b = 0.6`，结果一致。

### ⑤ 自测 3 问

1. **为什么训练/推理都先做 L2 归一化？**
   要点：模长被固定为 1 后，余弦相似度等于点积，矩阵乘法一次算完整个候选集；也避免"向量越长分数越大"的尺度漂移。见 `_l2_normalize()` 与 `retrieve._visual_search()` 的 `@`。

2. **CLIP 为什么答不了"这张发票合计金额是多少"？**
   要点：它的文本塔编码的是"整段描述的语义"，不是逐字识别；密集小字、表格数字在 224x224 的输入里被压扁，数字级信息基本丢失。所以答案仍要靠 OCR + BM25 + 抽取，CLIP 只负责"哪块看起来相关"。

3. **视觉不可用时，检索链路会崩吗？**
   要点：不会。`build_document` 里 `vectors=None`、`visual_name="null"`；`_visual_search` 直接返回 `[]`；`retrieve` 的 `rankings` 只剩文本一路，RRF 退化为文本排序；`_diversify` 走 `candidates[: settings.top_k]` 的截断分支。

---

## 第 9 讲 RRF 与 MMR

### ① 这一讲解决什么问题

两路召回各有盲区，但它们的分数完全不可比：BM25 无上界（本例中 1.70），余弦在 `[-1, 1]`。直接加权求和要调权重、还对离群值敏感。同时，只按相关性排序会把"同一段的上下两块"一起返回，证据列表看起来很多、实际信息重复。这一讲用 RRF 解决"怎么合"，用 MMR 解决"怎么不重复"。

### ② 核心概念与公式

RRF（Reciprocal Rank Fusion），只用名次：

```text
score(c) = Σ_r  w_r / (k + rank_r(c) + 1)
```

`rank` 从 0 开始，`k` 默认 60（`Settings.rrf_k`）。`k` 越大，头部名次优势越被削平。

MMR（Maximal Marginal Relevance），逐轮贪心：

```text
MMR(d) = λ · sim(q, d) - (1 - λ) · max_{s in selected} sim(d, s)
```

`λ=1` 退化为纯相关性，`λ=0` 只追求多样性；默认 `mmr_lambda=0.7`。第一个被选中的永远是相关性最高者（`selected` 为空时返回 `relevance[index]`）。

对外分数归一化（`retrieve.retrieve`）：

```text
top = fused[0][1]
score(c) = min(1.0, fused_score(c) / top)
```

排序不受这个缩放影响，但它让 `Settings.min_citation_score` 之类的阈值有意义。

### ③ 对应代码

- `src/visioncite/index/fuse.py` — `reciprocal_rank_fusion(rankings, k=60, weights=None)`、`cosine_similarity(a, b)`、`mmr_select(chunk_ids, relevance, embeddings, k, lambda_=0.7)`、`_mmr_score(...)`。
- `src/visioncite/index/retrieve.py` — `Retrieved`、`retrieve(document, question, settings)`、`_visual_search(...)`、`_diversify(...)`。
- 配置：`src/visioncite/config.py` 的 `rrf_k=60`、`mmr_lambda=0.7`、`top_k=8`。

一个需要你知道的真实细节：`_diversify()` 传给 `mmr_select` 的 `relevance` 是**未归一化的 RRF 分数**（量级 0.03），而多样性项是 `[0,1]` 的余弦。量纲不匹配时，`0.7*0.03 - 0.3*0.9` 会是负数，多样性项几乎完全主导。这不是 bug 报告，而是这套代码当前的真实行为；练习 9 就是把它改成"归一化后再做 MMR"并跑 A/B。

### ④ 一个手算小例子

两路召回，`k = 60`：

```text
文本路： [c1, c2, c3]        （名次 0,1,2）
视觉路： [c3, c1]            （名次 0,1）

score(c1) = 1/(60+0+1) + 1/(60+1+1) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032523
score(c3) = 1/(60+2+1) + 1/(60+0+1) = 1/63 + 1/61 = 0.015873 + 0.016393 = 0.032266
score(c2) = 1/62                      = 0.016129
```

结果 `c1 > c3 > c2`：`c1` 在两路都靠前（1+1 名），`c3` 是一路第一、一路第三。两路"共识"比单路第一更值钱，这正是 RRF 的假设。

MMR 手算（已选出 `d1`，候选 `d2`、`d3`，`λ=0.7`）：

```text
d2: relevance 0.9, 与 d1 的余弦 0.95 → 0.7*0.9 - 0.3*0.95 = 0.630 - 0.285 = 0.345
d3: relevance 0.6, 与 d1 的余弦 0.10 → 0.7*0.6 - 0.3*0.10 = 0.420 - 0.030 = 0.390
```

`d3` 相关性更低却排在 `d2` 前面——多样性赢了。实测选择顺序是 `d1, d3, d2`。

复现：

```bash
python -c "import numpy as np; from visioncite.index.fuse import reciprocal_rank_fusion, mmr_select; print(reciprocal_rank_fusion([['c1','c2','c3'],['c3','c1']], k=60)); rel=np.array([1.0,0.9,0.6],dtype=np.float32); emb=np.array([[1,0],[0.95,0.31],[0.1,0.995]],dtype=np.float32); print(mmr_select(['d1','d2','d3'], rel, emb, 3, 0.7))"
# c1 0.032522…, c3 0.032266…, c2 0.016129…
# ['d1', 'd3', 'd2']
```

### ⑤ 自测 3 问

1. **为什么 RRF 只用名次，不直接用分数加权？**
   要点：两路分数量纲与分布都不同（BM25 无上界、余弦有界），加权求和要调权重且对离群分数敏感；名次只保留相对顺序，对单调变换不变，且零可训练参数、易解释。

2. **`k` 从 60 改成 1000 会怎样？**
   要点：`1/(k+rank)` 之间的差异被压平，head 优势消失，排序更依赖"是否多路命中"的共识；`k` 变小时头部名次几乎决定一切，退化成"取两路并集后按最好名次排"。

3. **为什么最后要把分数除以 `top`？**
   要点：把对外分数压到 `0~1`，便于阈值判断与前端展示；同除一个正常数不改变顺序，所以不影响排序质量。注意 `min(1.0, ...)` 只为防御浮点误差。

---

## 第 10 讲 引用校验

### ① 这一讲解决什么问题

模型（尤其 VLM）会编造两样东西：不存在的 `chunk_id`，以及看起来合理的数字。引用校验把"看起来对的答案"筛成"有证据支撑的答案"：引用必须存在、数字必须能在证据里找到、答案的词面必须被证据覆盖。三条规则全部可测试、可解释，且不依赖任何模型。

### ② 核心概念与公式

四条检查（`verify_answer`）：

```text
1. has_answer        : answer.strip() 非空
2. citations_exist   : 每条 citation.chunk_id 都在可用 chunks 里
3. numbers_grounded  : numbers_in(answer) ⊆ numbers_in(被引证据)
4. lexical_support   : |{answer tokens} ∩ {evidence tokens}| / |answer tokens| >= min_support(0.5)
```

数字抽取与归一化：

```text
_NUMBER   = \d+(?:[.,]\d+)*(?:\s*%)?      整数/小数/千分位/百分号
_CURRENCY = [¥$€£]                        先替换成空格
归一化：normalize(text) → 去货币 → 匹配 → 去逗号与空格
```

词面支撑度的一个细节：`lexical_support` 会排除纯数字 token（`not token.isdigit()`），避免"12"这种短答案因为一个数字没命中就整体判 0；但小数 `"1286.40"` 含点，`isdigit()` 为 False，仍会被计入。

引用命中（评测口径，`eval/metrics.py`）：

```text
citation_hit(citations, gold, threshold=0.5)
    = any(c.bbox.iou(gold) >= 0.5 for c in citations)

IoU(A, B) = 交面积 / (A.area + B.area - 交面积)
```

0.5 这个阈值与 COCO 的 AP50 同一约定。

### ③ 对应代码

- `src/visioncite/answer/verify.py` — `Verification(ok, reasons, checks)`、`numbers_in(text)`、`numbers_supported(answer, evidence_texts)`、`lexical_support(answer, evidence_texts)`、`citation_ids_exist(citations, chunks)`、`verify_answer(answer, citations, chunks, min_support=0.5)`、`_evidence_texts(...)`。
- `src/visioncite/answer/qa.py` — `_build_citations(cited, results)`（把块转成 `Citation`，`snippet=chunk.text[:160]`）、`_map_citations(ids, chunks)`（VLM 引用白名单）。
- `src/visioncite/answer/abstain.py` — `should_abstain(...)` 消费 `Verification`。
- `src/visioncite/types.py` — `Citation`、`BBox.iou()`、`BBox.intersection_area()`。
- `src/visioncite/eval/metrics.py` — `citation_hit()`、`citation_precision()`。

### ④ 一个手算小例子

证据原文 `E = "合计金额：1,286.40 元"`。

答案 A（正常抽取）：

```text
numbers_in(A) = ["1286.40"],  numbers_in(E) = ["1286.40"]  → 数字通过
answer tokens = [合计, 计金, 金额, 1286.40, 元]               （5 个）
evidence tokens 同样是这 5 个                                  → 支撑度 5/5 = 1.0
结论：ok = True
```

答案 B（模型自己算了个税额）：

```text
B = "合计金额：1,286.40 元，税额 128.64 元"
numbers_in(B) = ["1286.40", "128.64"]
"128.64" ∉ numbers_in(E)  → numbers_supported = False
reasons = ("答案中的数字不在证据里",)   → ok = False → 弃答
```

IoU 手算：标准证据框是整张表 `gold = [70, 200, 930, 550]`（面积 `860*350 = 301000`），模型只框住上半部分 `pred = [70, 200, 930, 300]`（面积 `860*100 = 86000`，完全被 gold 包含）。

```text
交 = 86000
并 = 301000 + 86000 - 86000 = 301000
IoU = 86000/301000 = 0.2857 < 0.5   → citation_hit = False
```

也就是说"框得不够准"会被判为未命中，Cite-P 会掉；这也是第 3 讲 `tighten_bbox` 存在的理由。

复现：

```bash
python -c "from visioncite.answer.verify import numbers_in, numbers_supported, lexical_support; e='合计金额：1,286.40 元'; print(numbers_in(e), numbers_supported(e,[e]), lexical_support(e,[e])); print(numbers_supported('合计金额：1,286.40 元，税额 128.64 元',[e]))"
# ['1286.40'] True 1.0
# False
```

### ⑤ 自测 3 问

1. **校验失败为什么用返回值而不是抛 `VerificationError`？**
   要点：弃答是正常业务结局，不是异常；抛异常会打断 `answer_question` 的编排，还要到处 `try/except`。`errors.py` 里定义了 `VerificationError` 但目前没有任何地方抛出，属于契约预留——你可以当它是"读代码时的一个已知不一致"。

2. **`numbers_in` 为什么要先 `normalize` 再抽数字？**
   要点：全角数字、千分位、货币符号会让同一个数出现多种写法（`1,286.40` / `￥1,286.40` / `１２８６.４０`）；不归一化就会出现"证据里明明有，校验说没有"的假弃答。

3. **`min_support=0.5` 对短答案有什么风险？实现怎么缓解？**
   要点：短答案 token 少，一个未命中就跌破 0.5；实现把纯数字 token 排除在分母外，缓解"12"这类答案被误杀。但含小数点的数字仍计入，所以极短的小数答案仍可能误弃答——这是练习 11 的改进点。

---

## 第 11 讲 弃答与 ECE

### ① 这一讲解决什么问题

"少答对一题"远好于"多答错一题"。但要弃答得准，需要一个可信的置信度：太低会误弃答（coverage 掉），太高会答错（accuracy 掉）。这一讲讲置信度是怎么拼出来的、三档弃答的顺序为什么不能换，以及怎么用 ECE 判断这个置信度是不是"过自信"。

### ② 核心概念与公式

抽取式作答的打分（`extractive.py`）：

```text
coverage = 命中的问题词数 / 问题词总数
density  = 命中次数 / sqrt(句子 token 数)
score    = 0.7 * coverage + 0.3 * min(1.0, density)
```

置信度（三信号加权）：

```text
margin     = max(0, (best.score - second.score) / best.score)     # 只有一个候选时 = 1
confidence = 0.55 * coverage + 0.25 * margin + 0.20 * min(1.0, score)
```

三档弃答（顺序即优先级，`should_abstain`）：

```text
1. citation_count == 0                      → 弃答（没有证据）
2. not verification.ok                      → 弃答（证据没支撑住答案）
3. confidence < settings.abstain_threshold  → 弃答（默认 0.35）
4. 否则                                     → 作答
```

ECE（Expected Calibration Error，10 个等宽分箱）：

```text
ECE = Σ_b (n_b / N) * |acc_b - conf_b|
acc_b  = 该箱内答对比例
conf_b = 该箱内平均置信度
分箱规则：(low, high]，且 confidence == 0 的样本强制进第一箱
```

评测里的"答对"定义为 `contains >= 1.0` 且未弃答（见 `eval/metrics.summarize`）。

### ③ 对应代码

- `src/visioncite/answer/extractive.py` — `_QUESTION_CHARS`、`STOPWORDS`、`Candidate`、`query_terms()`、`score_sentence()`、`rank_candidates()`、`_sentences()`、`compose_answer()`、`_margin()`、`_confidence()`、`_cited_chunks()`。
- `src/visioncite/answer/abstain.py` — `should_abstain(confidence, citation_count, verification, settings)`。
- `src/visioncite/config.py` — `abstain_threshold=0.35`、`min_citation_score=0.02`。
- `src/visioncite/eval/metrics.py` — `expected_calibration_error(confidences, corrects, bins=10)`、`summarize(records)`。

`query_terms` 的两个工程细节值得单独看：它按**字符**删除疑问词（`_QUESTION_CHARS`），因为中文没有空格、"是多少"被切成 `是多/多少` 两个 bigram，按词表删不干净；删除后再切 bigram。副作用是会产生跨词边界的"伪 bigram"（下面例子里的 `金元`），这是真实行为。

### ④ 一个手算小例子

问题 `q = "合计金额是多少元？"`，证据句 `s = "合计金额：1,286.40 元"`（与 `reports/` 里 invoice 类样本同构）。

第一步，提取问题词：

```text
删疑问字符（是/多/少）→ "合计金额元？"
normalize → "合计金额元?"
tokenize → "合计金额元" 是连续中文 → bigram: 合计, 计金, 金额, 金元
terms = [合计, 计金, 金额, 金元]        共 4 个（金元 是删字符后的伪 bigram）
```

第二步，给句子打分：

```text
sentence tokens = [合计, 计金, 金额, 1286.40, 元]       共 5 个
命中 terms: 合计(中) 计金(中) 金额(中) 金元(不中)          3 / 4
coverage = 0.75
density  = 3 / sqrt(5) = 3/2.2361 = 1.3416 → min(1.0, ·) = 1.0
score    = 0.7*0.75 + 0.3*1.0 = 0.525 + 0.300 = 0.825
```

第三步，置信度（只有 1 个候选，`margin = 1.0`）：

```text
confidence = 0.55*0.75 + 0.25*1.0 + 0.20*0.825
           = 0.4125 + 0.2500 + 0.1650 = 0.8275
```

0.8275 ≥ 0.35，不弃答。这个数字与仓库里 layout 模式报告中的 invoice 金额题完全一致，可以互相对照。

第四步，ECE 手算（`bins=2`，让分箱只有两个）：

```text
confidences = [0.9, 0.8, 0.35, 0.2]
corrects    = [1,   0,   1,    0  ]

箱 1 (0, 0.5]: 0.35(对), 0.2(错) → n=2, conf=0.275, acc=0.5 → |0.5-0.275| = 0.225
箱 2 (0.5, 1]: 0.9(对),  0.8(错) → n=2, conf=0.850, acc=0.5 → |0.5-0.850| = 0.350

ECE = 0.5*0.225 + 0.5*0.350 = 0.1125 + 0.1750 = 0.2875
```

复现：

```bash
python -c "from visioncite.answer.extractive import query_terms, score_sentence; q='合计金额是多少元？'; s='合计金额：1,286.40 元'; print(query_terms(q)); print(score_sentence(s, query_terms(q)))"
python -c "from visioncite.eval.metrics import expected_calibration_error as e; print(e([0.9,0.8,0.35,0.2],[True,False,True,False],bins=2))"
# 0.28750000000000003
```

### ⑤ 自测 3 问

1. **为什么必须先判 `citation_count == 0`，再判置信度？**
   要点：没有引用时 `margin` 往往为 1（单候选），置信度可能虚高到 0.8，只看阈值会答出"没有证据支撑的答案"；顺序是"有证据 → 被支撑 → 够自信"，先决条件不能交换。

2. **`coverage = 1` 但 `margin = 0` 时置信度是多少？说明什么？**
   要点：`0.55*1 + 0.25*0 + 0.2*min(1, score)`；若 `score=1` 则为 0.75，仍高于默认阈值 0.35。说明这个公式对"两个候选并列"的惩罚有限，并列时不会自动弃答——这是练习 4（温度缩放/校准）的动机之一。

3. **为什么不能只看一次运行的 ECE？**
   要点：ECE 是分箱统计量，样本少时每箱样本极少，`acc_b` 方差极大，且 ECE 本身没有置信区间；正确做法是在 bootstrap 重采样上同时报 ECE 与区间，并看分箱明细而不是一个标量。

---

## 第 12 讲 评测方法与 bootstrap

### ① 这一讲解决什么问题

"F1 0.8"这句话在面试里没有意义，除非你能说清：样本多少、分母是谁、弃答算不算错、区间多宽。这一讲把评测拆成三件事——可复现的语料与标注、口径明确的指标、带置信区间的比较——并用 `page/layout/hybrid` 三种模式做消融，回答"收益到底来自版面、来自文本、还是来自视觉"。

### ② 核心概念与公式

指标口径（`eval/metrics.py` 模块注释是第一手定义）：

```text
coverage          = 实际作答样本数 / 全部样本数
accuracy(AnswerAcc) = 在"可答 且 已作答"样本上，gold 出现在 prediction 中的比例
em                = 同上样本上，归一化后完全相等的比例
f1                = 同上样本上 token 级 P/R 的调和平均（复用 retrieve 的 tokenize）
cite_precision    = 引用中 IoU>=0.5 的比例
cite_recall       = 是否至少有一条引用命中（逐样本 0/1 再平均）
abstain_accuracy  = 可答题未弃答 + 不可答题已弃答 的比例
ece               = 校准误差（上一讲）
p50 / p95         = 延迟分位数（np.percentile 线性插值）
```

核心原则：**弃答不算答错，它是"没答"**。把弃答混进 accuracy 会让"多弃答"虚高准确率，所以 accuracy 与 coverage、abstain_accuracy 必须一起看。

bootstrap 均值置信区间（`eval/bootstrap.py`）：

```text
for b in 1..B:
    sample = 有放回抽 n 个（n = 原样本数）
    means[b] = mean(sample)
排序后取分位：
    lower = means[int((alpha/2) * B)]
    upper = means[int((1 - alpha/2) * B) - 1]
```

配对 bootstrap（消融 A/B 用）：

```text
diffs[i] = a[i] - b[i]        # 同一批样本上的逐样本差值
对 diffs 做上面的 bootstrap → 区间不含 0 通常认为差异显著
```

三种消融模式（`eval/run.py`）：

```text
page    : 每页压成"整页一个区域"（_flatten_to_page），没有版面结构 —— 基线
layout  : XY-cut 区域 + BM25 —— 只用文本结构
hybrid  : 区域 + BM25 + CLIP 视觉召回 —— 完整系统
```

### ③ 对应代码

- `src/visioncite/eval/run.py` — `MODES`、`build_documents()`、`evaluate()`、`run_mode()`、`write_report()`、`_flatten_to_page()`、`main(argv)`。
- `src/visioncite/eval/metrics.py` — `normalize_text()`、`exact_match()`、`answer_contains()`、`token_f1()`、`citation_hit()`、`citation_precision()`、`expected_calibration_error()`、`percentile()`、`EvalRecord`、`MetricsSummary.to_dict()`、`summarize()`、`safe_log()`。
- `src/visioncite/eval/bootstrap.py` — `bootstrap_mean_ci()`、`paired_bootstrap_ci()`。
- `src/visioncite/eval/dataset.py` — `QAItem.to_dict()` / `from_dict()`、`save_items()`、`load_items()`。
- `src/visioncite/eval/corpus_gen.py` — `find_cjk_font()`、`generate_corpus(root, per_template=2)`、`load_or_generate()`、`_write_summary()`、`CorpusFontError`。
- `src/visioncite/eval/templates.py` — `TEMPLATES`（`invoice`/`table`/`chart`/`slide`）、`Fonts`、`load_fonts()`、`new_canvas()`、`draw_line()`。

合成语料的关键取舍：模板用 `ImageDraw.textbbox` 在绘制时精确记录文本框，所以**标准答案与证据框是构造出来的真值**，引用 IoU 评测是严格的；代价是分布简单（无倾斜、无噪点、无手写），因此报告里明确写"不代表真实扫描件难度"。`find_cjk_font()` 找不到字体时直接抛 `CorpusFontError`，而不是静默生成豆腐块——避免把噪声当指标。

### ④ 一个手算小例子

仓库里一次参考运行（`--mode layout --per 2`，36 题 / 8 图；`reports/` 为空时重跑一次就会再生成）的结果：

```text
n = 36,  answered = 26,  answered & answerable = 24   （26 条作答里有 2 条其实是不可答题）
coverage        = 26/36 = 0.7222
accuracy        = 24/24 = 1.00        （可答且已作答样本全对）
f1              = 0.4027              （答案常是整句，gold 是片段 → P 低）
cite_precision  = 0.8333
cite_recall     = 1.00
abstain_accuracy= (可答未弃答 + 不可答已弃答)/36 = 30/36 = 0.8333
ece             = 0.1899
p50 = 0.30 ms, p95 = 0.64 ms
```

逐样本拆一下 `abstain_accuracy = 30/36`：4 题可答却被弃答（table 的"合计金额"与 slide 的"ECE 是多少"各 2 题，因为表/幻灯片上的数字没被 OCR 正确切出），2 题不可答却被作答（table 的"哪个产品退货最多"），`36 - 4 - 2 = 30`。

再算 CI 的下标（这就是 `bootstrap_mean_ci` 里那两行的含义）：

```text
B = 1000, alpha = 0.05
lower = means[int(0.025 * 1000)]     = means[25]
upper = means[int(0.975 * 1000) - 1] = means[974]

B = 10, alpha = 0.2（80% 区间）
lower = means[int(0.1 * 10)]     = means[1]
upper = means[int(0.9 * 10) - 1] = means[8]
```

本报告的 CI 是 `[1.00, 1.00]`：24 个"可答且已作答"样本全对，任何重采样都全对，区间退化成一个点。这不是"特别稳"，而是"退化"——样本太少，不要用它做显著性论证。

复现（依赖已装好的环境；`reports/` 为空时先跑一次）：

```bash
python -m visioncite.eval.run --mode layout --per 2
python -c "from visioncite.eval.bootstrap import bootstrap_mean_ci; print(bootstrap_mean_ci([0,1,1,0,1], n_boot=200, seed=0))"
# (0.2, 1.0)
```

### ⑤ 自测 3 问

1. **为什么 `accuracy` 只在"可答且已作答"的样本上算？**
   要点：弃答是"没答"，不是"答错"；混在一起会让"保守弃答"的系统显得准确率很高。必须同时报 coverage、abstain_accuracy，读者才能看到"少答换准确"的取舍。

2. **`accuracy` 的 95% CI 是 `[1.00, 1.00]`，说明什么？**
   要点：24 个"可答且已作答"样本全对，bootstrap 每次重采样都全对，区间退化；它说明样本量不足以支撑"差异显著"这类结论，应扩大语料或改用配对比较，而不是把 CI 当成"稳"。

3. **配对 bootstrap 与"两条独立 CI 不重叠"相比，好在哪？**
   要点：配对利用同一样本上的差值 `a[i]-b[i]`，消掉"样本难度"这个共同方差，区间更窄、检验更有力；两条独立 CI 不重叠是更保守也更粗糙的判据，且样本量不同时不可比。

---

## 附录 A 十二讲与文件的对照表

| 讲 | 必读 | 建议同时读 |
|---|---|---|
| 1 | `pipeline.py`、`types.py`、`config.py` | `errors.py`、`logging.py` |
| 2 | `layout/binarize.py` | `tests/test_layout.py` 前 5 个用例 |
| 3 | `layout/xycut.py` | `CutConfig`、`regions.build_regions` |
| 4 | `layout/table.py`、`layout/regions.py` | `eval/templates.py::table` |
| 5 | `ocr/base.py`、`ocr/factory.py` | `ocr/rapidocr_backend.py`、`pipeline._analyze_page` |
| 6 | `index/tokenize.py`、`index/chunking.py` | `tests/test_index.py` 前 4 个用例 |
| 7 | `index/bm25.py` | `index/retrieve.py` |
| 8 | `index/visual.py` | `pipeline._encode_chunks` |
| 9 | `index/fuse.py`、`index/retrieve.py` | `config.rrf_k` / `mmr_lambda` |
| 10 | `answer/verify.py` | `answer/qa._build_citations`、`eval/metrics.citation_hit` |
| 11 | `answer/extractive.py`、`answer/abstain.py` | `eval/metrics.expected_calibration_error` |
| 12 | `eval/run.py`、`eval/metrics.py` | `eval/bootstrap.py`、`eval/corpus_gen.py` |

## 附录 B 继续往下读时要注意的已知不一致

读源码时你可能会撞上下面这些点，它们不是你的理解问题：

1. `pyproject.toml` 的 `[project.scripts]` 声明了 `visioncite = "visioncite.cli:main"`，但源码里没有 `cli.py`；安装后 `visioncite` 命令不可用，请用 `python -m visioncite.eval.run`。
2. `pyproject.toml` 的 `readme = "README.md"` 指向的文件不在仓库里。做可编辑安装（`pip install -e .`）时如果因缺 readme 失败，先补一个 README，或者直接用 `PYTHONPATH=src` 跑模块（本文档所有命令都只依赖 `PYTHONPATH=src`）。
3. `errors.py` 里的 `OcrUnavailableError` 与 `VerificationError` 已定义但当前没有被抛出/捕获；降级靠返回值与 `Null*` 实现。
4. `ingest/image.py` 的 `perceptual_hash()` / `hash_distance()` 与 `ingest/pdf.py` 的 `count_pages()` 已实现但未接入主链路（练习 8 会把 dHash 接进上传去重，并在真实语料上量出它为什么会误判）。
5. `index/retrieve.py` 对 `pipeline.Document` 的类型依赖让"只导入 answer 层"也要装 `pypdfium2`。
6. `api/app.py` 的路由注册拆成了 `_register_health_routes` / `_register_document_routes` / `_register_asset_routes` / `_register_ui_routes` 四组，上传的落盘与解析在 `_store_document()` 里；读的时候别在 `upload()` 里找去重逻辑，它只有大小校验一行。

> **v0.2 修订**：上面第 1、2 条已修复（`cli.py` 与 `README.md` 都已存在），并新增 `agent/`、`tools/`、`llm/`、`mcp/` 四层，见下方的第 13 讲。

---

## 第 13 讲（v0.2 增补）工具调用、Agent 与 MCP

### ① 这一讲解决什么问题

前 12 讲的系统是"固定流水线"：检索 → 抽句 → 校验。它只能回答"文档里哪句话包含答案"。
真实问题常常需要**多步**（先查目录、再翻到某页）或**非文本能力**（这段字什么颜色）。
这一讲把系统能力包装成工具，让模型自己决定调什么，并把"模型给的答案"重新拉回证据上。

### ② 核心概念与公式

Agent 循环（`agent/loop.py`）：

```text
messages = [system, user]
for step in 1..max_steps:
    resp = provider.chat(messages, tools=registry.specs())
    if resp 没有 tool_calls:  → 拿到最终答案，跳出
    messages += [assistant(tool_calls)]
    for call in tool_calls:
        result = registry.call(call.name, call.arguments)   # JSON 字符串
        messages += [tool(name=call.name, tool_call_id=call.id, content=result)]
        记录 evidence_ids += result.results[].chunk_id
引用 = evidence_ids 映射成的 Citation（只取工具真正返回过的 id）
校验 = verify_answer(answer, citations, chunks, min_support=0.35)
```

工具分发（`tools/base.py`）：

```text
call(name, args):
    spec = tools[name]                  # 未知工具 → {"error": ...}
    missing = required - keys(args)     # 缺参 → {"error": ...}
    return json(handler(args))          # 抛异常 → {"error": "Type: msg"}
```

MCP（stdio + JSON-RPC 2.0）：

```text
请求：{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":..., "arguments":{...}}}
响应：{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"<json>"}],"isError":false}}
通知：无 id → 不响应
错误码：-32700 解析失败 / -32601 方法不存在 / -32602 参数错
```

### ③ 对应代码

- `src/visioncite/llm/base.py` — `ChatMessage`、`ToolCall`、`Usage`、`LLMProvider`。
- `src/visioncite/llm/fake.py` — `FakeProvider`、`tool_call_response()`、`final_response()`、`demo_script()`。
- `src/visioncite/llm/openai_compat.py` — `OpenAICompatProvider.chat/_parse`、`_to_wire()`、`_parse_call()`。
- `src/visioncite/tools/base.py` — `ToolSpec`、`ToolRegistry.register/specs/call`、`_missing_required()`。
- `src/visioncite/tools/document_tools.py` — `build_document_tools()` 与四个工具。
- `src/visioncite/agent/loop.py` — `run_agent()`、`_evidence_from()`、`_finalize()`、`_fallback()`。
- `src/visioncite/mcp/protocol.py` / `server.py` / `client.py` — 协议、服务端分发、客户端与子进程。

### ④ 一个手算小例子

一次问答的工具轨迹（fake provider，`--fake`）：

```text
step 1: search_document({"query": "合计金额", "top_k": 3})
        → results[0] = {chunk_id: "edab...:p0-r11:0", page: 1, score: 1.0, text: "合计金额：1,286.40元"}
step 2: 模型给出 "合计金额：1,286.40 元。"
引用  = [p0-r11:0, p0-r12:0, p0-r1:0]   （按工具返回顺序取前 3 条）
校验  = numbers_in(answer)=["1286.40"] ⊆ numbers_in(evidence) ✓
        lexical_support(answer, evidence) ≥ 0.35 ✓
结果  = answer 非空、abstained=false、tokens=446
```

MCP 手算（进程内）：

```text
client.initialize()      → serverInfo.name = "visioncite"
client.list_tools()      → ["dominant_color", "get_region", "list_pages", "search_document"]
client.call_tool("search_document", {"query": "金额"})
                         → content[0].text 里含 "chunk_id"
```

复现：

```bash
visioncite agent corpus/images/invoice-0.png "合计金额是多少？" --fake
python -m visioncite.mcp list corpus/images/invoice-0.png
python -m visioncite.mcp call corpus/images/invoice-0.png search_document '{"query":"金额"}'
```

### ⑤ 自测 3 问

1. **模型能不能自己决定引用哪一块？**
   要点：只能从工具返回的 `chunk_id` 里选（`_finalize` 只映射这些 id），不能凭空写；这是"可验证引用"的关键。

2. **为什么 fake provider 不是"假功能"？**
   要点：它验证编排（工具分发、消息拼接、终止、降级），真实模型接上走同一条路径；CI 因此不需要 API key 也能覆盖 Agent 逻辑。

3. **MCP 为什么只做 tools？**
   要点：项目只需要"工具发现 + 调用"；resources/prompts 没有场景，硬做只会增加未验证代码，已写进 LIMITATIONS。
