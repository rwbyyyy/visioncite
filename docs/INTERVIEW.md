# VisionCite 面试深挖手册

用法：每题先背"一句话回答"，再按"展开"讲两到三层，最后主动扔一个"坑"给面试官。
所有行号对应当前仓库代码，引用格式为 `路径:起始行-结束行`。

## 0. 事实基线（先对齐口径，再谈亮点）

仓库里只有一次真实评测：`reports/eval-20261106-134357.md`（与之配套的 `.json` 含 36 条逐样本记录）。

| 项 | 值 | 口径 |
|---|---|---|
| 语料 | 36 题 / 8 张图 | 程序合成，4 类模板（invoice / table / chart / slide）× 2 份 |
| 可答 / 不可答 | 28 / 8 | `answerable` 字段 |
| 模式 | layout（仅此一行） | 区域 + BM25，`NullVisualEncoder`，无 VLM |
| 答题率 coverage | 0.72（26/36） | 未弃答样本 / 全部样本 |
| AnswerAcc | 1.00（24/24） | 仅"可答且实际作答"的 24 条 |
| 95% CI | [1.00, 1.00] | bootstrap 1000 次，样本全部为 1，区间退化 |
| F1 | 0.40 | 字 bigram token 级 F1，同样本集 |
| Cite-P / Cite-R | 0.83 / 1.00 | 引用框与 gold 框 IoU ≥ 0.5 |
| 弃答判断准确率 | 0.83（30/36） | 可答必答 + 不可答必弃答 |
| ECE | 0.19 | 10 等宽箱，"正确" = AnswerAcc=1 且未弃答 |
| 延迟 | p50 0.30 ms / p95 0.64 ms | 检索+抽取+校验，不含 OCR 与建索引；CPU-only |

三个必须自己先说出口的事实：

1. 4 道可答题被弃答（table 的"合计金额"×2、slide 的"ECE"×2），答题率因此只有 0.72。
2. 2 道不可答题被误答（table 的"哪个产品退货最多"×2，置信度 0.44）。
3. 报告只有 layout 一种模式，`page` 与 `hybrid` 的代码路径存在但这次没跑；CLIP 增益不能引用。

---

### Q1. XY-cut 递归分割是怎么做的？复杂度怎么估？

**一句话回答**：在墨迹矩阵上交替做行/列投影，投影为 0 的空白带就切，切不动换方向再试，递归到深度上限；每层总代价不超过页面像素数，所以是 O(D·N)。

**展开**：
- 入口 `xy_cut(mask)` 拿整页 `BBox(0,0,W,H)` 进递归，`axis=0` 表示先按行投影（横切）。
- 每层先算 `piece.sum(axis=1)` 得到每行墨迹数，`profile > 0` 视为有内容，`_true_runs` 用"首尾补 False + 差分"找连续区间，间隔小于 `min_gap`（12 px）的段合并——这一条让行距不被当成栏间距。
- 切出多段就分别递归，方向翻转（`1-axis`）；**当前方向切不动时换另一轴再试一次**，这是多栏文档能切开的关键。
- 三个终止条件：深度到 `max_depth`（6）、边长小于 `min_size`（24）、或两轴都切不动 → 成为叶子；叶子再按墨迹占比 `_MIN_INK=0.002` 丢掉噪点。
- 复杂度：同一层的子区域互不重叠，每层总代价 ≤ W·H，最坏 O(D·W·H)。1000×1300 的合成页、D=6，实测毫秒级。
- 为什么不用深度学习版面模型：无参数、CPU 可跑、每一刀能画出来解释；代价是只能处理"有留白"的规则排版。

**代码位置**：`src/visioncite/layout/xycut.py:46-102`（主流程与换轴重试），`src/visioncite/layout/xycut.py:122-153`（投影+合并空白带），`src/visioncite/layout/xycut.py:156-172`（差分找连续区间），`src/visioncite/layout/xycut.py:32-43`（CutConfig 三个参数）

**常见坑 / 追问**：
- 坑：`_split_segments` 返回长度为 1 时也会进入"能切"分支吗？不会，`len(segments) > 1` 才切，否则继续换轴，最后落叶子。
- 追问"会不会切太碎"：`min_size` + `_drop_tiny(min_area=900)` + `merge_overlapping(IoU≥0.3)` 三层兜底（`regions.py:226-273`）。
- 追问"倾斜 1 度会怎样"：投影不再有整行为 0 的空白带，切不动，退化成一整块——这是已知失效，写在 `xycut.py:9-13` 的注释里。

### Q2. Otsu 阈值怎么推导？为什么用累积和实现？

**一句话回答**：遍历 256 个候选阈值，选"前景/背景两类之间方差最大"的那个；类间方差可写成 `(μ_T·w0 - μ0)² / (w0(1-w0))`，用 `cumsum` 一次算完，避免 256 次循环。

**展开**：
- 设阈值 t 把像素分成 `≤t`（背景，比例 w0、均值 μ0）和 `>t`（前景，w1、μ1），类间方差 `σ_b² = w0·w1·(μ0-μ1)²`。
- 恒等式 `μ_T = w0·μ0 + w1·μ1`，代入后等价于 `(μ_T·w0 - μ0)² / (w0·(1-w0))`——少了两次均值差，数值更稳。
- 实现用直方图：`prob = hist/total`，`omega = cumsum(prob)`，`mu = cumsum(prob*levels)`，然后对 256 个点求 `argmax`，整体 O(256 + 像素数)。
- 分母 `w0(1-w0)` 在 `w0=0` 或 `w0=1` 时为 0（典型是 t=255，全黑图在 t=0 也会发生），代码用 `np.errstate` 静默并把方差置 0，避免除零告警。
- 灰度化用 BT.601 的 `0.299R + 0.587G + 0.114B`，权重差异来自人眼对绿色最敏感。

**代码位置**：`src/visioncite/layout/binarize.py:36-68`（推导与累积和实现），`src/visioncite/layout/binarize.py:19-33`（灰度化），`src/visioncite/layout/binarize.py:71-82`（`gray < cut` 生成墨迹）

**常见坑 / 追问**：
- 坑：`ink_mask` 用的是**严格小于**，所以全白页 Otsu 返回 0 时 `gray < 0` 全 False，正好得到"无墨迹"；但全黑图同理也会被判成无墨迹，这是 Otsu 双峰假设的直接后果。
- 追问"光照不均"：Otsu 是全局阈值，手机拍照有阴影就崩，应换自适应阈值（`binarize.py:7-8` 自己标注了这个局限）。
- 追问"能不能用大津法以外"：可以，但会引入参数；这里要的是"无参数 + 可解释"。

### Q3. 表格线检测怎么做的？哪些场景会失效？

**一句话回答**：把"整行/整列墨迹占比超过 25%"的细带当线，横线 ≥3 且竖线 ≥2 才判为表格；无线表、三线表、倾斜扫描、多张小表都会失效。

**展开**：
- `rule_bands(mask, axis)` 求按行或按列的均值，`>= min_ratio`（0.25）的视为线带，再取连续区间；厚度超过 12 px 的带被过滤掉——否则整块深色图片会被当成一条巨粗的线。
- `find_tables` 要求横线 ≥3、竖线 ≥2，且外接框至少 40×30 px，否则返回空。
- 判定只用于**给区域打 type 标签**（`classify` 里 overlap/area > 0.5 → `table`），不负责还原单元格结构；`rows/cols` 只是"线数-1"的展示用估计。
- 失效清单（代码注释里主动写了）：无线表/三线表（横线不足 3 条，退化成 text）、多张小表被包成一个大框、倾斜 1~2 度线断成多段、跨页表；反向风险也存在——密排文字行的墨迹占比可能超过 25%，被误判成线。

**代码位置**：`src/visioncite/layout/table.py:41-60`（线带提取与厚度过滤），`src/visioncite/layout/table.py:63-84`（阈值与单候选返回），`src/visioncite/layout/regions.py:118-121`（重叠 >0.5 判 table）

**常见坑 / 追问**：
- 坑：`find_tables` 永远最多返回 1 个候选（`return [TableCandidate(...)]`），所以"一页多表"必然合并。
- 追问"三线表怎么办"：现在检测不到，区域会被当 text，文本仍能检索（因为 OCR 有字），只是表格结构信息丢了——引用和答案不会错，但跨列推理做不了。
- 追问"为什么不直接用 PaddleOCR 的表格模型"：依赖体积换来的收益在这个语料上不成立；更重要的是当前实现每一步都能解释。

### Q4. 中文为什么用字 bigram 而不是分词？

**一句话回答**：去掉词典依赖，且对未登录词、专有名词和 OCR 错字更稳；代价是索引膨胀到约 1 倍字数，并产生"计金"这类无意义 bigram。

**展开**：
- `tokenize` 先用一条正则把文本切成"连续中文 / 连续英文 / 数字"三类；中文片段按 2 字滑窗，长度 n 的片段产生 n-1 个 token，单字保留。
- 举例：`合计金额` → `合计 / 计金 / 金额`；查询侧同样切，所以只要 2 字重合就能命中，不依赖分词器把"总金额"切对。
- 英文和数字分开切是因为 OCR 常把 `ECE0.08` 粘成一个 token，混切会让 `ECE` 和 `0.08` 都检索不到（`tokenize.py:23-26` 注释写了这一点）。
- 归一化顺序：NFKC（全角转半角）→ 去千分位逗号 → 小写 → 压缩空白。`1,286.40` 与 `1286.40` 因此等价。
- 为什么不用 jieba：多一个词典依赖、冷启动与内存成本，且短问题 vs 短块的场景下 bigram 已经够；虚词不建停用词表，靠 BM25 的 IDF 自动降权（"的"出现在几乎每个块 → IDF 接近 0）。

**代码位置**：`src/visioncite/index/tokenize.py:22-27`（正则与千分位），`src/visioncite/index/tokenize.py:30-44`（归一化），`src/visioncite/index/tokenize.py:82-93`（bigram 生成）

**常见坑 / 追问**：
- 坑：`query_terms` 会丢掉单个中文字（`answer/extractive.py:75`），所以"几月"这种查询在作答阶段可能 0 候选——索引层保留单字、作答层过滤单字，两层口径不一致。
- 追问"为什么是 2 不是 3"：3-gram 更稀疏，短查询容易一个都命中不了；2-gram 是覆盖率与区分度的折中，且不需要调参。
- 追问"索引多大"：token 数约等于字符数，倒排是 `词 → {块下标: tf}`，没有位置信息，所以不支持短语查询——这是明确的取舍。

### Q5. BM25 公式逐项讲一下，IDF 为什么用 ln(1+…)?

**一句话回答**：`Σ IDF(t)·tf(k1+1)/(tf + k1(1-b+b·|d|/avgdl))`；tf 饱和控制"出现 10 次不等于重要 10 倍"，b 压制长块碰巧命中，`ln(1+…)` 保证 IDF 非负。

**展开**：
- IDF：`ln(1 + (N - df + 0.5)/(df + 0.5))`。经典写法 `ln((N-df+0.5)/(df+0.5))` 在 df 接近 N 时会变成负数，加上 1 之后下界是 0（`_idf` 里 df=N 时返回 ln(1+0.5/(N+0.5))，恒正）。
- df 取 `len(posting)`，即"包含该词的块数"；N 取 `len(chunk_ids)`，即块数。分母是块粒度，不是文档粒度——这点被问过。
- tf 项：`tf(k1+1)/(tf + k1·length_norm)`，k1=1.5，tf 从 1 涨到 10 时分数的增量迅速衰减；k1 越大饱和越慢。
- 长度归一：`length_norm = 1 - b + b·(doc_length/avgdl)`，b=0.75。avgdl 在每次 search 时现算（`sum(doc_lengths)/size`），没有缓存。
- 查询侧：`for term in set(terms)`，同一查询词只累加一次 IDF，**不做 query tf 饱和**（查询是一句话，重复词没有信息量）。
- 为什么不用 TF-IDF：没有长度归一和 tf 饱和，长块与高频词会主导排序；BM25 是稀疏检索的工业默认值。

**代码位置**：`src/visioncite/index/bm25.py:1-17`（公式与逐项解释），`src/visioncite/index/bm25.py:95-104`（IDF 平滑），`src/visioncite/index/bm25.py:106-117`（tf 与长度归一），`src/visioncite/index/bm25.py:78-93`（查询侧去重与打分）

**常见坑 / 追问**：
- 坑：`avgdl = sum(...)/self.size or 1.0`——`or` 只在商为 0 时兜底，而 size>0 时除法已经先执行了；这里没有真正的除零保护（靠 `size == 0` 提前 return 兜住）。
- 追问"k1/b 怎么定的"：用经典默认值 1.5/0.75，没有在自建语料上调（36 题调参只会过拟合）；`config.py` 里给了名字，可被改。
- 追问"BM25 分数有上界吗"：没有，这正是后面要用 RRF 按名次融合的原因。

### Q6. 双路召回为什么用 RRF，而不是把分数加权相加？

**一句话回答**：BM25 分数无上界、余弦在 [-1,1]，两者尺度不可比；加权重和要先归一化，而归一化对离群值敏感；RRF 只用名次，零训练参数。

**展开**：
- `score(d) = Σ_r w_r / (k + rank_r(d) + 1)`，k=60。名次从 0 开始，所以第一名得 1/61。
- 好处一：对分数尺度免疫。BM25 给 42.7、CLIP 给 0.31，直接相加等于让 BM25 独占；RRF 只看"排第几"。
- 好处二：名次比分数稳定。分数会随查询长度、语料规模漂移，名次不会。
- 好处三：没有可训练参数，不需要为融合权重准备标注数据。
- 代价：丢掉了分数间隔信息。"0.9 与 0.1"和"0.51 与 0.49"在 RRF 眼里完全一样。
- 代码里 `weights` 参数存在但评测中始终等权；融合后还会除以 top-1 分数做展示归一化（见 Q21）。

**代码位置**：`src/visioncite/index/fuse.py:24-46`（RRF 实现），`src/visioncite/index/retrieve.py:51-59`（两路召回与融合），`src/visioncite/index/retrieve.py:73-98`（视觉一路）

**常见坑 / 追问**：
- 坑：k 越大越平权，k→0 时第一名的优势被放大到极端；60 是论文经验值，不是最优值。
- 追问"能不能学权重"：能，但需要标注的 relevance，36 题撑不起来；这里是刻意的"无参数优先"。
- 追问"两路都空怎么办"：`retrieve` 直接返回空列表，`answer_question` 据此弃答（`answer/qa.py:56-57`）。

### Q7. MMR 怎么去冗余？λ=0.7 意味着什么？

**一句话回答**：每轮选 `λ·sim(q,d) - (1-λ)·max sim(d, 已选)` 最大的候选；λ=0.7 表示七成看相关性、三成罚重复。

**展开**：
- 公式里第一项是相关性，第二项是"与已选证据的最大相似度"，用负号惩罚。
- 第一轮没有已选，退化成纯相关性；之后每选一个，都要对剩余候选重算一次冗余项，整体 O(k²·n) 次余弦。
- 为什么需要：只按相关性排，同一段的上下两个块会一起进 top-k，用户看到的是三条几乎一样的证据。
- 在本项目里还有一个隐性效果：同一 region 被切成多个 chunk 时，这些 chunk 共享**同一个 CLIP 向量**（编码时按 region 去重，`pipeline.py:220-230`），冗余度恒为 1，会被 MMR 直接压掉——这既是去冗余，也是"同一区域的重复引用"被消掉的原因。
- 只在 `document.vectors is not None` 时启用；没有向量就退化成截断到 top_k（`retrieve.py:117-118`），所以 layout 模式其实没跑 MMR。

**代码位置**：`src/visioncite/index/fuse.py:65-95`（选择循环），`src/visioncite/index/fuse.py:98-120`（MMR 打分），`src/visioncite/index/retrieve.py:101-137`（接入与降级），`src/visioncite/pipeline.py:220-232`（按 region 去重编码）

**常见坑 / 追问**：
- 坑：相关性和冗余项来自不同空间（RRF 融合分 vs CLIP 余弦），量纲不同却直接线性组合，这是个工程启发式，不是有理论保证的公式。
- 追问"λ 调小会怎样"：更分散，但可能选到不相关的证据；`config.py:83` 里是 0.7，没有做 λ 扫描。
- 追问"复杂度能不能降"：预计算成对相似度矩阵，或用聚类先分组。

### Q8. CLIP 的对比学习和 InfoNCE 用直觉讲一下。

**一句话回答**：一个 batch 里 N 对（图, 文），对角线是正例、其余 N²-N 是负例，损失就是"把正例的余弦相似度推高、负例压低"的 softmax 交叉熵；温度 τ 控制分布的尖锐程度。

**展开**：
- 对称 InfoNCE：`L = -1/N Σ_i log[ exp(sim(i,i)/τ) / Σ_j exp(sim(i,j)/τ) ]`，图→文、文→图各算一次再平均。
- 关键点：**负例来自 batch 内其他样本**，不需要人工构造，所以 batch 越大负例越多、效果越好（这也是 CLIP 用超大 batch 的原因之一）。
- 训练完，图和"描述它的那句话"落在同一空间，于是可以用文字直接检索图片（zero-shot），不需要为每类图片训练分类头。
- 工程上：`clip-vit-base-patch32`，投影维度 512；`encode_images` 与 `encode_texts` 都做 L2 归一化，归一化后余弦相似度等于点积，检索就是一次矩阵乘。

**代码位置**：`src/visioncite/index/visual.py:1-19`（原理与失效说明），`src/visioncite/index/visual.py:90-147`（图/文编码），`src/visioncite/index/visual.py:203-214`（L2 归一化），`src/visioncite/index/retrieve.py:96-98`（矩阵乘打分）

**常见坑 / 追问**：
- 坑：`available()` 只检查 torch/transformers 是否装好，**不检查权重**；真正的下载发生在第一次 `load()`，失败会把 `_load_failed` 置位并永久降级。
- 追问"τ 是什么"：可学习的温度参数，训练时一起优化；推理时不需要它。
- 追问"为什么 batch 是 8"：CPU 上显存/内存与速度的折中，写死在构造函数默认值里。

### Q9. CLIP 在文档小字上为什么失效？那为什么还留着它？

**一句话回答**：patch32 每个 token 只看 32×32 像素，224 输入下小字被缩糊；文本侧最多 77 token，和长 OCR 文本也不匹配。它只做"视觉相近"的召回一路，不做答案依据。

**展开**：
- 分辨率：`clip-vit-base-patch32` 的输入是 224×224，切成 7×7 个 patch。一页 1000×1300 里 26 px 高的字行，缩到 224 后只剩约 4 px，笔画糊在一起。
- 文本侧：CLIP 的 text encoder 是短句训练出来的（char-level BPE，77 token 上限），对"这张发票的金额是多少"这类问法和 OCR 出来的长串文本的匹配，远弱于 BM25 的精确词面命中。
- 训练分布：CLIP 学的是自然图像 + alt text，表格、票面、公式属于分布外。
- 所以定位是：CLIP 负责"这张图大概是什么"（图表、流程图、版式相似的页面），BM25 负责"金额、合同编号"这类精确串；两者盲区互补。
- 代码里的保护：`_visual_search` 失败只记日志返回空，不影响文本检索；编码异常时 `_encode_chunks` 返回 None，整篇文档退化成纯文本索引。

**代码位置**：`src/visioncite/index/visual.py:11-19`（主动性声明失效场景），`src/visioncite/index/retrieve.py:88-95`（异常隔离），`src/visioncite/pipeline.py:216-237`（编码失败的兜底）

**常见坑 / 追问**：
- 坑：区域裁图是 `bbox.pad(4)`，一行字的裁图再 resize 到 224，等于把 300×30 的图拉成方形，形变本身也伤语义。
- 追问"怎么改"：区域裁图放大后再编码、多尺度裁图、或者干脆放弃 CLIP 改用 OCR 文本；要走视觉路线就得换 LayoutLMv3 / Donut 这类文档预训练模型。
- 追问"CLIP 有什么用"：在当前语料上没验证（报告只有 layout 模式），这是必须诚实说的一点。

### Q10. 引用校验有哪三条规则？为什么是这三条？

**一句话回答**：引用必须存在、答案里的数字必须在被引证据里逐字出现、答案 token 被证据覆盖的比例 ≥ 0.5；分别挡"编 id""编数字""答非所问"。

**展开**：
- 规则 1 `citation_ids_exist`：把模型/抽取给出的 chunk_id 与真实块集合求交，防伪造引用。
- 规则 2 `numbers_supported`：先归一化（去货币符号、去千分位、去空格），再做集合包含。`1,286.40` 与 `1286.40` 视为同一个数；只要有一个数字不在证据里就失败。数字幻觉（金额、日期、比例）是最危险的一类错误，一条规则挡住大部分。
- 规则 3 `lexical_support`：答案 token 里出现在证据 token 中的比例，默认下限 0.5；纯数字 token 先剔除（否则规则 2 已经管过的东西重复计分）。
- 三条的分工：存在性管"引用是否有效"，数字管"事实是否落地"，词面管"语义是否跑题"。它们都不能证明答案正确，但能把"看起来对"筛成"有据可查"。
- 对 extractive 路径几乎总通过（答案本来就是摘录的）；真正的价值在 VLM 路径——模型越自由，越需要这层网。

**代码位置**：`src/visioncite/answer/verify.py:110-148`（三条规则汇总），`src/visioncite/answer/verify.py:43-73`（数字抽取与比对），`src/visioncite/answer/verify.py:76-93`（词面支撑），`src/visioncite/answer/verify.py:96-107`（引用存在性）

**常见坑 / 追问**：
- 坑：`checks` 字典是靠"reason 字符串里有没有'引用'/'数字'"反推出来的（`verify.py:142-147`），字符串一改就静默错位；正确做法是直接从布尔量组装。
- 追问"0.5 怎么定"：经验值，可调；调高会误杀改写过的正确回答，调低会放过跑题回答。
- 追问"数字比对会不会误判"：会。`2026` 出现在日期里、`2` 出现在"2 台"里，都被当成"有出处"；反过来 `1,286.40元` 若 OCR 成 `1286.40 元` 也能通过归一化——这正是我们想要的双向容忍。

### Q11. 弃答阈值 0.35 怎么用？ECE 在这里说明什么？

**一句话回答**：三档顺序判断（无证据 → 校验失败 → 置信度 < 0.35），任一命中就弃答；ECE 用来检查"置信度说得准不准"，当前 0.19，说明这套自造置信度只是粗略排序信号。

**展开**：
- 顺序即优先级：没有证据时硬答一定是编，所以第一条不看置信度；有证据但校验不过，说明证据没支撑住答案；最后才是置信度低。
- 阈值来自 `Settings.abstain_threshold`，可以从环境变量覆盖；注释里说评测脚本会扫多个阈值画"答题率-准确率"曲线，但当前 `eval/run.py` 只跑了单一阈值，这条曲线还没实现。
- ECE 定义：把样本按置信度分 10 个等宽箱，`ECE = Σ_b (n_b/N)·|acc_b - conf_b|`；空箱跳过。0 表示完美校准。
- 当前 ECE=0.19 主要来自两个方向：置信度为 0 但弃答判断正确的样本（acc 高、conf 0）把误差拉高；少数错答集中在 0.44 附近。**这个数只能当量级参考**：36 个样本，单个样本最多能移动 ECE 约 0.028。
- 为什么不用"模型概率"：extractive 路径没有模型，没有 logprob；置信度是人为构造的代理（见 Q22），ECE 就是用来检验这个代理靠不靠谱。

**代码位置**：`src/visioncite/answer/abstain.py:18-44`（三档判断），`src/visioncite/config.py:84`（默认 0.35），`src/visioncite/eval/metrics.py:138-173`（ECE），`src/visioncite/eval/metrics.py:303-310`（汇总时"正确"的定义）

**常见坑 / 追问**：
- 坑：弃答样本的置信度并不总是 0——低置信路径弃答时会带着真实置信度（例如 0.22），这些样本进入 ECE 计算，会让"弃答"与"答错"在校准上同等对待。
- 追问"阈值调到 0.2 会怎样"：答题率上升，准确率大概率下降；没有跑曲线之前，任何数字都是猜的——这一点我会直接承认。
- 追问"业务上怎么选"：按错误成本定。金额类场景宁可少答，所以 0.35 是偏保守的。

### Q12. bootstrap 置信区间怎么算？为什么这次是 [1.00, 1.00]？

**一句话回答**：有放回重采样 B=1000 次，每次算均值，取 2.5% 与 97.5% 分位；本次 24 个样本全对，每次重采样都是 1.0，区间必然退化。

**展开**：
- 实现只有十几行：`rng = random.Random(seed)`，每次抽 `len(values)` 个下标（有放回），算均值，排序后取 `means[int(α/2·B)]` 与 `means[int((1-α/2)·B)-1]`。
- 为什么不用正态近似：n 小、指标有界在 [0,1]、比例接近 1 时正态区间会越界（上界超过 1）；bootstrap 不假设分布。
- 配对版本 `paired_bootstrap_ci`：先算逐样本差值，再对差值做同样的重采样；区间不含 0 才敢说"有差异"。这是消融实验做显著性判断的工具。
- 本次 CI 退化不是 bug，是"满分 + 小样本"的必然结果——它反过来暴露了一个更重要的问题：这个 n 下 CI 没有分辨力，报 [1.00, 1.00] 等于没报。
- seed 固定，重跑结果一致；`n_boot` 默认 1000，单次毫秒级。

**代码位置**：`src/visioncite/eval/bootstrap.py:19-47`（均值 CI），`src/visioncite/eval/bootstrap.py:50-78`（配对检验），`src/visioncite/eval/run.py:142-144`（评测里只对 AnswerAcc 求 CI）

**常见坑 / 追问**：
- 坑：分位点用整数下标截断，B 小的时候上下界会偏；1000 次够用，100 次就明显抖。
- 追问"为什么只对 accuracy 求 CI"：代码里确实只对 accuracy 做了；F1、Cite-P 的区间没算，属于没做完的部分。
- 追问"能不能用 t 检验"：可以，但要假设近似正态；bootstrap 更稳妥，也更容易在面试里手推。

### Q13. 消融实验是怎么设计的？实际跑了几组？

**一句话回答**：代码里定义了三种模式（page / layout / hybrid），控制同一语料、同一 settings、同一 OCR；但仓库里的报告只跑了 layout 一组，所以只能引用 layout 的数字。

**展开**：
- `MODES = ("page", "layout", "hybrid")`，`--mode all` 会依次跑三组，各出一行汇总 + 逐样本记录。
- `page`：把每页压成"整页一个区域"，去掉版面结构，用来回答"版面切分到底值多少"；实现是 `_flatten_to_page`，把区域内词拼成一整段文本后重建块与索引，并清空向量。
- `layout`：区域 + BM25，视觉编码器换成 `NullVisualEncoder`，MMR 退化成截断。
- `hybrid`：在 layout 基础上加 CLIP 视觉召回，两路 RRF 融合 + MMR 去冗余。
- 控制变量：语料固定（`load_or_generate` 已存在就复用，不重新渲染）、settings 固定、OCR 后端复用同一个实例、随机种子固定（bootstrap seed=0）。
- 显著性判断的准备：`paired_bootstrap_ci` 对同一批样本的逐样本得分做配对差值区间；区间不含 0 才认为差异不是噪声。

**代码位置**：`src/visioncite/eval/run.py:46`（模式定义），`src/visioncite/eval/run.py:49-77`（建索引），`src/visioncite/eval/run.py:210-233`（page 模式的降级实现），`src/visioncite/eval/run.py:268-277`（循环跑模式）

**常见坑 / 追问**：
- 坑（必须主动说）：`reports/eval-20261106-134357.md` 只有一行 layout，说明当时是 `--mode layout` 跑的；page/hybrid 的代码路径没有被这次报告覆盖。
- 追问"为什么不做 hybrid"：CLIP 要下载权重、CPU 上编码 8 张图 × 若干区域，成本高；这个环境的取舍是先保证可复现的文本链路。
- 追问"这算不算消融"：一半。代码具备消融结构，数字只覆盖其中一臂，所以简历里不能写"视觉检索带来 X 点提升"。

### Q14. 没有 OCR / 没有 CLIP / 没有 VLM 时分别怎么降级？

**一句话回答**：无 OCR 用 PDF 文本层（整页一个区域）；无 CLIP 走单路 BM25 + 截断；无 VLM 走 extractive；VLM 调用失败或引用无效时回退 extractive，三层都有日志与可观测状态。

**展开**：
- OCR：`create_ocr` 先看显式关闭（`VISIONCITE_OCR=off`），否则试 RapidOCR，`ImportError` 就返回 `NullOcrBackend`。`NullOcrBackend.recognize` 返回空列表，`_analyze_page` 发现"无词但有文本层"就把整页包成一个 `Region`——能答，但证据框覆盖整页，引用精度会掉。
- CLIP：`create_visual` 在 `visual_mode=off` 或 import 失败时返回 `NullVisualEncoder`；`_encode_chunks` 见 `available()` 为 False 直接返回 `None`；检索层 `document.vectors is None` 时只保留文本一路，MMR 退化为截断。
- VLM：`settings.vlm_enabled` 只在 base_url 与 model 都非空时为真；调用抛异常、答案为空、引用映射后为空，三种情况都回退 extractive（`qa.py:100-111`）。
- 可观测性：`/health` 返回实际的 ocr / visual 名字和 `vlm_enabled`，`build_document` 打完索引会 log 出 `vectors=on/off`。
- 设计原则：降级不是"少个功能"，而是"明确、可观测、不抛异常的能力开关"。

**代码位置**：`src/visioncite/ocr/factory.py:19-37`，`src/visioncite/index/visual.py:217-233`，`src/visioncite/answer/qa.py:100-111`，`src/visioncite/pipeline.py:173-178`，`src/visioncite/api/app.py:69-82`

**常见坑 / 追问**：
- 坑：`NullOcrBackend.available()` 返回 True（"空实现也是一种可用状态"），所以调用方无法用 `available()` 区分"没装 OCR"和"OCR 关了"，只能看 `name`。
- 追问"整页一个区域有什么问题"：引用框 = 整页，IoU 对任何 gold 框都容易 ≥0.5，Cite-R 会虚高；这也是我强调引用指标要看语料的原因。
- 追问"降级了用户知道吗"：前端会把 `OCR=xxx · 视觉=xxx` 打在上传结果里，不隐藏。

### Q15. FastAPI 这层怎么分层的？路由里为什么不写业务逻辑？

**一句话回答**：路由只做三件事——校验输入、调用用例、把异常翻译成 HTTP 状态码；业务全在 pipeline / answer，路由函数因此很薄，也方便用 TestClient 测。

**展开**：
- `create_app(settings)` 是工厂：注入配置、建目录、把 `documents` 字典挂到 `app.state`，然后注册路由；测试可以传不同 settings，不用改全局。
- 契约独立成 `api/schemas.py`：请求/响应都是 Pydantic 模型，FastAPI 自动生成 OpenAPI；`AskRequest` 用 `min_length=1, max_length=500` 把明显非法的请求挡在业务层之前。
- 错误映射：超过大小 → 413，类型不支持 → 415，解析失败 → 400，文档不存在 → 404；异常类型来自 `errors.py` 的小异常树，每个都带 `user_message`（中文，可直接展示给用户）。
- 幂等与去重：`doc_id` 是文件内容的 sha256 前 16 位，同一文件重复上传直接返回已有索引，不重复解析。
- 静态前端只有一个页面 + 一个 js + 一个 css，直接从 `web/` 目录用 `FileResponse` 返回，没有前端构建步骤。

**代码位置**：`src/visioncite/api/app.py:42-59`（工厂），`src/visioncite/api/app.py:84-108`（上传与去重），`src/visioncite/api/app.py:110-136`（问答），`src/visioncite/api/schemas.py:42-70`（请求/响应契约），`src/visioncite/errors.py:11-69`（异常树）

**常见坑 / 追问**：
- 坑：`documents` 是进程内字典，多副本或重启就丢索引；生产要换 Redis / 对象存储，代码注释里自己标了。
- 坑：`_bbox()`（`app.py:206-215`）是没被调用的死代码，只为了保留"未来扩展"。
- 追问"为什么路由用 def 而不是 async def"：`ask` 是同步的（内部是 CPU 计算），FastAPI 会把它丢到线程池，不阻塞事件循环；上传是 `async def`，因为要 `await file.read()`。
- 追问"并发安全"：`documents` 字典的读写没有加锁，Python 的 GIL 让单次赋值安全，但"检查是否存在 → 建索引 → 写入"不是原子操作，同文件并发上传会重复建索引。

### Q16. 测试是怎么做的？覆盖了哪些、没覆盖哪些？

**一句话回答**：81 个测试函数（一处 parametrize 展开成 2 例，合计 82 例），全离线——不联网、不下载模型、不依赖本机 OCR；缺真实 OCR 与 CLIP 的集成测试。

**展开**：
- 夹具的核心是一个 `FakeOcr`：直接返回预置的 `Word` 列表，于是版面、检索、作答、API 四层都能在"已知文本 + 已知坐标"上断言，不需要模型。
- `document_path` 用 PIL 画两个黑矩形模拟文字行，不依赖字体文件，跨平台可跑；`settings` 把缓存目录指到 `tmp_path`，不污染仓库。
- API 测试用 `TestClient` + `monkeypatch.setattr("visioncite.api.app.build_document", ...)` 替换重依赖，只验证路由、状态码与响应结构。
- 断言方式偏"性质"而非"快照"：IoU 相同框 =1、不相交 =0、半重叠 =1/3；Otsu 阈值落在两峰之间；`mmr_select` 在 λ=0.5 时不会同时选两个几乎相同的向量。
- 覆盖清单：BBox 几何（9）、版面（14）、索引/分词/BM25/RRF/MMR/切块（15）、作答/校验/弃答（14）、API（8）、端到端（7）、指标与 bootstrap（14）。
- 没覆盖：真实 RapidOCR 的识别结果、CLIP 编码、`eval/run.py` 的端到端回归（报告数字无法被测试自动复现）、VLM 客户端。

**代码位置**：`tests/conftest.py:27-82`（夹具与 FakeOcr），`tests/test_api.py:17-28`（TestClient 注入），`tests/test_layout.py:54-70`（XY-cut 性质断言），`tests/test_index.py:72-91`（RRF/MMR 行为），`tests/test_pipeline.py:33-49`（端到端断言）

**常见坑 / 追问**：
- 坑（必须主动说）：这台机器没装 pytest，测试数量是按 `^def test_` 静态统计的，**没有实际执行过**；`pyproject.toml` 里 `[tool.pytest.ini_options]` 已配好 testpaths。
- 追问"为什么不用真实 OCR 跑回归"：识别结果随模型版本漂移，做断言会脆；真实链路的稳定性靠 `reports/` 里的评测快照来盯。
- 追问"覆盖率"：`pyproject.toml` 配了 `branch = true` 与 coverage 报告，但没有提交覆盖率数字，不能引用。

### Q17. 如果重做一遍，你会改什么？

**一句话回答**：四件事——把答案粒度从"整句"改成"实体/数值"、把表格做成真正的单元格网格、把弃答阈值做成可解释的 risk-coverage 曲线、把评测从 36 题扩到可分层统计的规模。

**展开**：
1. 答案粒度：现在 AnswerAcc=1.00 而 EM=0.00、F1=0.40（`reports` 实测），因为 extractive 返回的是"包含答案的整句"。要么训练/约束一个抽取器只吐实体，要么让 VLM 输出结构化答案再做校验；否则指标永远在"对但不好看"的状态。
2. 表格：现在把行拼成 `a | b` 文本（`regions.py:127-143`），丢了列对齐、多级表头和合并单元格。应做单元格网格 + 行列坐标，答案可以引用到单元格而不是整张表。
3. 弃答：把阈值扫一遍画 risk-coverage 曲线，按业务错误成本选点；同时把已定义但没接线的 `min_citation_score` 接上，让"低分引用"真的被过滤。
4. 评测：36 题、4 类模板，n=24 时 CI 退化成 [1,1]；至少要扩到几百题、覆盖真实扫描件与倾斜/无线表，再做分层 bootstrap，并把 page/layout/hybrid 三臂真正跑完。
5. 工程债：`pyproject.toml` 里的 `visioncite = "visioncite.cli:main"` 指向不存在的 `cli.py`，`readme = "README.md"` 指向不存在的 README，代码注释里提到的 `docs/LIMITATIONS.md`、`docs/ARCHITECTURE.md`、`docs/EVALUATION.md` 也都不在仓库里——这些"说了没做"的地方要么补齐，要么删掉引用。

**代码位置**：`pyproject.toml:9`（readme）与 `pyproject.toml:34-35`（scripts），`src/visioncite/config.py:85`（未接线的 min_citation_score），`src/visioncite/ingest/image.py:102-135`（未接入主链路的 dHash），`src/visioncite/eval/templates.py:239`（语料里写死的"60 条样本，ECE 0.08"与报告不一致）

**常见坑 / 追问**：
- 坑：不要答"上更大的模型"。这个项目的瓶颈在评测规模与答案粒度，不在模型大小；模型换了，指标口径问题还在。
- 追问"最想改哪一个"：表格。它是唯一一个"文本对了但结构全丢"的环节，也是真实票据场景的核心。
- 追问"为什么当初不做"：时间盒 + CPU-only + 无标注数据，先把可解释的基线跑通再做重活。

### Q18. 为什么 bbox 用左闭右开？IoU ≥ 0.5 是哪来的约定？

**一句话回答**：左闭右开让面积直接等于 `(x1-x0)*(y1-y0)`，相邻块拼接不会重复计像素；IoU≥0.5 沿用 COCO 的 AP50 约定，让"引用命中"有可比的阈值。

**展开**：
- 坐标约定写在 `types.py` 顶部：原点左上、x 向右、y 向下、单位像素、`bbox=(x0,y0,x1,y1)` 左闭右开。
- `BBox.__post_init__` 会把传反的坐标自动交换（frozen dataclass 用 `object.__setattr__` 绕过），避免上游传反后在 IoU 里算出负面积。
- `iou = inter / (A + B - inter)`，两个空框返回 0；`intersection_area` 用 `min(x1)-max(x0)` 先判正再乘，避免负数乘积。
- 评测里 `citation_hit` 对每条引用与 gold 框算 IoU，任一条 ≥0.5 即命中；`citation_precision` 是命中引用数 / 全部引用数。
- 为什么用 IoU 而不是"中心点落在框内"：中心点判据对大小框不对称，IoU 对"框太大"和"框太小"都敏感。

**代码位置**：`src/visioncite/types.py:8-12`（坐标约定），`src/visioncite/types.py:42-53`（坐标规范化），`src/visioncite/types.py:85-101`（IoU），`src/visioncite/eval/metrics.py:105-135`（命中与精确率）

**常见坑 / 追问**：
- 坑：区域框会被 `tighten_bbox` 收紧到文字外接框再 `pad(4)`（`regions.py:67,77-98`），与渲染时记录的 gold 框天然有几像素偏差，0.5 的阈值是在容忍这个偏差。
- 坑：同一个 region 切出的多个 chunk **共享同一个 bbox**（`chunking.py:57`），所以引用两条同区域的 chunk 不会提高 Cite-R，只会拉低 Cite-P。
- 追问"能不能用 IoU 0.75"：可以，但在这个几像素偏差的量级下 0.75 会误杀；0.5 与 COCO 一致，便于对外解释。

### Q19. 区域分类和阅读顺序是怎么定的？哪里最容易错？

**一句话回答**：跟表格候选重叠超过一半 → table；有墨迹但没识别到字 → figure；其余 → text；阅读顺序按 y0 每 40 px 分带、带内按 x 排序。

**展开**：
- 分类刻意"可解释、可调试"：三条规则，没有模型，错了一眼能看出来。
- 表格优先：`inter / box.area > 0.5` 才判 table，避免表格旁边的一行文字被吞进去。
- figure 的判据是"没有词 + 墨迹密度 > 0.05"：图表、照片、印章都会落这里；密度 ≤0.05 的给 `unknown`，通常是噪点或极淡的线。
- 阅读顺序：`y0 // 40` 分带是硬编码经验值，带内按 x0 排；对"标题 + 两栏正文"这种版式够用。
- 装配链路：`build_regions` = 二值化 → 表格候选 → XY-cut → 合并重叠 → 丢小块 → 词归属 → 分类 → 拼文本 → 收紧框 → 排序。

**代码位置**：`src/visioncite/layout/regions.py:27-74`（装配主流程），`src/visioncite/layout/regions.py:101-124`（分类规则），`src/visioncite/layout/regions.py:290-305`（阅读顺序），`src/visioncite/layout/regions.py:206-223`（词归属：包含优先，否则最近中心）

**常见坑 / 追问**：
- 坑：40 px 分带是魔数，跨带的大块会乱序；多栏文档会被"按 x 排序"排成一行一行地串栏。
- 坑：`assign_words` 遇到"中心点不在任何块里"的词会给最近块，所以页边距的页码可能被塞进正文块，污染该块文本。
- 追问"为什么不用 XY-cut 的顺序"：XY-cut 的遍历顺序依赖切分路径，不稳定；按最终几何位置排序更可预测。

### Q20. 切块为什么是 240 字？不重叠会不会切断答案？

**一句话回答**：240 字是"保留上下文 vs 不稀释 BM25 信号"的经验折中；按句切不重叠，代价是跨块的答案会被切断，而且超长单句不会被硬切。

**展开**：
- 切句正则 `(?<=[。！？!?；;])|(?<=\n)`：中文句末标点后切，换行后也切——后者让表格的每一行天然成为独立句子。
- 贪心装箱：累加到超过 `max_chars`（240）就落一块；因为按句边界切，块长是"不超过 240 的句长累加"，可能略超。
- 为什么不用重叠窗口：上下文冗余会让同一段内容在 top-k 里出现多次，MMR 之后又会被压掉，等于白算；代价是答案正好跨在块边界时两头都不完整。
- 为什么是 240：中文 150~300 字一块是常见经验；再长会被 BM25 的 b 项压分（长块里只有一个词相关时，分数被平均稀释）。
- 块 id 是 `{doc_id}:{region_id}:{index}`，任何一条引用都能回查到页面和区域。

**代码位置**：`src/visioncite/index/chunking.py:19`（切句正则），`src/visioncite/index/chunking.py:35-62`（切块与 id 生成），`src/visioncite/index/chunking.py:65-85`（贪心装箱）

**常见坑 / 追问**：
- 坑：单句超过 240 字时不会硬切（`_pack` 在 `current` 为空时直接累加），表格如果被当成一整行，就会变成超大块——本报告的 table 样本正是"整张表一个块"。
- 坑：`chunk_region` 对所有块都赋 `region.bbox`，所以同一区域的多块在引用上完全无法区分。
- 追问"怎么改"：句级 + 语义边界（按段落/标题层级），加 10~20% 的滑动重叠，或者按"块内句子完整度"做动态规划。

### Q21. 检索分数归一化到 0~1 之后，还能当阈值用吗？

**一句话回答**：不能。归一化是除以 top-1 融合分，所以第一名恒为 1.0，分数只有相对意义；绝对阈值（配置里的 `min_citation_score`）在这条链路上是失效的。

**展开**：
- `raw / top`，`top = fused[0][1]`，再 `min(1.0, ...)` 截断；目的是把不同量级的分数压到同一个展示区间。
- 后果一：top-1 永远是 1.00，前端看起来"很自信"，但那是构造出来的。
- 后果二：第二名之后的分差被保留（相对比例），所以排序可用、阈值不可用。
- `Settings.min_citation_score = 0.02` 定义了、写进了注释，但全仓库没有任何地方读取它——这是个死配置，我在 Q17 里把它列为工程债。
- 如果真要做阈值，正确做法是保留原始 RRF 分（或对每路做分位归一化），再在同一分布上扫阈值。

**代码位置**：`src/visioncite/index/retrieve.py:62-69`（归一化），`src/visioncite/config.py:85`（未使用的阈值），`src/visioncite/index/fuse.py:24-46`（原始 RRF 分）

**常见坑 / 追问**：
- 坑：`dict(fused).get(chunk_id, 0.0)` 每次循环都重建一次字典，是 O(n) 的重复开销（小数据无感，大数据要改）。
- 追问"那 Cite-P 0.83 里的 0.5 是怎么来的"：`_cited_chunks` 会带"最佳块 + 另一个不同块的次优句"两条引用（`extractive.py:194-209`），第二条常常不命中 gold 框，于是精确率掉到 0.5。
- 追问"为什么不只引一条"：单条引用在"答案跨两块"时信息不足；这是可解释性与精确率的取舍，可以通过 `top_k`/引用上限做成配置。

### Q22. 置信度是怎么算的？为什么是 0.55/0.25/0.20？

**一句话回答**：`0.55·coverage + 0.25·margin + 0.20·min(1,score)`，三个信号分别代表"答在点上""没有歧义""分数高"，权重是手调的，用 ECE 检验是否过自信。

**展开**：
- coverage：答案句覆盖了多少比例的问题 token，是主信号——没覆盖到就一定是答非所问。
- margin：第一名与第二名的相对差 `(s1-s2)/s1`，差距大说明没有歧义；只有一条候选时定义为 1.0（这也是它会被高估的原因）。
- score：句子得分本身，只作为弱信号，且被 `min(1,·)` 截断。
- 为什么是这三个而不是模型概率：extractive 路径没有任何概率输出；只能用可计算的代理，并且必须能被 ECE 检验。
- 结果：本报告 ECE=0.19，说明这个代理偏乐观/偏保守混在一起，只能当粗排序信号用。

**代码位置**：`src/visioncite/answer/extractive.py:176-191`（置信度组合），`src/visioncite/answer/extractive.py:162-173`（margin），`src/visioncite/answer/extractive.py:79-102`（句子打分）

**常见坑 / 追问**：
- 坑：`margin` 在单候选时返回 1.0，等于无条件给 0.25 分；表格类问题经常只有一条候选，置信度因此虚高。
- 坑：`score = 0.7·coverage + 0.3·min(1, density)`，density 除以句子长度的平方根——长句仍然占优，只是没有线性增长那么夸张。
- 追问"能不能学一个校准器"：能，用 isotonic / Platt scaling 在几百条标注上拟合；36 题不够，会直接过拟合。

### Q23. AnswerAcc=1.00 但 EM=0.00、F1=0.40，这不是自相矛盾吗？

**一句话回答**：不矛盾，是三个指标问的问题不同：AnswerAcc 问"标准答案是否出现在回答里"，EM 问"是否完全相等"，F1 问"token 重合多少"；抽取式返回整句，所以包含成立、完全相等不成立。

**展开**：
- `answer_contains`：gold 归一化后是否是 prediction 的子串；`exact_match`：两者归一化后完全相等。
- `token_f1`：中文按字 bigram 切，算 P/R 的调和平均；返回整句时 precision 被大量无关 token 拉低，recall 接近 1。
- 归一化会去空白、去中英文标点、转小写，所以 `¥1,286.40` 与 `1286.40` 在 contains 判据下等价。
- 报告口径：accuracy/EM/F1/Cite-* 只在"可答且实际作答"的样本上算；弃答单独报 coverage 与 abstain_accuracy。这样"少答换准确"不会被一个混合指标掩盖。
- 全部数字都能在 `.json` 的逐样本记录里复核，不是手填的。

**代码位置**：`src/visioncite/eval/metrics.py:3-15`（口径声明），`src/visioncite/eval/metrics.py:59-102`（contains / F1），`src/visioncite/eval/metrics.py:280-318`（汇总），`src/visioncite/eval/run.py:106-115`（逐样本打分）

**常见坑 / 追问**：
- 坑：EM=0 说明抽取粒度是"整句"，这在产品上通常可接受（用户能读），但在指标上很难看；面试时不要把它包装成优点，直接说"粒度问题，Q17 里列了改法"。
- 追问"为什么 F1 只有 0.40"：table 类回答把整张表拼成一句（F1 0.1），是主要拖累项；invoice 的 0.4 与 chart 的 0.5 也是整句造成的。
- 追问"能不能用 BLEU/ROUGE"：对短答案不敏感，且中文分词依赖更重；token F1 足够解释。

### Q24. 配置为什么集中在一个 Settings？有什么代价？

**一句话回答**：所有可调参数只有一个来源，能被环境变量覆盖、能被 `/health` 打印、评测与线上不会出现两套数值；代价是 `from_env` 只映射了一部分字段。

**展开**：
- `@dataclass(frozen=True)`：不可变，避免运行期被某处随手改掉；所有默认值集中在类体里（bm25_k1=1.5、b=0.75、top_k=8、rrf_k=60、mmr_lambda=0.7、abstain_threshold=0.35）。
- `from_env` 用 `_env_int/_env_float/_env_str` 包了一层，非法值静默回退默认——服务不因为一个坏环境变量起不来。
- `describe()` 给 `/health` 用，只暴露非敏感项（不含密钥）；`ensure_dirs()` 幂等建目录。
- 命名统一 `VISIONCITE_` 前缀，便于容器化。

**代码位置**：`src/visioncite/config.py:50-93`（字段与默认值），`src/visioncite/config.py:95-114`（环境变量映射），`src/visioncite/config.py:127-146`（派生属性与 describe）

**常见坑 / 追问**：
- 坑（真）：`from_env` 只映射了 cache/render_dpi/max_image_side/max_upload_mb/abstain_threshold/ocr_mode/visual_mode/clip_model/hf_endpoint/vlm_*；`bm25_k1`、`bm25_b`、`top_k`、`rrf_k`、`mmr_lambda`、`min_citation_score` **没有对应环境变量**，设了也不生效。
- 追问"怎么只改一处"：评测脚本与 API 都从 `Settings` 取参数，没有硬编码散落值——除了 `chunking` 的 240、`reading_order` 的 40、`_drop_tiny` 的 900 这类"结构性常量"还是字面量，属于有意为之的局部默认。
- 追问"密钥怎么管"：`VlmAnswerer` 从 `VISIONCITE_VLM_API_KEY` 读，`describe()` 不打它。

### Q25. doc_id 用内容哈希，那"同一份文档换个分辨率重新导出"会怎样？

**一句话回答**：会当成新文档，索引重建；dHash 判重的代码写了但没接进主链路，所以现在只有"字节完全相同"才能命中去重。

**展开**：
- `document_id_from_bytes(data)` = `sha256(data).hexdigest()[:16]`；`_fingerprint(path)` 分块读文件算同样的东西。上传接口先用字节算 id，命中字典就直接返回旧文档，不重新解析。
- 好处：天然幂等、无需额外元数据、跨进程可复现（同一个文件永远同一个 id）。
- 坏处：任何字节差异（重新压缩、加水印、改分辨率）都变成新文档；同内容不同格式（PDF vs 导出的 PNG）也是两份。
- 仓库里其实有 `perceptual_hash`（dHash，64 位，汉明距离 ≤5 视为同画面）与 `hash_distance`，还从 `ingest/__init__.py` 导出了，但没有任何调用点——典型的"写了没接线"。
- 如果接上，正确做法是：先用内容哈希做精确去重，再用 dHash 做近似去重的候选召回，最后可以用区域级文本相似度复核。

**代码位置**：`src/visioncite/pipeline.py:92-101`（内容指纹），`src/visioncite/api/app.py:93-95`（上传去重），`src/visioncite/ingest/image.py:102-135`（dHash 与汉明距离，未接线）

**常见坑 / 追问**：
- 坑：只取前 16 位十六进制 = 64 bit，碰撞概率对本场景可忽略，但如果要做全局去重应保留更长。
- 追问"为什么不用文件名"：文件名可重名、可改，且不反映内容。
- 追问"缓存目录怎么清"：没有 TTL/GC，`work_dir` 会一直长；生产要靠外部清理或生命周期策略，这是没做的部分。

### Q26. VLM 这条路径是怎么约束输出的？验证过吗？

**一句话回答**：system prompt 约定严格 JSON（answer / citations / confidence），temperature=0 + `response_format=json_object`，解析时用正则兜底提取 JSON；**没有在真实厂商端点上验证过**，代码注释里自己写明了。

**展开**：
- 输入是"问题 + 每条证据（id + 页码 + 文本）+ 对应裁图（data URI）"，证据 id 就是 chunk_id，模型只能用这些 id。
- 输出解析：`_JSON_BLOCK = \{.*\}` 先粗提，`json.loads` 失败就抛 `AnswerBackendError`；`citations` 非 list 就当空，`confidence` 强制截到 [0,1]。
- 引用白名单：`_map_citations` 只保留真实存在的 chunk_id，未知 id 直接丢弃——模型编造 id 不会进入最终引用。
- 失败策略：网络错误、非 JSON、答案为空、引用映射后为空，四种情况全部回退 extractive（`qa.py:100-111`），不向用户报错。
- 为什么做成可选适配器：不同公司用不同模型，统一到 OpenAI 兼容协议，换模型只改环境变量；没有 key 也能跑完整链路。

**代码位置**：`src/visioncite/answer/vlm.py:34-40`（system prompt 四条规则），`src/visioncite/answer/vlm.py:122-162`（请求体构造），`src/visioncite/answer/vlm.py:164-200`（解析与截断），`src/visioncite/answer/qa.py:143-154`（引用白名单）

**常见坑 / 追问**：
- 坑：`re` 提取 `{.*}` 是贪婪的，模型输出两段 JSON 时会取到跨段内容导致解析失败；正确做法是逐个括号配对或直接用 `response_format` 保证单对象。
- 坑：每条证据都塞一张 data URI 图，token 与带宽成本随 top_k 线性涨；没有做图片数量上限。
- 追问"为什么不流式"：当前是同步阻塞 HTTP，超时 45 s；流式会牵动前端与校验层（校验必须等完整 JSON），是有意的取舍。

### Q27. 异常体系为什么这么设计？HTTP 状态码怎么映射？

**一句话回答**：小异常树 + 每个异常带 `user_message`，调用方靠类型区分"用户传错文件"和"服务器缺模型"，路由层再把类型翻译成 413/415/404/400。

**展开**：
- 基类 `VisionCiteError(message, user_message)`：`message` 给开发者看日志，`user_message` 是中文一句话，可以直接返回给前端。
- 六个子类：`UnsupportedFileError`（415）、`IngestError`（400）、`OcrUnavailableError` / `VisualUnavailableError`（可降级，不上抛给用户）、`AnswerBackendError`（回退 extractive）、`VerificationError`（语义上是"安全网工作"，转成弃答而非报错）。
- 路由层只捕获两类：`UnsupportedFileError` → 415，其余 `VisionCiteError` → 400；文档不存在是 404，文件过大是 413。
- 可降级错误的原则：记录日志 + 换后端，不打断请求；不可降级错误：翻译成用户能懂的提示，不暴露栈。

**代码位置**：`src/visioncite/errors.py:11-69`（异常树），`src/visioncite/api/app.py:100-106`（状态码映射），`src/visioncite/errors.py:63-69`（VerificationError 的定位说明）

**常见坑 / 追问**：
- 坑：`VerificationError` 定义了但没在 `verify.py` 里抛——校验用返回值（`Verification.ok`）而不是异常，这个类目前是"语义占位"。
- 坑：`OcrUnavailableError` 也没有抛点：OCR 不可用走的是 Null 后端，不抛异常。
- 追问"为什么不统一 500"：用户能据此知道"是我文件的问题还是服务的问题"，减少无效重试。

### Q28. PDF 文本层这条降级路径的代价是什么？

**一句话回答**：能答但引用不准——整页被包成一个 Region，引用框就是整页，IoU 对任何 gold 框都容易过 0.5，Cite-R 会虚高。

**展开**：
- `render_pages` 用 pypdfium2 按 DPI 渲染（scale = dpi/72），同时用 `get_text_range()` 抽文本层；扫描件抽到空串是正常情况。
- `_analyze_page`：OCR 没词但文本层非空 → 不跑 XY-cut，直接 `_text_layer_region` 造一个覆盖整页的 Region，`text=page.text_layer`。
- 后果：整页只有一个 chunk（或按 240 字切成几个），区域框是 `(0,0,W,H)`；引用展示时高亮整页，用户看不出具体位置。
- 为什么还留着这条路：电子版 PDF 的文本层比 OCR 准得多（没有识别错误），在没有 OCR 依赖的环境里这是唯一能答的来源。
- 更彻底的做法：用 PDFium 的字符矩形接口（`count_rects` / `get_rect`）建区域，但这会引入第二套坐标体系，当前没做。

**代码位置**：`src/visioncite/pipeline.py:165-198`（降级分支与整页区域），`src/visioncite/ingest/pdf.py:47-85`（渲染），`src/visioncite/ingest/pdf.py:110-123`（文本层抽取），`src/visioncite/ingest/pdf.py:61`（DPI→scale）

**常见坑 / 追问**：
- 坑：`max_pages=40` 写死在 `render_pages` 默认值里，超过 40 页的 PDF 会被静默截断，没有任何提示——这是真实的坑。
- 追问"引用虚高怎么办"：评测里应把"整页框"这种退化引用单独统计，或者对 gold 框也做同粒度比较；当前指标没有区分。
- 追问"为什么不用 pdfplumber"：pypdfium2 是 PDFium 绑定，wheel 自带二进制，跨平台行为一致，不依赖系统 poppler。

### Q29. 表格的文本表示为什么是 `a | b` 行拼接？

**一句话回答**：保留列序、人和 VLM 都能读，且换行让每一行天然成为一个句子；代价是丢掉列对齐、多级表头和合并单元格，跨列推理做不了。

**展开**：
- `build_text` 对 table 区域按行拼 `" | ".join(...)`，对 text 区域用空格拼；行由 `group_rows` 按"垂直中心差 < 中位字高 × 0.6"聚类得到，行内按 x 排序。
- 用 `|` 而不是逗号：逗号在金额里本来就有（`1,286.40`），会歧义；`|` 在文本里罕见，切分安全。
- 每行以换行分隔，而切句正则会按换行切——所以表格的每一行会变成独立句子，BM25 与抽取都在行粒度工作，不会把整张表当成一句话（除非整表没有换行，本报告的 table 样本就是这种情况）。
- 局限：多级表头（合并单元格）会被拆成多行；空单元格 `-` 可能与数字粘连；单元格内的换行会破坏行结构。

**代码位置**：`src/visioncite/layout/regions.py:127-143`（按类型拼文本），`src/visioncite/layout/regions.py:146-177`（行聚类），`src/visioncite/index/chunking.py:19`（按换行切句）

**常见坑 / 追问**：
- 坑：本报告的 table 回答是整张表拼成一段（`reports` 里 table-0-q0 的 answer 就是全表），说明那一页的行没被正确拆分——既拉低 F1（0.1），也让"合计"这一行丢了标签。
- 追问"怎么改"：输出结构化单元格（行列索引 + 文本 + bbox），答案引用到单元格；或让 VLM 直接读表格图并输出 JSON。
- 追问"为什么不用 HTML table"：会引入标签噪声，且 BM25 不需要结构标记；结构化留给下游，而不是塞进检索文本。

### Q30. 这些数字应该怎么读？哪些话不能说？

**一句话回答**：36 题、4 类模板、程序合成、只有 layout 一臂、CI 已退化；可以说"在这个自建小基准上 AnswerAcc 1.00、弃答判断 0.83、p50 0.30 ms"，不能说"效果全面更好"这类没有对照组的评价，也不能说"视觉检索带来增益"（这一臂没跑）。

**展开**：
- 语料是合成的：标准答案与证据框由渲染坐标精确得到（`templates.py` 用 `draw.textbbox` 记录），所以 IoU 命中比真实扫描件乐观；文档自己声明了"不代表真实扫描件的难度"。
- 样本量：36 题里可答 28、实际作答 24；n=24 时 bootstrap CI 退化成 [1,1]，没有统计分辨力。
- 单臂：报告只有 layout（区域 + BM25，无 CLIP、无 VLM），page/hybrid 没跑，所以任何"双路融合更好"的说法都缺证据。
- 失败样本要主动讲：4 道可答题弃答、2 道不可答题误答；这比只报 1.00 更可信。
- 延迟口径：p50 0.30 ms 是"检索 + 抽取 + 校验"，不含 OCR 与建索引，也不是端到端 HTTP 延迟；说的时候必须带这个限定。
- 一句话模板：「自建 36 题合成基准（8 图，4 类模板），CPU-only，layout 模式：AnswerAcc 1.00（24/24），弃答判断 0.83，p50 0.30 ms；4 题可答被弃答、2 题不可答被误答。」

**代码位置**：`src/visioncite/eval/run.py:169-207`（报告生成与口径说明），`src/visioncite/eval/metrics.py:3-15`（指标定义），`src/visioncite/eval/templates.py:24`（页面尺寸与坐标来源），`reports/eval-20261106-134357.md:1-22`（真实报告）

**常见坑 / 追问**：
- 坑：`templates.py:239` 里那行"评测：60 条样本，ECE 0.08"是**合成幻灯片里的演示文本**，不是项目指标；实际是 36 题、ECE 0.19。被问到要立刻澄清，否则会被当成夸大。
- 追问"为什么不用公开数据集"：DocVQA/ChartQA 几个 GB、标注框风格不一、CPU 跑不动；自建的核心优势是"证据框完全已知"，能做严格的 IoU 评测。
- 追问"下一步怎么让数字可信"：扩规模 + 真人扫描件 + 分层 bootstrap + 三臂消融 + 逐类别置信区间，缺一不可。
