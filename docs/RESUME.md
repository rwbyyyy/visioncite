# VisionCite 简历材料（按最终评测数字重写）

> 仓库：https://github.com/rwbyyyy/visioncite · 数据来源：`reports/eval-20261106-170428.md`（可复现）

## 一、简历条目（可直接粘贴，中文）

**VisionCite · 多模态文档问答与证据溯源系统**（个人项目，2026.10–2026.11）
技术栈：Python、PyTorch(CPU)、CLIP、RapidOCR/ONNXRuntime、numpy、FastAPI、pytest、GitHub Actions、
工具调用（function calling）、MCP（JSON-RPC/stdio）

- 设计并实现"上传 PDF/图片 → 版面分析 → OCR → 双路检索 → 带 bbox 引用的答案 → 校验/弃答"全链路：
  Otsu 二值化 + 递归 XY-cut + 表格线检测把页面切成区域，区域坐标直接作为引用证据。
- 自建 36 题 / 8 页合成基准（发票、表格、图表、幻灯片 4 类模板，答案与证据框由渲染坐标精确得到），
  实现 AnswerAcc / EM / F1 / 引用 P-R(IoU≥0.5) / 弃答准确率 / ECE / 延迟分位数与 bootstrap 置信区间。
- 检索侧不引入训练参数：中文字 bigram BM25（k1=1.5、b=0.75）+ CLIP ViT-B/32 视觉召回，
  按名次做 RRF（k=60）融合、MMR（λ=0.7）去冗余；单题问答 p50≈0 ms、p95 2 ms（CPU-only，不含 OCR）。
- 可信度闭环：引用存在性 + 答案数字一致性 + 词面支撑三道校验，三级弃答（无证据/校验失败/置信度<0.35）；
  实测引用召回 Cite-R 1.00、引用精确 Cite-P 0.85、弃答判断准确率 0.89、ECE 0.17。
- 工程化：119 个离线测试、ruff 零告警（含复杂度/类型注解/docstring 规则）、
  GitHub Actions（Ubuntu + Windows × Python 3.11/3.12）、FastAPI + 单页前端（点击引用高亮页面证据框）；
  缺 OCR / CLIP / VLM 时逐级降级，并由 `/health` 报告实际启用的后端。
- 工具调用与 MCP：实现不依赖框架的最小 Agent 循环（模型选工具 → 执行 → 结果回灌 → 终止），
  把检索/取区域/颜色统计暴露成 4 个工具，答案引用**只能来自工具实际返回的 chunk_id**；
  同一套工具再通过自实现的 MCP Server（stdio + JSON-RPC 2.0）暴露，支持 initialize / tools/list / tools/call。
  提供 fake provider，无 API key 也能完整跑通工具链路并为 119 个用例提供确定性测试。

## 二、精简版（两行，一页简历空间紧张时用）

独立开发多模态文档问答系统：Otsu + 递归 XY-cut 版面分析、OCR 与 CLIP 双路检索（RRF 融合 / MMR 去冗）、
答案携带页面区域引用与三道校验 + 弃答；再实现工具调用 Agent 与 MCP Server（同一套工具可被外部宿主发现/调用）；
自建 36 题基准上 Cite-R 1.00、弃答准确率 0.89；119 个离线测试 + CI 全绿。

## 三、英文版

**VisionCite — Multimodal Document QA with Verifiable Visual Citations** (personal project)
Stack: Python, PyTorch (CPU), CLIP, RapidOCR/ONNXRuntime, numpy, FastAPI, pytest, GitHub Actions.

- Built an end-to-end pipeline: layout analysis (Otsu + recursive XY-cut + ruled-table detection) →
  offline OCR → BM25 + CLIP hybrid retrieval fused by RRF and de-duplicated with MMR →
  citation-grounded answers with numeric/lexical verification and abstention.
- Built a 36-question benchmark over 8 rendered pages with geometry-exact gold boxes; measured
  AnswerAcc, citation precision/recall at IoU≥0.5, abstention accuracy, ECE and latency percentiles.
- Results on the synthetic benchmark (CPU-only): citation recall 1.00, citation precision 0.85,
  abstention accuracy 0.89; 119 offline tests, ruff clean, CI on Ubuntu and Windows.
- Built a framework-free tool-calling agent (model picks tools → runtime executes → results fed back)
  and exposed the same tools over a hand-written MCP server (stdio + JSON-RPC 2.0); citations can only
  come from `chunk_id`s actually returned by tools. Fake provider keeps the whole path testable offline.

## 四、面试展开要点（被追问时按这个顺序讲）

1. **为什么 XY-cut 而不是版面模型**：无参数、毫秒级、每一刀可解释；代价是倾斜/多栏失效——已写进 LIMITATIONS。
2. **为什么 RRF 不直接加权**：BM25 分数无上界、余弦在 [-1,1]，尺度不可比；RRF 只用名次，无可训练参数。
3. **为什么中文用字 bigram**：免词典、对 OCR 错字和未登录词更稳；代价是索引膨胀，虚词靠 IDF 降权。
4. **引用怎么校验**：引用 id 必须存在、答案数字必须逐字出现在被引文本、词面支撑 ≥0.5；任一失败则弃答。
5. **数字型答案怎么收敛**：表头对齐列（百分制/绩点/单价），无表头时按"锚点词后的数字 + 单位优先"挑；
   编号类数字（`INV-2026-100`、学号）不参与。
6. **颜色类问题怎么答**：HSV 分桶统计引用区域主色；"底色"统计亮且低饱和像素；"印章"先定位红色外接框。
7. **为什么自己做 Agent 循环**：只有"模型选工具 → 执行 → 回灌 → 终止"四步，框架抽象反而挡路；
   自己写能控制最大步数、错误回传、以及"引用只能来自工具返回"这条硬约束。
8. **MCP 为什么只做 tools**：这个项目只需要"工具发现 + 调用"闭环，resources/prompts/sampling
   没有使用场景，硬做只会增加未验证代码——已写进 docs/AGENT_MCP.md。
9. **如果重做**：表格单元格级还原、弃答阈值扫描 + 温度缩放校准、把 VLM 接进来做跨段归纳。

## 五、必须主动说的局限（避免被问穿）

- 基准是**程序合成**的 36 题、8 页，样本小；全对时 bootstrap 95% CI 退化为 [1.00, 1.00]，没有统计分辨力。
- **CLIP 在该语料上没有质量增益**，只增加约 30 ms 延迟；视觉一路的收益需要在图表类真实数据上再验证。
- 表格只支持**有可见框线**的；倾斜扫描、无线表、多栏混排未覆盖。
- VLM 适配器按 OpenAI 兼容协议实现，但**未在真实厂商端点验证**；Dockerfile 未实机构建。
- 颜色只给**区域主色**，不识别字体/字号；单一颜色的水印可能干扰判断。
- Agent 依赖的外部 LLM **未在真实厂商端点验证**（只用 fake provider 跑通链路）；
  MCP 只实现 tools 子集；`McpClient.spawn` 的子进程路径未进 CI。

## 六、数字速查（口径写死，避免口误）

| 指标 | 值 | 口径 |
|---|---|---|
| 语料 | 36 题 / 8 图 | 合成，4 类模板 × 2 份 |
| 答题率 | 0.78（28/36 作答） | 全部题目 |
| AnswerAcc | 1.00 | 可答且实际作答（26 题中 26 对） |
| F1 | 0.72 | 字 bigram token 级 |
| Cite-P / Cite-R | 0.85 / 1.00 | IoU ≥ 0.5 |
| 弃答判断准确率 | 0.89 | 可答必答 + 不可答必弃答 |
| ECE | 0.17 | 10 等宽箱 |
| 延迟 | p50≈0 ms / p95 2 ms | 检索+作答+校验，不含 OCR 索引 |
| 视觉一路开销 | p50 30 ms | 同上，hybrid 模式 |
| 测试 / 静态检查 | 119 用例 / ruff 0 告警 | 离线，无模型下载 |
| 工具 / MCP | 4 个工具；MCP 支持 4 个方法 | search/get_region/list_pages/dominant_color |

## 七、三分钟自述讲稿（面试开场照着说）

> "我做过一个多模态文档问答系统，叫 VisionCite。它解决的问题是：企业里的 PDF、扫描件、截图，
> 大模型能读，但答完你没法核对它说的是不是原文里那句。所以我做的核心不是'调模型'，而是
> **把答案落回页面上的一块区域**。
>
> 系统分四层：最底层把 PDF/图片解析成页面，用 Otsu 二值化 + 递归 XY-cut 做版面切分，
> 再叠加 OCR 拿到带坐标的文字；中间层做检索，中文按字 bigram 走 BM25，同时用 CLIP 做视觉召回，
> 两路按**名次**做 RRF 融合、再用 MMR 去冗余；上层作答，默认是抽取式——不需要 API key、
> 毫秒级，同时充当'基线'；最后是可信度层，三道校验：引用 id 必须存在、答案里的数字必须逐字出现在
> 被引文本、词面支撑要够，任一不过就弃答。
>
> 评测是我花时间最多的地方：自建 36 题、8 页的合成基准，答案和证据框都由绘制坐标精确得到，
> 所以引用可以做严格的 IoU 评测。实测引用召回 1.00、引用精确 0.85、弃答判断准确率 0.89，
> 单题延迟 p50 不到 1 毫秒。
>
> 后来又加了两块：一是工具调用，把检索、看区域、颜色统计包成工具，模型自己决定查什么，
> 但引用只能来自工具真正返回过的 chunk_id，模型伪造不了；二是把这些工具用自己写的 MCP Server
> 暴露出去，让别的宿主也能接。
>
> 最大的局限我也很清楚：基准是合成的、只有 36 题，CLIP 在这份语料上只增加延迟没有质量增益，
> 真实扫描件的表格和倾斜页面还没覆盖——这些我都写进了 LIMITATIONS。"

## 八、高频追问速答（每题一句话，细节见 docs/INTERVIEW.md）

| 追问 | 一句话回答 |
|---|---|
| 为什么不用版面检测模型？ | 无参数、毫秒级、每一刀可解释；代价是倾斜/多栏失效，已明确列为局限 |
| 为什么中文用 bigram？ | 免词典、抗 OCR 错字与未登录词；代价是索引膨胀，虚词靠 IDF 降权 |
| RRF 为什么不加权求和？ | BM25 分数无上界、余弦在 [-1,1]，尺度不可比；名次融合没有可调参数 |
| MMR 解决什么？ | 同一段被切成多块时，避免 top-k 里全是近似重复的证据 |
| 怎么防止编数字？ | 答案里的每个数字必须逐字出现在被引文本里，归一化后比对 |
| 为什么弃答？ | 文档场景里"答错"的成本远高于"没答"，三档判断：无证据→校验失败→低置信 |
| AnswerAcc 1.00 是不是过拟合？ | 它只统计"可答且已作答"的 26 题，coverage 只有 0.78；EM 只有 0，说明抽取粒度是整句 |
| 95% CI 为什么是 [1,1]？ | 26 题全对，bootstrap 每次重采样都全对，区间退化——说明样本量不够做显著性 |
| Agent 的引用怎么保证可信？ | 引用只从工具返回的 chunk_id 生成，模型只能决定用不用，不能决定引用谁 |
| 如果重做最想改什么？ | 表格单元格级还原（现在整行拼文本），其次是阈值扫描 + 置信度校准 |
