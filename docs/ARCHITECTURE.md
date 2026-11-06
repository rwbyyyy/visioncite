# 架构说明

## 1. 分层与依赖方向

```
api/  cli.py  eval/          ← 入口层（HTTP / CLI / 评测）
        ↓
answer/qa.py                 ← 用例层：检索 → 作答 → 校验 → 弃答
        ↓
index/  →  layout/  ocr/  ingest/   ← 能力层（检索、版面、OCR、解析）
        ↓
types.py  config.py  errors.py  logging.py   ← 契约层
```

**依赖只能向下**。`layout` 不知道 `answer` 存在，`ingest` 不知道 `index` 存在。
这条规则带来两个直接好处：

1. 每一层都能单独测（见 `tests/` 与模块一一对应）；
2. 替换实现不影响上游，例如把 RapidOCR 换成云 OCR，只需新增一个 `OcrBackend`。

## 2. 数据流

```
文件字节
  → ingest.load_pages：PDF 渲染成图 / 图片直读；统一成 Page（RGB + 宽高）
  → layout.build_regions：Otsu 二值化 → XY-cut → 表格线 → 区域（text/table/figure）
  → ocr.recognize：RapidOCR 给出 Word(text, bbox, confidence)
  → index.chunks_from_pages：区域文本按句切块，保留 region_id 与 bbox
  → index.BM25Index.fit + visual 编码：文本倒排 + 区域向量
  → 用户提问
  → index.retrieve：BM25 topN + CLIP topN → RRF 融合 → MMR 去冗余
  → answer.extractive / answer.vlm：生成答案（默认抽取式，无需 API key）
  → answer.verify：引用存在性 + 数字一致性 + 词面支撑
  → answer.abstain：不达标则弃答
  → AnswerResult(answer, citations[bbox], confidence, latency)
```

## 3. 关键设计决定

| 决定 | 理由 | 代价 |
|---|---|---|
| PDF 一律渲染成图再走统一流水线 | 扫描件与电子版共用一条路径，证据框统一 | 渲染耗时；文本层信息需要额外降级使用 |
| 版面用 XY-cut 而非深度模型 | 无参数、毫秒级、可解释；无 GPU 也能跑 | 复杂多栏/倾斜版面失效 |
| 中文用字 bigram | 免词典、对未登录词稳 | 索引膨胀、虚词靠 IDF 压制 |
| 检索用 RRF 而非分数加权 | 不同检索器分数尺度不可比 | 丢失分数幅度信息 |
| 默认抽取式作答 | 离线、可复现、作为 VLM 的对照基线 | 不能跨段归纳、算术 |
| 证据必须绑定 bbox | 可信 AI 的最低要求：可核对 | 版面错误会直接变成引用错误 |

## 4. 如果上生产，需要改什么

- **存储**：`app.state.documents` 换成 Redis + 对象存储；页面图放 S3 兼容存储。
- **索引**：BM25 换成 OpenSearch/Elasticsearch；向量换成 Milvus/pgvector。
- **OCR**：换成带版面信息的商用服务（或 PaddleOCR + 版面模型），提高表格与多栏准确率。
- **异步**：上传后立刻返回 `doc_id`，解析放队列（Celery/RQ），前端轮询状态。
- **可观测**：接入 OpenTelemetry，记录每阶段耗时、弃答率、引用命中率。
- **安全**：上传文件做病毒扫描；限制页数/分辨率；对 OCR 文本做提示注入检测。

## 5. 目录对照

| 目录 | 职责 | 测试 |
|---|---|---|
| `ingest/` | 文件识别、PDF 渲染、图片预处理 | `tests/test_pipeline.py` |
| `layout/` | 二值化、XY-cut、表格线、区域装配合并 | `tests/test_layout.py` |
| `ocr/` | OCR 接口、RapidOCR、空实现、工厂 | `tests/test_pipeline.py`（假 OCR） |
| `index/` | 分词、BM25、CLIP、RRF/MMR、切块、检索 | `tests/test_index.py` |
| `answer/` | 抽取式作答、VLM、校验、弃答、编排 | `tests/test_answer.py` |
| `api/` | HTTP 契约与路由 | `tests/test_api.py` |
| `eval/` | 语料生成、指标、bootstrap、报告 | `tests/test_eval.py` |
