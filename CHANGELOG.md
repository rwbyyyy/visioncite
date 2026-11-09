# 更新日志

## v0.2（2026-11-09）

- **Agent**：最小工具调用循环（LLM 选工具 → 执行 → 结果回灌 → 终止）；引用只能来自工具真正返回的
  `chunk_id`，最终答案仍走数字一致性/词面支撑校验；模型失败自动回退抽取式。
- **工具层**：`search_document` / `get_region` / `list_pages` / `dominant_color`，带参数校验与错误回传。
- **LLM 抽象**：`LLMProvider` 接口 + `FakeProvider`（离线演示/测试）+ OpenAI 兼容客户端（未在真实端点验证）。
- **MCP**：stdio + JSON-RPC 2.0 的最小 Server/Client（initialize / tools/list / tools/call / ping），
  同一套工具可被外部宿主发现与调用。
- **接口**：新增 `POST /api/agent`，返回答案、引用与工具执行轨迹。
- **测试**：118 → 119 个用例（新增 LLM/工具/Agent/MCP 四组）。

## 未发布

- **数值收敛**：问"成绩多少/单价多少"时返回具体值而不是整行；
  有表头时按列名对齐（百分制/绩点/学分/单价），无表头时按语义与单位启发式挑选；
  编码类数字（`INV-2026-100`、学号）不会被误当答案。
- **颜色属性问答**：字体颜色 / 底色 / 印章颜色由像素统计回答（HSV 分桶），
  印章会先定位红色区域再统计。
- **表格按行切块**：修复表格行被句号打包器粘连（`小计显示器`）的问题。
- **上传不阻塞事件循环**：解析改为同步路由，走线程池，上传期间 `/health` 不再超时。

## v0.1.0（2026-11-06）

首个可运行版本。

**能力**

- 文档接入：PDF（pypdfium2 渲染 + 文本层）、PNG/JPEG/WebP 图片
- 版面分析：Otsu 二值化、递归 XY-cut 区域分割、带框线表格检测、区域类型判定
- OCR：RapidOCR（离线 CPU，中英），无依赖时自动降级为 PDF 文本层
- 检索：中文 bigram BM25 + CLIP ViT-B/32 视觉向量，RRF 融合 + MMR 去冗余
- 作答：抽取式答案（默认，无需 API key）；可选 OpenAI 兼容 VLM 适配器
- 可信度：引用存在性校验、数字一致性校验、词面支撑度校验、低置信弃答
- 服务：FastAPI（上传/提问/页面图/证据裁图/健康检查）+ 单页前端
- 评测：合成双语料、AnswerAcc/EM/F1/Cite-P/Cite-R/AbstainAcc/ECE、bootstrap 置信区间、三档消融

**工程**

- 全部模块带类型注解与中文 docstring；`ruff`（含 C901/D/ANN）零告警
- pytest 82 个用例，完全离线
- GitHub Actions：Ubuntu + Windows × Python 3.11/3.12

**已知限制**（详见 docs/LIMITATIONS.md）

- 表格只支持有可见框线的；多张小表可能被包成一张
- 评测语料为程序合成，难度低于真实扫描件
- VLM 适配器未在真实厂商端点验证
- Dockerfile 未在本机构建验证
