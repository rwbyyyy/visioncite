# 更新日志

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
