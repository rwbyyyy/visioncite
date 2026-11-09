"""HTTP 契约（Pydantic 模型）。

把"请求/响应长什么样"单独定义，好处是：
- FastAPI 自动生成 OpenAPI 文档；
- 前端可以照着 schema 写，不靠口头约定；
- 字段改名会立刻在类型检查/测试里暴露。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegionOut(BaseModel):
    """一个版面区域。"""

    region_id: str = Field(description="页内唯一 id，如 p0-r3")
    kind: str = Field(description="text / table / figure / unknown")
    bbox: list[int] = Field(description="[x0, y0, x1, y1]")


class PageOut(BaseModel):
    """一页的元信息。"""

    index: int
    width: int
    height: int
    regions: list[RegionOut]


class DocumentOut(BaseModel):
    """上传成功后的文档信息。"""

    doc_id: str
    filename: str
    page_count: int
    ocr: str = Field(description="实际使用的 OCR 后端")
    visual: str = Field(description="实际使用的视觉后端")
    pages: list[PageOut]


class AskRequest(BaseModel):
    """提问请求。"""

    doc_id: str = Field(description="上传接口返回的 doc_id")
    question: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    """一条证据引用。"""

    chunk_id: str
    region_id: str
    page: int
    bbox: list[int]
    snippet: str
    score: float


class AskResponse(BaseModel):
    """问答响应。"""

    answer: str
    abstained: bool
    confidence: float
    latency_ms: float
    backend: str
    citations: list[CitationOut]
    reasons: list[str] = []


class HealthOut(BaseModel):
    """健康检查。"""

    status: str
    version: str
    ocr: str
    visual: str
    vlm_enabled: bool
    documents: int


class AgentAskRequest(BaseModel):
    """Agent 提问请求。"""

    doc_id: str
    question: str = Field(min_length=1, max_length=500)


class AgentStepOut(BaseModel):
    """Agent 的一步工具调用。"""

    tool: str
    arguments: dict = {}
    result: str = ""


class AgentAskResponse(BaseModel):
    """Agent 问答响应。"""

    answer: str
    abstained: bool
    citations: list[CitationOut]
    steps: list[AgentStepOut]
    tokens: int
    backend: str
    reasons: list[str] = []
