"""FastAPI 服务：把问答能力暴露成 HTTP 接口 + 一个最小可用的网页。

分层：
    路由层（本文件）只做三件事：校验输入、调用用例、把结果/异常翻译成 HTTP。
    业务逻辑全部在 pipeline / answer 里，因此路由函数很薄，也好测。

存储：进程内字典。够用于演示与单机部署；多副本部署时应换成 Redis/对象存储，
这一条写在 docs/ARCHITECTURE.md 的"如果上生产"里。

路由注册被拆成四个小组（文档 / 问答 / 资源 / 页面），
每个函数只注册一类路由 —— 这样单个函数的圈复杂度不会超标，读起来也分块清晰。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from visioncite.answer.qa import answer_question
from visioncite.api.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AgentStepOut,
    AskRequest,
    AskResponse,
    CitationOut,
    DocumentOut,
    HealthOut,
    PageOut,
    RegionOut,
)
from visioncite.config import Settings
from visioncite.errors import UnsupportedFileError, VisionCiteError
from visioncite.ingest.image import crop_image, to_png_bytes
from visioncite.ingest.loader import detect_kind
from visioncite.logging import get_logger, setup_logging
from visioncite.pipeline import Document, build_document, document_id_from_bytes

logger = get_logger("api")

STATIC_DIR = Path(__file__).resolve().parent.parent / "web"
_VERSION = "0.2.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造应用（工厂模式，方便测试注入不同配置）。

    Args:
        settings: 配置；None 时从环境变量读取。

    Returns:
        FastAPI 实例。
    """
    setup_logging()
    config = settings or Settings.from_env()
    config.ensure_dirs()
    app = FastAPI(title="VisionCite", version=_VERSION, description="多模态文档问答与证据溯源")
    app.state.settings = config
    app.state.documents = {}

    _register_health_routes(app)
    _register_document_routes(app)
    _register_agent_routes(app)
    _register_asset_routes(app)
    _register_ui_routes(app)
    return app


def _register_health_routes(app: FastAPI) -> None:
    """健康检查路由。

    Args:
        app: FastAPI 实例。
    """

    @app.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        """报告实际启用的能力与已索引文档数。"""
        documents: dict[str, Document] = app.state.documents
        sample = next(iter(documents.values()), None)
        return HealthOut(
            status="ok",
            version=_VERSION,
            ocr=sample.ocr_name if sample else "unknown",
            visual=sample.visual_name if sample else "unknown",
            vlm_enabled=app.state.settings.vlm_enabled,
            documents=len(documents),
        )


def _register_document_routes(app: FastAPI) -> None:
    """上传与问答路由。

    Args:
        app: FastAPI 实例。
    """

    @app.post("/api/documents", response_model=DocumentOut)
    def upload(file: UploadFile = File(...)) -> DocumentOut:
        """上传并解析文档；同一内容重复上传会直接复用。

        刻意写成**同步**函数：解析是 CPU 密集的（OCR + 版面 + 编码），
        FastAPI 会把同步路由丢进线程池，事件循环因此不会被阻塞，
        解析大文件时 /health 与页面资源仍然可访问。
        （实测：写成 async def 并直接调用解析，上传期间 /health 会超时。）
        """
        data = file.file.read()
        limit = app.state.settings.max_upload_mb * 1024 * 1024
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"文件超过 {app.state.settings.max_upload_mb}MB")
        return _store_document(app, data, file.filename or "upload")

    @app.post("/api/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        """提问：检索 + 作答 + 校验。"""
        documents: dict[str, Document] = app.state.documents
        document = documents.get(request.doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在，请先上传")
        result = answer_question(document, request.question, app.state.settings)
        return AskResponse(
            answer=result.answer,
            abstained=result.abstained,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            backend=result.backend,
            citations=[
                CitationOut(
                    chunk_id=citation.chunk_id,
                    region_id=citation.region_id,
                    page=citation.page,
                    bbox=citation.bbox.to_list(),
                    snippet=citation.snippet,
                    score=citation.score,
                )
                for citation in result.citations
            ],
            reasons=list(result.reasons),
        )


def _register_agent_routes(app: FastAPI) -> None:
    """Agent 路由：工具调用式问答。

    Args:
        app: FastAPI 实例。
    """

    @app.post("/api/agent", response_model=AgentAskResponse)
    def agent_ask(request: AgentAskRequest) -> AgentAskResponse:
        """让 Agent 先用工具检索、再作答，并返回执行轨迹。"""
        from visioncite.agent.loop import run_agent
        from visioncite.llm.factory import create_provider
        from visioncite.llm.fake import FakeProvider

        document = app.state.documents.get(request.doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在，请先上传")
        provider = create_provider(app.state.settings)
        if isinstance(provider, FakeProvider):
            provider = FakeProvider()  # 演示模式：每次请求重置脚本
        result = run_agent(document, request.question, app.state.settings, provider=provider)
        return AgentAskResponse(
            answer=result.answer,
            abstained=result.abstained,
            citations=[
                CitationOut(
                    chunk_id=citation.chunk_id,
                    region_id=citation.region_id,
                    page=citation.page,
                    bbox=citation.bbox.to_list(),
                    snippet=citation.snippet,
                    score=citation.score,
                )
                for citation in result.citations
            ],
            steps=[
                AgentStepOut(tool=step.name, arguments=dict(step.arguments), result=step.result[:200])
                for step in result.steps
            ],
            tokens=result.usage.total_tokens,
            backend=result.backend,
            reasons=list(result.reasons),
        )


def _register_asset_routes(app: FastAPI) -> None:
    """页面图与证据裁图路由。

    Args:
        app: FastAPI 实例。
    """

    @app.get("/api/documents/{doc_id}/pages/{page}.png")
    def page_image(doc_id: str, page: int) -> Response:
        """返回页面 PNG，供前端叠加高亮框。"""
        document = app.state.documents.get(doc_id)
        target = document.page(page) if document else None
        if target is None:
            raise HTTPException(status_code=404, detail="页面不存在")
        return Response(content=to_png_bytes(target.image), media_type="image/png")

    @app.get("/api/documents/{doc_id}/regions/{region_id}.crop")
    def region_crop(doc_id: str, region_id: str) -> Response:
        """返回某个区域的裁图（前端引用缩略图）。"""
        document = app.state.documents.get(doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        for page in document.pages:
            for region in page.regions:
                if region.region_id == region_id:
                    box = region.bbox.pad(8, (page.width, page.height))
                    return Response(
                        content=to_png_bytes(crop_image(page.image, box)),
                        media_type="image/png",
                    )
        raise HTTPException(status_code=404, detail="区域不存在")


def _register_ui_routes(app: FastAPI) -> None:
    """静态页面路由。

    Args:
        app: FastAPI 实例。
    """

    @app.get("/")
    def index() -> FileResponse:
        """返回单页前端。"""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/app.js")
    def script() -> FileResponse:
        """前端脚本。"""
        return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def styles() -> FileResponse:
        """前端样式。"""
        return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


def _store_document(app: FastAPI, data: bytes, filename: str) -> DocumentOut:
    """把上传字节落盘、解析并放进内存缓存。

    Args:
        app: FastAPI 实例。
        data: 文件字节。
        filename: 原始文件名。

    Returns:
        DocumentOut。

    Raises:
        HTTPException: 类型不支持（415）或解析失败（400）。
    """
    documents: dict[str, Document] = app.state.documents
    doc_id = document_id_from_bytes(data)
    if doc_id in documents:
        return _to_document_out(documents[doc_id])

    suffix = Path(filename).suffix.lower() or ".bin"
    target = app.state.settings.work_dir / f"{doc_id}{suffix}"
    target.write_bytes(data)
    try:
        detect_kind(target)
        document = build_document(target, app.state.settings, doc_id=doc_id)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=415, detail=exc.user_message) from exc
    except VisionCiteError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    documents[doc_id] = document
    return _to_document_out(document)


def _to_document_out(document: Document) -> DocumentOut:
    """把内部 Document 转成响应模型。

    Args:
        document: 文档。

    Returns:
        DocumentOut。
    """
    pages = [
        PageOut(
            index=page.index,
            width=page.width,
            height=page.height,
            regions=[
                RegionOut(region_id=region.region_id, kind=region.kind, bbox=region.bbox.to_list())
                for region in page.regions
            ],
        )
        for page in document.pages
    ]
    return DocumentOut(
        doc_id=document.meta.doc_id,
        filename=document.meta.filename,
        page_count=document.meta.page_count,
        ocr=document.ocr_name,
        visual=document.visual_name,
        pages=pages,
    )


# 模块级实例：`uvicorn visioncite.api.app:app` 直接可用，且不会触发
# "ASGI app factory detected" 警告（提示是给"导出的是工厂函数"的场景用的）。
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080, reload=False)
