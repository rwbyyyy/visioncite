"""HTTP 接口测试：用 TestClient，不启动真实服务器。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visioncite.api.app import create_app
from visioncite.config import Settings
from visioncite.index.visual import NullVisualEncoder
from visioncite.pipeline import build_document
from visioncite.types import Word


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    document_path: tuple[Path, list[Word]],
    fake_ocr,  # noqa: ANN001
    settings: Settings,
) -> TestClient:
    """构造一个注入了假文档解析的 TestClient。"""
    path, _ = document_path
    document = build_document(path, settings, ocr=fake_ocr, visual=NullVisualEncoder())
    monkeypatch.setattr("visioncite.api.app.build_document", lambda *args, **kwargs: document)
    return TestClient(create_app(settings))


def test_health(client: TestClient) -> None:
    """健康检查返回 ok 与能力信息。"""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vlm_enabled"] is False


def test_upload_returns_document(client: TestClient, document_path: tuple[Path, list[Word]]) -> None:
    """上传成功返回 doc_id 与页面结构。"""
    path, _ = document_path
    response = client.post(
        "/api/documents",
        files={"file": ("page.png", path.read_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 1
    assert body["pages"][0]["width"] == 640
    assert body["doc_id"]


def test_ask_after_upload(client: TestClient, document_path: tuple[Path, list[Word]]) -> None:
    """上传后能提问并拿到引用。"""
    path, _ = document_path
    upload = client.post(
        "/api/documents",
        files={"file": ("page.png", path.read_bytes(), "image/png")},
    ).json()
    response = client.post(
        "/api/ask",
        json={"doc_id": upload["doc_id"], "question": "合计金额是多少？"},
    )
    assert response.status_code == 200
    body = response.json()
    assert not body["abstained"]
    assert body["citations"]
    assert body["citations"][0]["bbox"]


def test_ask_unknown_document_returns_404(client: TestClient) -> None:
    """未知文档返回 404，而不是 500。"""
    response = client.post("/api/ask", json={"doc_id": "missing", "question": "金额"})
    assert response.status_code == 404


def test_upload_rejects_unsupported_file(client: TestClient) -> None:
    """不支持的类型返回 415。"""
    response = client.post(
        "/api/documents",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_rejects_too_large(client: TestClient) -> None:
    """超过大小限制返回 413。"""
    client.app.state.settings = Settings(max_upload_mb=0)
    response = client.post(
        "/api/documents",
        files={"file": ("page.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert response.status_code == 413


def test_page_image_and_crop(client: TestClient, document_path: tuple[Path, list[Word]]) -> None:
    """页面图与证据裁图都能取到 PNG。"""
    path, _ = document_path
    doc_id = client.post(
        "/api/documents",
        files={"file": ("page.png", path.read_bytes(), "image/png")},
    ).json()["doc_id"]
    page = client.get(f"/api/documents/{doc_id}/pages/0.png")
    assert page.status_code == 200
    assert page.headers["content-type"] == "image/png"

    body = client.post("/api/ask", json={"doc_id": doc_id, "question": "合计金额是多少？"}).json()
    region_id = body["citations"][0]["region_id"]
    crop = client.get(f"/api/documents/{doc_id}/regions/{region_id}.crop")
    assert crop.status_code == 200
    assert crop.headers["content-type"] == "image/png"


def test_index_page(client: TestClient) -> None:
    """单页前端可以访问。"""
    response = client.get("/")
    assert response.status_code == 200
    assert "VisionCite" in response.text
