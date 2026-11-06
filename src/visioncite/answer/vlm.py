"""可选 VLM 后端：任何 OpenAI 兼容的视觉大模型。

为什么做成"可选适配器"而不是核心依赖？
    1. 面试官会问"没有 API 能不能跑"——答案是能，默认走 extractive；
    2. 不同公司用的模型不同（Qwen-VL / GPT-4o / 本地 vLLM），
       统一到 OpenAI 的 ``/chat/completions`` 协议，换模型只改环境变量；
    3. 模型输出必须被结构化约束（JSON + 引用白名单），否则校验层形同虚设。

协议约定（本项目自定义，写进 prompts）：
    输入：问题 + 若干条证据（id + 文本 + 对应图片）
    输出：严格 JSON ``{"answer": str, "citations": [chunk_id, ...], "confidence": 0~1}``
    证据不足时：``{"answer": "", "citations": [], "confidence": 0}``

注意：本文件**没有在真实厂商端点上验证过**（仓库运行环境无 key），
代码按协议实现，README 的"未验证"清单里会写明，不夸大。
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass

import httpx

from visioncite.errors import AnswerBackendError
from visioncite.logging import get_logger
from visioncite.types import Chunk

logger = get_logger("answer.vlm")

SYSTEM_PROMPT = """你是严谨的文档问答助手。规则：
1. 只能依据给出的证据回答，不得使用外部知识；
2. 证据不足、问题与证据无关、或证据互相矛盾时，answer 置空、citations 置空、confidence 置 0；
3. 答案中的每个数字都必须在被引证据中逐字出现；
4. 只输出 JSON，不要解释，不要 Markdown 代码块。
输出格式：{"answer": "...", "citations": ["chunk_id", ...], "confidence": 0.0}
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class VlmOutput:
    """VLM 的结构化输出。

    Attributes:
        answer: 答案文本（可能为空）。
        citations: 模型选择的 chunk_id。
        confidence: 模型自报置信度。
    """

    answer: str
    citations: tuple[str, ...]
    confidence: float


class VlmAnswerer:
    """OpenAI 兼容的视觉问答客户端。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 45.0,
    ) -> None:
        """初始化。

        Args:
            base_url: 例如 ``https://dashscope.aliyuncs.com/compatible-mode/v1``。
            model: 模型名，例如 ``qwen-vl-max``。
            api_key: 密钥；None 时从环境变量 ``VISIONCITE_VLM_API_KEY`` 读取。
            timeout_s: 超时秒数。
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("VISIONCITE_VLM_API_KEY", "")
        self._timeout = timeout_s

    def answer(
        self,
        question: str,
        evidence: list[Chunk],
        crops: dict[str, bytes],
    ) -> VlmOutput:
        """调用模型。

        Args:
            question: 用户问题。
            evidence: 证据块。
            crops: chunk_id → 证据裁图（PNG 字节）。

        Returns:
            结构化输出。

        Raises:
            AnswerBackendError: 网络失败、响应非 JSON、字段缺失。
        """
        if not self._api_key:
            msg = "未配置 VISIONCITE_VLM_API_KEY"
            raise AnswerBackendError(msg)

        payload = self._build_payload(question, evidence, crops)
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"VLM 请求失败: {exc}"
            logger.warning(msg)
            raise AnswerBackendError(msg) from exc

        return self._parse(response.json())

    def _build_payload(
        self,
        question: str,
        evidence: list[Chunk],
        crops: dict[str, bytes],
    ) -> dict[str, object]:
        """拼装 messages：文本证据 + 对应的图。

        Args:
            question: 问题。
            evidence: 证据块。
            crops: 裁图。

        Returns:
            OpenAI 格式的请求体。
        """
        content: list[dict[str, object]] = [{"type": "text", "text": f"问题：{question}\n\n证据："}]
        for chunk in evidence:
            content.append(
                {
                    "type": "text",
                    "text": f"[{chunk.chunk_id}] (第{chunk.page + 1}页) {chunk.text}",
                }
            )
            png = crops.get(chunk.chunk_id)
            if png:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_uri(png)},
                    }
                )
        return {
            "model": self._model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }

    def _parse(self, body: dict) -> VlmOutput:
        """解析模型返回。

        Args:
            body: 响应 JSON。

        Returns:
            VlmOutput。

        Raises:
            AnswerBackendError: 结构不符合约定。
        """
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = "VLM 响应缺少 choices"
            raise AnswerBackendError(msg) from exc

        match = _JSON_BLOCK.search(text or "")
        if match is None:
            msg = "VLM 未返回 JSON"
            raise AnswerBackendError(msg)
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            msg = f"VLM JSON 解析失败: {exc}"
            raise AnswerBackendError(msg) from exc

        citations = parsed.get("citations") or []
        if not isinstance(citations, list):
            citations = []
        confidence = parsed.get("confidence", 0.0)
        return VlmOutput(
            answer=str(parsed.get("answer", "")).strip(),
            citations=tuple(str(item) for item in citations),
            confidence=max(0.0, min(1.0, float(confidence) if isinstance(confidence, (int, float)) else 0.0)),
        )


def _data_uri(png: bytes) -> str:
    """把 PNG 字节编码成 data URI。

    Args:
        png: PNG 字节。

    Returns:
        ``data:image/png;base64,...``。
    """
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
