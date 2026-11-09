"""OpenAI 兼容的模型客户端（对话 + 工具调用）。

只依赖 httpx，不引厂商 SDK：任何提供 ``/chat/completions`` 的服务
（DeepSeek、Qwen、vLLM、Ollama 的兼容层…）都能接。

诚实说明：
    - 该客户端**未在真实厂商端点上验证**（仓库环境没有 key）；
    - 只实现非流式对话；流式输出见 ROADMAP（对 Agent 不是必需）。
"""

from __future__ import annotations

import json
import os
import time

import httpx

from visioncite.errors import LLMError
from visioncite.llm.base import ChatMessage, LLMResponse, ToolCall, Usage
from visioncite.logging import get_logger

logger = get_logger("llm.openai")


class OpenAICompatProvider:
    """OpenAI 兼容端点客户端。"""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        """初始化。

        Args:
            base_url: 例如 ``https://api.deepseek.com/v1``。
            model: 模型名。
            api_key: 密钥；None 时从环境变量 ``VISIONCITE_LLM_API_KEY`` 读取。
            timeout_s: 超时秒数。
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("VISIONCITE_LLM_API_KEY", "")
        self._timeout = timeout_s

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
    ) -> LLMResponse:
        """执行一次对话。

        Args:
            messages: 对话历史。
            tools: OpenAI function-calling 工具定义。

        Returns:
            LLMResponse。

        Raises:
            LLMError: 未配置 key、网络失败、响应格式非法。
        """
        if not self._api_key:
            msg = "未配置 VISIONCITE_LLM_API_KEY"
            raise LLMError(msg)

        payload: dict[str, object] = {
            "model": self._model,
            "messages": [_to_wire(message) for message in messages],
            "temperature": 0,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": tool} for tool in tools]

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"模型请求失败: {exc}"
            logger.warning(msg)
            raise LLMError(msg) from exc

        return self._parse(response.json(), started)

    def _parse(self, body: dict, started: float) -> LLMResponse:
        """解析响应。

        Args:
            body: 响应 JSON。
            started: 请求开始时间。

        Returns:
            LLMResponse。

        Raises:
            LLMError: 结构不符合 OpenAI 约定。
        """
        try:
            choice = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = "模型响应缺少 choices"
            raise LLMError(msg) from exc

        calls = tuple(_parse_call(item) for item in choice.get("tool_calls") or [])
        usage = body.get("usage") or {}
        return LLMResponse(
            message=ChatMessage(role="assistant", content=choice.get("content") or "", tool_calls=calls),
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_ms=(time.perf_counter() - started) * 1000,
            ),
        )


def _to_wire(message: ChatMessage) -> dict[str, object]:
    """把内部消息转成 OpenAI 线上格式。

    Args:
        message: 内部消息。

    Returns:
        dict。
    """
    wire: dict[str, object] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        wire["tool_call_id"] = message.tool_call_id
    if message.name:
        wire["name"] = message.name
    return wire


def _parse_call(item: dict) -> ToolCall:
    """解析一个 tool_call。

    Args:
        item: 响应里的 tool_call 对象。

    Returns:
        ToolCall。
    """
    function = item.get("function") or {}
    raw_arguments = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = {}
    return ToolCall(
        call_id=str(item.get("id", "")),
        name=str(function.get("name", "")),
        arguments=arguments if isinstance(arguments, dict) else {},
    )
