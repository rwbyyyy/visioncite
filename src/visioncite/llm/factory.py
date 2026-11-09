"""LLM 工厂：按配置选提供方。

无配置时退回 fake，并打一条 warning：
这样 CI、演示、单测都不需要 key，也不会因为"忘了配 key"而崩。
"""

from __future__ import annotations

from visioncite.config import Settings
from visioncite.llm.base import LLMProvider
from visioncite.llm.fake import FakeProvider
from visioncite.llm.openai_compat import OpenAICompatProvider
from visioncite.logging import get_logger

logger = get_logger("llm.factory")


def create_provider(settings: Settings) -> LLMProvider:
    """按配置创建模型提供方。

    Args:
        settings: 运行配置（llm_base_url / llm_model）。

    Returns:
        真实客户端或 fake。
    """
    if not (settings.llm_base_url and settings.llm_model):
        logger.warning("未配置 LLM，使用 fake provider（工具链路仍可完整演示）")
        return FakeProvider()
    return OpenAICompatProvider(settings.llm_base_url, settings.llm_model)
