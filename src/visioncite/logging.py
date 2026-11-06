"""日志：给线上排查用的最小设施。

原则：
1. 用标准库 ``logging``，不引第三方依赖；
2. 格式统一为 ``时间 级别 模块: 消息``，便于 grep；
3. 只记录"发生了什么"和"耗时"，不记录文档内容与用户问题（隐私）。
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根 logger，重复调用不会重复加 handler。

    Args:
        level: 日志级别，默认 INFO。
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("visioncite")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """取得带命名空间的 logger。

    Args:
        name: 子模块名，如 ``"ingest.pdf"``。

    Returns:
        ``visioncite.ingest.pdf`` 对应的 Logger。
    """
    return logging.getLogger(f"visioncite.{name}")
