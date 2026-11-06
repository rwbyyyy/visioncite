"""弃答策略：知道自己"不知道"，比多答对一题更重要。

三档判断（顺序即优先级）：
1. 没有任何证据 → 弃答（检索没找到，硬答一定是编）；
2. 校验不通过 → 弃答（有证据但答案没被支撑住）；
3. 置信度低于阈值 → 弃答（证据弱、歧义大）。

阈值来自配置，评测脚本会扫多个阈值并画"答题率-准确率"曲线，
最终选一个业务上可接受的折中（宁可少答，不可答错）。
"""

from __future__ import annotations

from visioncite.answer.verify import Verification
from visioncite.config import Settings


def should_abstain(
    confidence: float,
    citation_count: int,
    verification: Verification,
    settings: Settings,
) -> tuple[bool, tuple[str, ...]]:
    """判断是否弃答。

    Args:
        confidence: 0~1 的置信度。
        citation_count: 引用数量（0 表示没找到证据）。
        verification: 校验结果。
        settings: 配置（含弃答阈值）。

    Returns:
        ``(是否弃答, 原因列表)``。
    """
    if citation_count == 0:
        return True, ("没有检索到可用证据",)

    if not verification.ok:
        return True, verification.reasons

    if confidence < settings.abstain_threshold:
        return True, (f"置信度 {confidence:.2f} 低于阈值 {settings.abstain_threshold:.2f}",)

    return False, ()
