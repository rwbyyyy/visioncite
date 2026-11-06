"""评测数据集：定义样本格式，负责 JSONL 读写。

为什么自己定格式？
    公开 VQA 数据集的字段千奇百怪（有的给答案列表、有的给多个标注者），
    本项目只关心三件事：问题、标准答案、证据框。
    固定成一个小 schema，评测代码就能保持简单，也方便你以后接自己的数据。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from visioncite.types import BBox


@dataclass(frozen=True)
class QAItem:
    """一条评测样本。

    Attributes:
        item_id: 样本 id。
        doc_file: 相对语料目录的图片路径。
        question: 问题。
        answer: 标准答案（抽取式，要求是证据里的原句片段）。
        citation: 标准证据框；不可答样本为 None。
        answerable: 是否可答。
        category: 类别（invoice/table/chart/slide）。
    """

    item_id: str
    doc_file: str
    question: str
    answer: str
    citation: BBox | None
    answerable: bool
    category: str

    def to_dict(self) -> dict[str, object]:
        """序列化为可写 JSON 的字典。

        Returns:
            字典。
        """
        return {
            "id": self.item_id,
            "file": self.doc_file,
            "question": self.question,
            "answer": self.answer,
            "citation": self.citation.to_list() if self.citation else None,
            "answerable": self.answerable,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> QAItem:
        """从 JSON 字典恢复。

        Args:
            data: 字典。

        Returns:
            QAItem。

        Raises:
            KeyError: 缺少必填字段。
        """
        citation_raw = data.get("citation")
        citation = BBox.from_list(citation_raw) if citation_raw else None  # type: ignore[arg-type]
        return cls(
            item_id=str(data["id"]),
            doc_file=str(data["file"]),
            question=str(data["question"]),
            answer=str(data.get("answer", "")),
            citation=citation,
            answerable=bool(data.get("answerable", True)),
            category=str(data.get("category", "unknown")),
        )


def save_items(path: Path, items: list[QAItem]) -> None:
    """把样本写成 JSONL。

    Args:
        path: 文件路径。
        items: 样本列表。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def load_items(path: Path) -> list[QAItem]:
    """读取 JSONL 样本。

    Args:
        path: 文件路径。

    Returns:
        样本列表。
    """
    items: list[QAItem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(QAItem.from_dict(json.loads(line)))
    return items
