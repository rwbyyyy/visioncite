"""合成语料生成器：把模板渲染成 PNG，并写出带标注的 JSONL。

输出目录结构：
    corpus/
      images/invoice-0.png ...
      items.jsonl
      items.json            # 便于人看的汇总

字体：中文语料依赖系统 CJK 字体；找不到时直接报错并提示安装，
而不是静默生成"豆腐块"图片（那会让评测数字变成噪声）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from visioncite.errors import VisionCiteError
from visioncite.eval.dataset import QAItem, save_items
from visioncite.eval.templates import TEMPLATES, load_fonts
from visioncite.types import BBox

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


class CorpusFontError(VisionCiteError):
    """找不到中文字体，无法生成可读语料。"""

    default_user_message = "缺少中文字体：请安装 Noto Sans CJK 或使用 Windows 的 msyh.ttc。"


@dataclass
class GeneratedCorpus:
    """生成结果。

    Attributes:
        root: 语料根目录。
        images: 图片路径列表。
        items: 全部样本。
    """

    root: Path
    images: list[Path]
    items: list[QAItem]


def find_cjk_font() -> Path:
    """在常见路径里找中文字体。

    Returns:
        字体路径。

    Raises:
        CorpusFontError: 找不到。
    """
    for candidate in _FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    msg = "未找到中文字体，试过: " + ", ".join(_FONT_CANDIDATES)
    raise CorpusFontError(msg)


def generate_corpus(root: Path, per_template: int = 2) -> GeneratedCorpus:
    """生成整套语料。

    Args:
        root: 输出根目录（会在其下创建 images/）。
        per_template: 每类模板生成几份。

    Returns:
        GeneratedCorpus。

    Raises:
        CorpusFontError: 缺少中文字体。
    """
    font_path = find_cjk_font()
    fonts = load_fonts(font_path)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images: list[Path] = []
    items: list[QAItem] = []
    for category, builder in TEMPLATES.items():
        for index in range(per_template):
            image, rows = builder(index, fonts)
            path = images_dir / f"{category}-{index}.png"
            image.save(path)
            images.append(path)
            for q_index, row in enumerate(rows):
                items.append(
                    QAItem(
                        item_id=f"{category}-{index}-q{q_index}",
                        doc_file=f"images/{path.name}",
                        question=str(row["question"]),
                        answer=str(row["answer"]),
                        citation=_to_bbox(row.get("bbox")),
                        answerable=bool(row.get("answerable", True)),
                        category=category,
                    )
                )

    save_items(root / "items.jsonl", items)
    _write_summary(root / "items.json", items)
    return GeneratedCorpus(root=root, images=images, items=items)


def _to_bbox(value: object) -> BBox | None:
    """把模板给出的四元组转成 BBox。

    Args:
        value: 四元组或 None。

    Returns:
        BBox 或 None。
    """
    if not value:
        return None
    return BBox.from_list(list(value))  # type: ignore[arg-type]


def _write_summary(path: Path, items: list[QAItem]) -> None:
    """写一份人类可读的汇总。

    Args:
        path: 输出路径。
        items: 样本。
    """
    summary = {
        "total": len(items),
        "answerable": sum(1 for item in items if item.answerable),
        "by_category": {
            category: sum(1 for item in items if item.category == category)
            for category in sorted({item.category for item in items})
        },
        "items": [item.to_dict() for item in items],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def load_or_generate(root: Path, per_template: int = 2) -> GeneratedCorpus:
    """语料已存在就直接复用，避免每次评测都重新渲染。

    Args:
        root: 语料根目录。
        per_template: 首次生成时每类模板份数。

    Returns:
        GeneratedCorpus。
    """
    items_path = root / "items.jsonl"
    if items_path.is_file():
        from visioncite.eval.dataset import load_items

        items = load_items(items_path)
        images = [root / item.doc_file for item in items]
        unique_images = list(dict.fromkeys(images))
        return GeneratedCorpus(root=root, images=unique_images, items=items)
    return generate_corpus(root, per_template=per_template)
