"""合成文档模板：四类可精确标注的页面。

为什么自己画语料？
    公开数据集（DocVQA/ChartQA）动辄几个 GB，且标注框风格不一，
    在 CPU-only 环境里跑不动。自建语料的优势是**标准答案与证据框完全已知**，
    能做严格的引用 IoU 评测；代价是分布简单，这一条会在 LIMITATIONS 里说清楚。

每个模板返回：
    image: PIL 图
    items: [{"question", "answer", "bbox", "answerable"}, ...]
其中 bbox 用绘制时记录的 textbbox 计算，保证与画面严格对齐。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Canvas = tuple[Image.Image, ImageDraw.ImageDraw]
Item = dict[str, object]

PAGE_SIZE = (1000, 1300)
MARGIN = 70
BLACK = (20, 20, 20)
GRID = (90, 90, 90)
BLUE = (58, 110, 165)
LIGHT = (235, 240, 247)


@dataclass(frozen=True)
class Fonts:
    """一组字体，避免每个模板重复加载。

    Attributes:
        title: 标题字号。
        body: 正文字号。
        small: 小字号（表格单元格）。
    """

    title: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont


def load_fonts(font_path: Path) -> Fonts:
    """加载三档字号。

    Args:
        font_path: TTF/TTC 路径。

    Returns:
        Fonts。
    """
    return Fonts(
        title=ImageFont.truetype(str(font_path), 40),
        body=ImageFont.truetype(str(font_path), 26),
        small=ImageFont.truetype(str(font_path), 23),
    )


def new_canvas() -> Canvas:
    """创建空白页面。

    Returns:
        ``(image, draw)``。
    """
    image = Image.new("RGB", PAGE_SIZE, "white")
    return image, ImageDraw.Draw(image)


def draw_line(
    canvas: Canvas,
    text: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = BLACK,
) -> tuple[int, int, int, int]:
    """写一行文字并返回其外接框。

    Args:
        canvas: 画布。
        text: 文本。
        x: 左上 x。
        y: 左上 y。
        font: 字体。
        fill: 颜色。

    Returns:
        ``(x0, y0, x1, y1)``。
    """
    _image, draw = canvas
    draw.text((x, y), text, font=font, fill=fill)
    return draw.textbbox((x, y), text, font=font)


def invoice(seed: int, fonts: Fonts) -> tuple[Image.Image, list[Item]]:
    """采购发票模板。

    Args:
        seed: 随机种子（用于编号）。
        fonts: 字体。

    Returns:
        ``(image, items)``。
    """
    canvas = new_canvas()
    image, draw = canvas
    number = f"INV-2026-{100 + seed:03d}"

    draw_line(canvas, "采购发票", MARGIN, 70, fonts.title)
    boxes = {
        "number": draw_line(canvas, f"发票号码：{number}", MARGIN, 170, fonts.body),
        "date": draw_line(canvas, "开票日期：2026-10-25", MARGIN, 230, fonts.body),
        "buyer": draw_line(canvas, "购买方：浙江大学", MARGIN, 290, fonts.body),
        "qty": draw_line(canvas, "数量：2 台", MARGIN, 400, fonts.body),
        "price": draw_line(canvas, "单价：643.20 元", MARGIN, 460, fonts.body),
        "total": draw_line(canvas, "合计金额：1,286.40 元", MARGIN, 560, fonts.body),
        "note": draw_line(canvas, "备注：设备验收后 30 日内付款", MARGIN, 700, fonts.small),
    }
    items: list[Item] = [
        {"question": "发票号码是多少？", "answer": number, "bbox": boxes["number"]},
        {"question": "开票日期是哪一天？", "answer": "2026-10-25", "bbox": boxes["date"]},
        {"question": "合计金额是多少元？", "answer": "1,286.40", "bbox": boxes["total"]},
        {"question": "购买方是谁？", "answer": "浙江大学", "bbox": boxes["buyer"]},
        {"question": "这张发票的税额是多少？", "answer": "", "bbox": None, "answerable": False},
    ]
    return image, items


def table(seed: int, fonts: Fonts) -> tuple[Image.Image, list[Item]]:
    """带框线的销售明细表。

    Args:
        seed: 随机种子（改变数字）。
        fonts: 字体。

    Returns:
        ``(image, items)``。
    """
    canvas = new_canvas()
    image, draw = canvas
    draw_line(canvas, "2026 年第三季度销售明细", MARGIN, 70, fonts.title)

    left, top = MARGIN, 200
    col_x = [left, left + 260, left + 450, left + 640]
    right = col_x[-1] + 220
    row_h = 70
    rows = [
        ["产品", "数量", "单价", "小计"],
        ["显示器", "12", "899.00", "10788.00"],
        ["键盘", "30", "129.00", "3870.00"],
        ["打印机", "2", "643.20", "1286.40"],
        ["合计", "-", "-", "15944.40"],
    ]
    draw.rectangle([left, top, right, top + row_h * len(rows)], outline=GRID, width=2)
    for r in range(len(rows)):
        y = top + r * row_h
        if r:
            draw.line([left, y, right, y], fill=GRID, width=2)
        for c, x in enumerate(col_x):
            if c:
                draw.line([x, top, x, top + row_h * len(rows)], fill=GRID, width=2)
            draw_line(canvas, rows[r][c], x + 14, y + 16, fonts.small)

    boxes = {
        "table": (left, top, right, top + row_h * len(rows)),
    }
    draw_line(canvas, "制表：财务部   核对：王工", MARGIN, 620, fonts.small)
    items: list[Item] = [
        {"question": "显示器卖了多少台？", "answer": "12", "bbox": boxes["table"]},
        {"question": "打印机的单价是多少？", "answer": "643.20", "bbox": boxes["table"]},
        {"question": "合计金额是多少元？", "answer": "15944.40", "bbox": boxes["table"]},
        {"question": "哪个产品的退货最多？", "answer": "", "bbox": None, "answerable": False},
    ]
    return image, items


def chart(seed: int, fonts: Fonts) -> tuple[Image.Image, list[Item]]:
    """柱状图 + 数据说明。

    Args:
        seed: 随机种子。
        fonts: 字体。

    Returns:
        ``(image, items)``。
    """
    canvas = new_canvas()
    image, draw = canvas
    draw_line(canvas, "月度访问量（万次）", MARGIN, 70, fonts.title)

    months = ["7 月", "8 月", "9 月", "10 月"]
    values = [12, 18, 25, 21]
    base_y = 720
    bar_w = 120
    gap = 60
    max_val = 30
    for i, (month, value) in enumerate(zip(months, values, strict=True)):
        x = 150 + i * (bar_w + gap)
        height = int(value / max_val * 500)
        draw.rectangle([x, base_y - height, x + bar_w, base_y], fill=BLUE)
        draw_line(canvas, str(value), x + 35, base_y - height - 40, fonts.body)
        draw_line(canvas, month, x + 20, base_y + 16, fonts.small)
    draw.line([100, base_y, 930, base_y], fill=GRID, width=3)

    caption = draw_line(canvas, "10 月访问量：21 万次", MARGIN, 850, fonts.body)
    caption2 = draw_line(canvas, "9 月访问量：25 万次", MARGIN, 910, fonts.body)
    caption3 = draw_line(canvas, "8 月访问量：18 万次", MARGIN, 970, fonts.body)
    items: list[Item] = [
        {"question": "10 月的访问量是多少？", "answer": "21 万次", "bbox": caption},
        {"question": "9 月的访问量是多少？", "answer": "25 万次", "bbox": caption2},
        {"question": "8 月的访问量是多少？", "answer": "18 万次", "bbox": caption3},
        {"question": "访问量的同比增速是多少？", "answer": "", "bbox": None, "answerable": False},
    ]
    return image, items


def slide(seed: int, fonts: Fonts) -> tuple[Image.Image, list[Item]]:
    """汇报幻灯片模板。

    Args:
        seed: 随机种子。
        fonts: 字体。

    Returns:
        ``(image, items)``。
    """
    canvas = new_canvas()
    image, draw = canvas
    draw.rectangle([0, 0, PAGE_SIZE[0], 140], fill=BLUE)
    draw_line(canvas, "VisionCite：多模态文档问答", MARGIN, 40, fonts.title, fill=(255, 255, 255))

    lines = [
        "检索：BM25 与 CLIP 双路召回",
        "定位：答案引用精确到区域框",
        "评测：36 条样本，ECE 0.19",
        "部署：FastAPI 与单页前端",
    ]
    boxes = []
    for i, text in enumerate(lines):
        y = 240 + i * 120
        draw.rectangle([MARGIN, y + 12, MARGIN + 12, y + 24], fill=BLUE)
        boxes.append(draw_line(canvas, text, MARGIN + 40, y, fonts.body))

    draw_line(canvas, "汇报人：rwbyyyy    2026-11", MARGIN, 800, fonts.small)
    items: list[Item] = [
        {"question": "检索部分用了什么方法？", "answer": "BM25 与 CLIP", "bbox": boxes[0]},
        {"question": "答案引用精确到什么？", "answer": "区域框", "bbox": boxes[1]},
        {"question": "ECE 是多少？", "answer": "0.19", "bbox": boxes[2]},
        {"question": "部署用了什么框架？", "answer": "FastAPI", "bbox": boxes[3]},
        {"question": "项目的预算缺口是多少？", "answer": "", "bbox": None, "answerable": False},
    ]
    return image, items


TEMPLATES = {
    "invoice": invoice,
    "table": table,
    "chart": chart,
    "slide": slide,
}
