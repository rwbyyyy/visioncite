"""生成一份"车辆用户手册"合成语料（12 页）+ 20 条问答，供 RAG 实验使用。

为什么自建：真实车主手册有版权、页数多、OCR 噪声不可控；
自建语料的价值是**答案与证据框完全已知**，能严格测"引用命中"。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
FONT = r"C:\Windows\Fonts\msyh.ttc"
PAGE = (1000, 1300)
MARGIN = 70
INK = (20, 20, 20)
BLUE = (30, 80, 140)


def font(size: int) -> ImageFont.FreeTypeFont:
    """加载中文字体。"""
    return ImageFont.truetype(FONT, size)


def draw_line(draw: ImageDraw.ImageDraw, text: str, y: int, size: int, color=INK):
    """画一行字并返回 bbox。"""
    f = font(size)
    draw.text((MARGIN, y), text, font=f, fill=color)
    return list(draw.textbbox((MARGIN, y), text, font=f))


PAGES = [
    {
        "title": "空调系统概览",
        "lines": [
            "本车配备双区自动空调，主驾与副驾可独立设定温度。",
            "空调控制面板位于中控屏下方，也可通过语音指令操作。",
            "制冷剂型号为 R134a，加注量请参考保养手册。",
            "空调滤芯建议每 12 个月或 2 万公里更换一次。",
        ],
        "qa": [
            ("空调滤芯多久更换一次？", 3),
            ("制冷剂是什么型号？", 2),
        ],
    },
    {
        "title": "空调温度设置",
        "lines": [
            "温度可在 16 摄氏度到 32 摄氏度之间调节，步进 0.5 度。",
            "按下 AUTO 键后，系统会根据车外温度自动选择风量。",
            "长按温度键 2 秒可快速切换至最低或最高温度。",
            "同步模式 SYNC 可让副驾温度跟随主驾设置。",
        ],
        "qa": [
            ("空调温度可以调到多少度？", 0),
            ("SYNC 键有什么用？", 3),
        ],
    },
    {
        "title": "风量与除霜",
        "lines": [
            "风量共 7 档，第 7 档为最大风量。",
            "前风挡除霜按键位于面板左侧，开启后自动切换外循环。",
            "后风挡除霜由电加热丝完成，10 分钟后自动关闭。",
            "长时间使用内循环可能导致车内空气浑浊，建议定期切换。",
        ],
        "qa": [
            ("风量最大是几档？", 0),
            ("后风挡除霜多久自动关闭？", 2),
        ],
    },
    {
        "title": "座椅加热与通风",
        "lines": [
            "主驾座椅加热分三档，开启后约 3 分钟达到设定温度。",
            "副驾座椅通风与加热不能同时开启。",
            "座椅记忆功能可保存两组坐姿，按键位于门板。",
            "儿童座椅建议安装在后排，并关闭对应位置的安全气囊。",
        ],
        "qa": [
            ("座椅加热有几档？", 0),
            ("座椅记忆可以保存几组坐姿？", 2),
        ],
    },
    {
        "title": "车窗与天窗",
        "lines": [
            "四门车窗均支持一键升降，长按钥匙锁车键可关闭全部车窗。",
            "天窗开启前需确认车顶无障碍物。",
            "防夹功能触发后，车窗会自动下降约 10 厘米。",
            "车速超过 80 公里每小时时，天窗会自动减小开度。",
        ],
        "qa": [
            ("车窗防夹触发后会怎样？", 2),
            ("车速多少时天窗会自动减小开度？", 3),
        ],
    },
    {
        "title": "灯光控制",
        "lines": [
            "灯光拨杆位于方向盘左侧，AUTO 档会根据环境光自动开关大灯。",
            "雾灯需要在大灯开启的前提下才能打开。",
            "转向灯闪烁频率异常通常表示灯泡故障。",
            "离家照明功能可在大灯关闭后延时 30 秒熄灭。",
        ],
        "qa": [
            ("雾灯在什么前提下才能打开？", 1),
            ("离家照明延时多久？", 3),
        ],
    },
    {
        "title": "导航与地图更新",
        "lines": [
            "车载导航支持在线地图与离线地图两种模式。",
            "离线地图每季度更新一次，可在设置中手动检查。",
            "导航语音提示音量可独立于媒体音量调节。",
            "目的地可通过手机 App 发送到车机。",
        ],
        "qa": [
            ("离线地图多久更新一次？", 1),
            ("导航语音音量可以单独调吗？", 2),
        ],
    },
    {
        "title": "多媒体与蓝牙电话",
        "lines": [
            "蓝牙配对最多可保存 8 台设备，同时连接 2 台。",
            "来电时短按方向盘按键接听，长按挂断。",
            "USB 接口支持最大 64GB 的 U 盘。",
            "音效模式提供标准、流行、古典三种预设。",
        ],
        "qa": [
            ("蓝牙最多能保存几台设备？", 0),
            ("U 盘最大支持多大容量？", 2),
        ],
    },
    {
        "title": "驾驶辅助系统",
        "lines": [
            "自适应巡航可在 30 到 150 公里每小时之间启用。",
            "车道保持辅助在车速高于 60 公里每小时时激活。",
            "前碰撞预警的灵敏度可在设置中选择三档。",
            "驾驶辅助只是辅助，驾驶员需始终保持对车辆的控制。",
        ],
        "qa": [
            ("自适应巡航的启用速度范围是多少？", 0),
            ("车道保持辅助在多少时速以上激活？", 1),
        ],
    },
    {
        "title": "充电与能耗",
        "lines": [
            "建议将动力电池电量保持在 20% 到 80% 之间以延长寿命。",
            "快充从 30% 充到 80% 约需 30 分钟。",
            "冬季低温会降低续航，建议出发前使用电池预热功能。",
            "长期停放时建议保持电量在 50% 左右。",
        ],
        "qa": [
            ("快充从 30% 到 80% 需要多久？", 1),
            ("长期停放建议电量保持多少？", 3),
        ],
    },
    {
        "title": "保养周期",
        "lines": [
            "首保为购车后 6 个月或 5000 公里，以先到者为准。",
            "常规保养间隔为 12 个月或 1 万公里。",
            "制动液每 2 年更换一次。",
            "空调滤芯与空气滤芯的更换周期不同，请分别查阅。",
        ],
        "qa": [
            ("首保的时间和里程是多少？", 0),
            ("制动液多久更换一次？", 2),
        ],
    },
    {
        "title": "故障码与应急处理",
        "lines": [
            "仪表盘出现红色故障灯时，应立即靠边停车并联系售后。",
            "黄色故障灯表示需要尽快检查，可继续低速行驶。",
            "随车工具包括三角警示牌、拖车钩和补胎液。",
            "紧急情况下可长按 SOS 按键 3 秒呼叫救援。",
        ],
        "qa": [
            ("黄色故障灯代表什么？", 1),
            ("紧急救援按键需要长按多久？", 3),
        ],
    },
]

UNANSWERABLE = [
    "这辆车的后备箱容积是多少升？",
    "车辆支持哪些 OTA 升级内容？",
    "轮胎的推荐胎压是多少？",
    "车载音响有几个扬声器？",
]


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for page_index, page in enumerate(PAGES):
        image = Image.new("RGB", PAGE, "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, PAGE[0], 110], fill=BLUE)
        title_font = font(38)
        draw.text((MARGIN, 34), page["title"], font=title_font, fill=(255, 255, 255))
        boxes = []
        y = 170
        for line in page["lines"]:
            boxes.append(draw_line(draw, line, y, 25))
            y += 78
        draw.text((MARGIN, y + 20), f"第 {page_index + 1} 页 · 车辆用户手册", font=font(20), fill=(120, 120, 120))
        path = IMAGES / f"page-{page_index + 1:02d}.png"
        image.save(path)
        for order, (question, line_index) in enumerate(page["qa"]):
            items.append(
                {
                    "id": f"p{page_index + 1}-q{order}",
                    "file": f"images/{path.name}",
                    "question": question,
                    "answer": page["lines"][line_index],
                    "citation": boxes[line_index],
                    "answerable": True,
                    "category": "manual",
                }
            )
    for order, question in enumerate(UNANSWERABLE):
        items.append(
            {
                "id": f"u{order}",
                "file": "images/page-01.png",
                "question": question,
                "answer": "",
                "citation": None,
                "answerable": False,
                "category": "manual",
            }
        )
    (ROOT / "items.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8"
    )
    print(f"pages={len(PAGES)} items={len(items)}")


if __name__ == "__main__":
    main()
