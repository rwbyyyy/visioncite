"""命令行入口：``visioncite ingest`` / ``ask`` / ``serve``。

设计：CLI 只做参数解析与输出格式化，业务全部复用库函数，
这样命令行、HTTP、eval 三条入口共享同一套逻辑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from visioncite.answer.qa import answer_question
from visioncite.config import Settings
from visioncite.errors import VisionCiteError
from visioncite.logging import setup_logging
from visioncite.pipeline import build_document


def _ingest(args: argparse.Namespace, settings: Settings) -> int:
    """解析一份文档并打印摘要。

    Args:
        args: 命令行参数。
        settings: 配置。

    Returns:
        退出码。
    """
    document = build_document(Path(args.path), settings)
    print(f"doc_id : {document.meta.doc_id}")
    print(f"pages  : {document.meta.page_count}")
    print(f"chunks : {len(document.chunks)}")
    print(f"ocr    : {document.ocr_name} / visual: {document.visual_name}")
    for region in document.pages[0].regions[:10]:
        print(f"  {region.region_id} [{region.kind}] {region.text[:40]!r}")
    return 0


def _ask(args: argparse.Namespace, settings: Settings) -> int:
    """解析文档并回答一个问题。

    Args:
        args: 命令行参数。
        settings: 配置。

    Returns:
        退出码。
    """
    document = build_document(Path(args.path), settings)
    result = answer_question(document, args.question, settings)
    if result.abstained:
        print(f"弃答：{'；'.join(result.reasons)}")
        return 1
    print(f"答案：{result.answer}")
    print(f"置信度：{result.confidence:.2f}  后端：{result.backend}")
    for citation in result.citations:
        print(f"  第 {citation.page + 1} 页 {citation.bbox.to_list()} · {citation.snippet[:40]}")
    return 0


def _serve(args: argparse.Namespace, settings: Settings) -> int:
    """启动 HTTP 服务。

    Args:
        args: 命令行参数。
        settings: 配置。

    Returns:
        退出码。
    """
    import uvicorn

    uvicorn.run("visioncite.api.app:app", host=args.host, port=args.port, reload=False)
    return 0


def _agent(args: argparse.Namespace, settings: Settings) -> int:
    """用 Agent 回答并打印执行轨迹。

    Args:
        args: 命令行参数。
        settings: 配置。

    Returns:
        退出码。
    """
    from visioncite.agent.loop import run_agent
    from visioncite.llm.factory import create_provider
    from visioncite.llm.fake import FakeProvider

    document = build_document(Path(args.path), settings)
    provider = FakeProvider() if args.fake else create_provider(settings)
    result = run_agent(
        document, args.question, settings, provider=provider, max_steps=args.max_steps
    )
    if result.abstained:
        print(f"弃答：{'；'.join(result.reasons)}")
        return 1
    print(f"答案：{result.answer}")
    print(f"后端：{result.backend}  token：{result.usage.total_tokens}")
    for step in result.steps:
        print(f"  [工具] {step.name}({step.arguments}) -> {step.result[:80]}")
    for citation in result.citations:
        print(f"  证据：第 {citation.page + 1} 页 {citation.bbox.to_list()} · {citation.snippet[:40]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。

    Returns:
        ArgumentParser。
    """
    parser = argparse.ArgumentParser(prog="visioncite", description="多模态文档问答与证据溯源")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="解析文档并打印摘要")
    ingest.add_argument("path")
    ingest.set_defaults(handler=_ingest)

    ask = sub.add_parser("ask", help="解析文档并提问")
    ask.add_argument("path")
    ask.add_argument("question")
    ask.set_defaults(handler=_ask)

    serve = sub.add_parser("serve", help="启动 Web 服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(handler=_serve)

    agent = sub.add_parser("agent", help="用工具调用式 Agent 回答问题")
    agent.add_argument("path")
    agent.add_argument("question")
    agent.add_argument("--max-steps", type=int, default=4)
    agent.add_argument("--fake", action="store_true", help="强制使用 fake provider（无需 API key）")
    agent.set_defaults(handler=_agent)
    return parser


def main(argv: list[str] | None = None) -> int:
    """程序入口。

    Args:
        argv: 参数；None 用 ``sys.argv``。

    Returns:
        退出码。
    """
    setup_logging()
    settings = Settings.from_env()
    settings.ensure_dirs()
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args, settings))
    except VisionCiteError as exc:
        print(f"失败：{exc.user_message}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
