# 常用命令（Windows 上没装 make 也能直接看下面每条命令的内容）

.PHONY: install install-all test lint eval serve corpus clean

install:
	uv sync --extra dev

install-all:
	uv sync --extra dev --extra ocr --extra visual

test:
	uv run pytest -q

lint:
	uv run ruff check .

fmt:
	uv run ruff check --fix .

corpus:
	uv run python -c "from pathlib import Path; from visioncite.eval.corpus_gen import generate_corpus; generate_corpus(Path('corpus'), per_template=2)"

eval:
	uv run python -m visioncite.eval.run --mode all

serve:
	uv run uvicorn visioncite.api.app:app --host 127.0.0.1 --port 8080

clean:
	rm -rf .pytest_cache .ruff_cache reports/*.json corpus
