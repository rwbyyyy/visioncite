# VisionCite

**Multimodal document QA with verifiable visual citations.**
Upload a PDF or an image, ask a question in natural language, and get an answer that
points back to the exact page region it came from — or an explicit refusal when the
evidence is not there.

> 中文说明见 [README.zh-CN.md](README.zh-CN.md)。

```
PDF / image ──▶ layout (XY-cut + table rules) ──▶ OCR (RapidOCR) ──▶ chunks
                                                                     │
question ──▶ BM25 + CLIP retrieval ──▶ RRF + MMR ──▶ answer ──▶ verify ──▶ cite / abstain
```

## Why this exists

Most "multimodal RAG" demos stop at *"the model said something plausible"*. VisionCite
adds the part that matters in production:

- **Evidence is geometry, not prose.** Every citation carries a bounding box; the UI draws
  it on the page, and the evaluation measures citation IoU against gold boxes.
- **Answers are checked before they are shown.** Citations must exist, every number in the
  answer must appear in the cited text, and lexical support must clear a threshold —
  otherwise the system abstains.
- **The retrieval path is inspectable.** BM25 over Chinese character bigrams, CLIP visual
  recall, RRF fusion without score-scale assumptions, MMR de-duplication.
- **Degrades instead of crashing.** No OCR → PDF text layer only. No CLIP → text-only
  retrieval. No VLM → extractive answers. Every state is reported by `/health`.
- **Answers visual attributes too.** "What colour is the text / the background / the seal?"
  is answered by pixel statistics on the cited region (HSV bucketing), not by guessing
  from OCR text — and the seal question first locates the red region before measuring it.

## Quick start

```bash
# Python 3.10+
pip install -e ".[dev]"          # core + tests
pip install -e ".[ocr,visual]"   # optional: offline OCR and CLIP retrieval

pytest -q                        # 82 tests, fully offline
uvicorn visioncite.api.app:app --port 8080
# open http://127.0.0.1:8080
```

CLI:

```bash
visioncite ingest path/to/doc.pdf
visioncite ask path/to/doc.pdf "合计金额是多少？"
visioncite serve --port 8080
```

First run of the OCR path loads RapidOCR (bundled ONNX models). The CLIP path downloads
`openai/clip-vit-base-patch32` (~600 MB) on first use; set `HF_ENDPOINT` if you need a mirror.

## Measured results

Synthetic benchmark, 36 questions over 8 rendered pages (invoices, tables, charts, slides),
CPU-only, `python -m visioncite.eval.run --mode all`. Full report: [`reports/`](reports/).

| Mode | Coverage | AnswerAcc | Cite-P | Cite-R | AbstainAcc | ECE | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| page (no layout) | 0.83 | 0.62 | 0.00 | 0.00 | 0.83 | 0.34 | 1 ms | 1 ms |
| layout (XY-cut + BM25) | 0.78 | 1.00 | 0.85 | 1.00 | 0.89 | 0.17 | 0 ms | 2 ms |
| hybrid (+ CLIP) | 0.78 | 1.00 | 0.85 | 1.00 | 0.89 | 0.17 | 30 ms | 33 ms |

Numbers instead of whole lines: the extractive backend now answers with the **value**
("88", "21 万次") when the question asks for one, using the table header to pick the
right column and skipping ID-like codes such as `INV-2026-100`.

Reading these honestly:

- `AnswerAcc = 1.00` is the **answered subset**; the coverage column shows 22 % of questions
  were refused. Refusing is a feature, not a failure.
- `page` mode cannot localise evidence (Cite-P/R = 0) — that column is the value of layout.
- CLIP adds **latency without quality gain** on this synthetic corpus: the queries are
  text-heavy and CLIP cannot read small print. On chart/figure questions it would matter;
  the benchmark is too small to prove it.
- 95 % bootstrap intervals collapse to `[1.00, 1.00]` because every answered question is
  correct — with n=24 that interval has no discriminating power. Do not read it as stability.

The benchmark is synthetic: clean backgrounds, upright pages, no handwriting. It is a
regression harness, not a claim about real scanned documents (see [docs/LIMITATIONS.md](docs/LIMITATIONS.md)).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/documents` | upload PDF/image → `doc_id`, pages, regions |
| POST | `/api/ask` | `{doc_id, question}` → answer + citations + confidence |
| POST | `/api/agent` | tool-calling agent → answer + citations + execution trace |
| GET | `/api/documents/{id}/pages/{n}.png` | page image |
| GET | `/api/documents/{id}/regions/{rid}.crop` | evidence crop |
| GET | `/health` | which backends are actually active |

## Design notes

- **Agent (v0.2)** — a minimal tool-calling loop: the model chooses tools, the runtime executes
  them, results are fed back, and citations come only from `chunk_id`s tools actually returned.
  Offline demo: `visioncite agent corpus/images/invoice-0.png "合计金额是多少？" --fake`.
- **MCP (v0.2)** — the same tools are exposed over MCP (stdio + JSON-RPC 2.0): `initialize`,
  `tools/list`, `tools/call`, `ping`. Deliberately a subset; see [docs/AGENT_MCP.md](docs/AGENT_MCP.md).

- **Layout** — Otsu binarisation, recursive XY-cut on ink projections, table detection from
  page-spanning rule lines. No deep model, no GPU, fully explainable; limitation is skew and
  borderless tables.
- **Retrieval** — Chinese text is indexed as character bigrams (no dictionary, robust to OOV);
  BM25 gives lexical precision, CLIP gives semantic/visual recall; scores are fused by rank
  (RRF, k=60) because BM25 scores and cosine similarities are not comparable; MMR removes
  near-duplicate evidence.
- **Grounded answering** — the extractive backend selects the best supporting sentence and
  cites its region; the optional VLM backend must return JSON with chunk ids from a whitelist.
  Both go through the same verifier.
- **Refusal** — three gates: no evidence, verification failure, confidence below threshold.
  A wrong answer is worse than no answer in document workflows.

## Docs

| File | Contents |
|---|---|
| [docs/CURRICULUM.md](docs/CURRICULUM.md) | 12-lesson course: theory → code → hand-worked example → exercises |
| [docs/CODE_TOUR.md](docs/CODE_TOUR.md) | 8-stop reading path through the codebase |
| [docs/EXERCISES.md](docs/EXERCISES.md) | 12 hands-on changes with acceptance criteria |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | layering, data flow, production gaps |
| [docs/AGENT_MCP.md](docs/AGENT_MCP.md) | tool-calling loop, fake provider, MCP subset |
| [docs/EVALUATION.md](docs/EVALUATION.md) | metric definitions and how to reproduce |
| [docs/INTERVIEW.md](docs/INTERVIEW.md) | 30 deep-dive Q&A with code pointers |
| [docs/RESUME.md](docs/RESUME.md) | resume bullets with explicit caveats |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | what this system cannot do |

## What is intentionally out of scope

Model training, multi-tenancy, authentication, crawling, and video timelines.
See [ROADMAP.md](ROADMAP.md).

## License

MIT — see [LICENSE](LICENSE).
