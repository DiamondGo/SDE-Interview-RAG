# SDE Interview RAG

[English](README.md) | [中文](README.zh-CN.md)

A RAG (retrieval-augmented generation) system built to help software engineers prepare for interviews.

It collects interview questions from question repositories and interview experiences from Hacker News and
dev.to, cleans and tags them (languages, frameworks, system design, AI/ML, automation, CS fundamentals,
algorithms, behavioral), indexes them for hybrid search, and answers two kinds of requests with cited sources:

- **What gets asked about a topic**: "What are common Go concurrency interview questions?" lists the questions
  that show up most often, grouped by subtopic, with how often and where each one was seen.
- **How to answer a specific question**: "How do I design a rate limiter?" gives an explanation, approach,
  code or key points, trade-offs and likely follow-ups. Claims about real interviews cite their sources;
  anything the sources do not support is labelled as model-supplied.

Questions can be asked in English or Chinese; answers come back in the language of the question.

The workflows are standard [LangGraph](https://github.com/langchain-ai/langgraph) graphs registered in
`langgraph.json` (`ingest`, `query`, `evaluate`), so any LangGraph runtime can run them. This project's CLI
and web UI are one way to run them.

- Design: [docs/DESIGN.md](docs/DESIGN.md) ([中文](docs/DESIGN.zh-CN.md))
- Progress: [docs/PROGRESS.md](docs/PROGRESS.md) ([中文](docs/PROGRESS.zh-CN.md))
- Published data: [dataset/NOTICE.md](dataset/NOTICE.md)

## How it works

```text
ingest (scheduled, every 12 h)
  sources -> list what changed -> download only new/changed -> clean -> extract Q&A -> tag (rules, then LLM)
          -> merge duplicates (frequency, "heat") -> chunks + BM25 index + embeddings

query (web UI / CLI / any LangGraph client)
  understand (intent, filters, English search terms) -> hybrid retrieval (BM25 + vectors, RRF)
          -> list common questions | write a cited answer -> verify citations -> log
```

Main properties:

- **Incremental crawling**: a second run downloads nothing that did not change (version checks, conditional
  requests, content hashes, cross-source link detection).
- **Everything persisted**: raw downloads, cleaned text, questions, tags, the FTS index and the vectors live in
  SQLite; a restart never re-embeds.
- **Configurable agents**: each role (router, tagger, generator, reviewer, ...) is mapped to a backend in
  configuration: any OpenAI-compatible endpoint or a CLI agent (`claude`, `agy`, Cursor `agent`), with
  fallbacks.
- **Stall detection**: a slow model is fine; a stuck one (no progress, runaway repetition) is abandoned and the
  next backend takes over.
- **Access control interfaces** are in place (not enforced yet) so that logins and per-user visibility can be
  added without touching retrieval.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, to get the published raw files, [Git LFS](https://git-lfs.com/).

```bash
git clone https://github.com/DiamondGo/SDE-Interview-RAG.git && cd SDE-Interview-RAG
git lfs pull                                      # the published raw files (dataset/raw/)
uv sync
cp config/agents.example.toml config/agents.toml  # model configuration: edit for your endpoints
cp .env.example .env                              # endpoint URLs and keys referenced by agents.toml
uv run sde-rag import-data                        # start from the published dataset: downloads the database (optional)
uv run sde-rag doctor                             # configuration problems, and does every backend answer?
```

The first run downloads the embedding model configured in `agents.toml` (by default
`intfloat/multilingual-e5-large`, about 2.2 GB) into `data/models/`.

## Configuration

Configuration is layered: `config/<name>.toml`, then `config/<name>.local.toml`, then environment variables.

| File | Committed | What | Env override |
|---|---|---|---|
| `agents.toml` | no (copy `agents.example.toml`) | Backends, which backend serves each role (with fallbacks and stall timeouts), the embedding model | `SDE_BACKEND__<NAME>__<FIELD>`, `SDE_ROLE__<NAME>__<FIELD>`, `SDE_MODEL__EMBEDDER__<FIELD>` |
| `settings.toml` | yes | Data directory, extra dotenv files, alert channels, HTTP politeness, per-run limits | `SDE_SETTING__<SECTION>__<FIELD>` |
| `sources.toml` | yes | Connectors: question repositories, Hacker News, dev.to; `publish = false` keeps a source out of the dataset | `SDE_SOURCE__<NAME>__<FIELD>` |
| `taxonomy.toml` | yes | Categories, tags and their aliases | — |

Keys never go into TOML files: a backend names the environment variable that holds its key (`api_key_env`).
`sde-rag agents show` prints which backend and model serve each role; `sde-rag doctor` checks that each
role's backend can do what the role needs (structured output, streaming) and that every backend answers.

## Running

```bash
uv run sde-rag ingest                     # fetch what is new, then clean, parse, tag, merge, index
uv run sde-rag ui                         # web UI on http://127.0.0.1:7860 (health check: /health)
uv run sde-rag query "Explain the Python GIL"
uv run sde-rag eval                       # retrieval sanity suite; non-zero exit when it regresses
uv run sde-rag stats
uv run langgraph dev                      # the three graphs in LangGraph Studio
```

Each ingest run key (default: the current half-day) runs once; `--force` runs again. An interrupted run resumes
from its checkpoint.

## Data

`data/` is the local working directory and is never committed: the working database (which also holds local
usage records), raw downloads, the vector snapshot, LangGraph checkpoints, logs and the model cache.

The published dataset has two parts. The raw files are in `dataset/raw/`, stored with Git LFS; they are named by
content hash, so each file is stored once. The database (about 80 MB) is attached to a GitHub
[release](https://github.com/DiamondGo/SDE-Interview-RAG/releases) as `rag.db.gz`, not committed: every version
of a file in Git LFS stays on the server for good, while old releases can be deleted. `dataset/MANIFEST.json` names
the release and the checksums; `sde-rag import-data` downloads the database and verifies it.

`uv run sde-rag publish-data` regenerates the dataset from the working database, without local usage records and
run logs and without sources marked `publish = false`. It uploads a new `dataset-*` release with the GitHub CLI
(`gh`, signed in) and deletes all but the newest `[dataset] release_keep` of them; then commit `dataset/`.
`--no-upload` only writes the files locally. The collected content belongs to its authors; see
[dataset/NOTICE.md](dataset/NOTICE.md).

## Tests

```bash
uv run pytest                                       # unit tests: no network, models are faked
uv run pytest -m integration                        # the query graph inside `langgraph dev`
```
