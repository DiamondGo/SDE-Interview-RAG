# SDE Interview RAG

[English](README.md) | [中文](README.zh-CN.md)

A RAG (retrieval-augmented generation) system built to help software engineers prepare for interviews.

It collects interview questions from question repositories and interview experiences from Hacker News, dev.to,
Reddit, Medium, LeetCode Discuss and 1Point3Acres (questions are extracted from the posts by an LLM, each one
checked against a quote from its post), merges near-duplicate questions, cleans and tags them (languages, frameworks, system design, AI/ML, automation, CS fundamentals,
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
- Status and roadmap: [docs/ROADMAP.md](docs/ROADMAP.md) ([中文](docs/ROADMAP.zh-CN.md))
- Running the project and its requirements: [docs/RUNNING.md](docs/RUNNING.md) ([中文](docs/RUNNING.zh-CN.md))
- Detailed progress log: [docs/PROGRESS.md](docs/PROGRESS.md) ([中文](docs/PROGRESS.zh-CN.md))
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
| `sources.toml` | yes | Connectors: question repositories, Hacker News, dev.to, Reddit, Medium, LeetCode Discuss, 1Point3Acres; `publish = false` keeps a source out of the dataset | `SDE_SOURCE__<NAME>__<FIELD>` |
| `taxonomy.toml` | yes | Categories, tags and their aliases | — |

1Point3Acres is read in headless Chrome (Google Chrome must be installed) with a dedicated profile that you
log in to by hand, started with `--password-store=basic` so that systemd jobs can read its cookies. Name it in
`config/settings.local.toml`; without it the source is skipped:

```toml
[onepoint3acres]
chrome_profile = "~/path/to/logged-in-profile"
```

It is read once a day (the 18:30 run), at most 30 posts, 10 to 20 seconds between pages, read-only. A login
page, a CAPTCHA or a permission prompt stops it with an alert; parts of posts hidden behind the site's points
threshold are not read. Its content is never published (`publish = false`).

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

### Scheduled runs (systemd)

```bash
uv run sde-rag install-systemd --enable   # timers: ingest 06:30/18:30, backup + eval 03:30, reconcile Sun 04:30,
                                          # UI health check every 5 min; add --ui to run the UI as a service too
uv run sde-rag install-systemd --remove   # disable them and remove the links
```

The units are written with this machine's paths to `deploy/systemd/` (not committed) and linked into
`~/.config/systemd/user`; timers run without a login session only with `loginctl enable-linger`.

Failures are retried rather than lost. A document that fails a stage waits 1, 2, 4 ... hours (up to 48) and goes
to the dead letter after 5 attempts; a failing source waits 1, 2, 4 intervals (up to 48 hours). An ingest run in
which nothing at all progresses for 20 minutes dumps every thread's stack to `data/logs/`, is marked `stalled`
and exits; the next run carries on with the queued work. Alerts go to the channels in `[notify]`.

```bash
uv run sde-rag replay --list              # dead letters (--status failed: waiting for a retry)
uv run sde-rag replay --stage parse       # send them through their stage again on the next run
uv run sde-rag backup                     # gzipped online backup in data/backups/ (7 kept)
uv run sde-rag reconcile                  # consistency repair, full GitHub listing next run, heat refresh
uv run sde-rag review --list              # the reviewer's queue and today's remaining calls
uv run sde-rag query "..." --review       # ask, then wait for the reviewer's verdict on the answer
```

Capability checks test each LLM role, as configured on this machine, against a golden set (`evals/agent/`) and
the thresholds in `config/agent_thresholds.toml` (relax them per machine in `agent_thresholds.local.toml`):

```bash
uv run sde-rag agent-check                # quick level of every role (about 10 minutes)
uv run sde-rag agent-check --level full --roles router,planner
uv run pytest -m agent [--agent-level full]
```

A result belongs to the role's configuration: after changing a model or backend, check again. A role without a
passing result is not used where it could do harm: ingest leaves extraction or tagging queued, the reviewer
pauses, and a failing generator puts the UI into retrieval-only mode. The nightly timer repeats the checks.

The reviewer (by default `claude -p` with Sonnet, read-only) checks whether an answer's knowledge and solution
are correct. In the UI, answers to specific questions are reviewed shortly after they are given and the verdict
appears under the answer; "请评审员复核" moves an answer to the front of the queue. `[review] daily_quota` limits
the calls per day.

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
