# Running the project

[English](RUNNING.md) | [中文](RUNNING.zh-CN.md)

What the project needs, how to install and configure it, how to run it by hand and on a schedule, and the
commands for day-to-day operation. Where the project stands: [ROADMAP.md](ROADMAP.md); the design:
[DESIGN.md](DESIGN.md).

## The parts

| Part | What it is |
|---|---|
| Three LangGraph graphs | `ingest` (crawling and processing), `query` (answering), `evaluate` (evaluation), registered in `langgraph.json`; any LangGraph runtime can run them |
| The `sde-rag` CLI | Runs the graphs, plus the operational commands (checks, backups, replay, reviews, ...) |
| Web UI | Gradio, `http://127.0.0.1:7860` by default, health check at `/health` |
| Scheduled jobs | systemd user units: ingest at 06:30 and 18:30, backup and evaluation at 03:30, consistency repair on Sundays at 04:30, a UI health check every 5 minutes |
| `data/` | The local working directory: SQLite database, raw downloads, vector snapshot, LangGraph checkpoints, model cache (not committed) |
| External services | The model services (below) and the source websites |

## Requirements

### Required

| Item | Requirement |
|---|---|
| Operating system | Linux. The scheduled jobs are systemd user units; elsewhere the commands still run by hand or from another scheduler |
| Python | 3.12 or later; uv installs it when missing |
| [uv](https://docs.astral.sh/uv/) | Installs the dependencies and runs the commands (`uv run ...`) |
| CPU / memory | Embedding and reranking run on the CPU; no GPU needed. A process that loads both models (the web UI, an ingest run) peaks at about 2.5 GB of memory |
| Disk | About 2.7 GB of model cache (`multilingual-e5-large` about 2.2 GB, `bge-reranker-v2-m3` about 0.57 GB); the database is about 120 MB now; 7 daily backups of about 45 MB each |
| Network | Access to the sources (GitHub API, HN Algolia, dev.to, Reddit, Medium, LeetCode, 1Point3Acres); model downloads from Hugging Face on the first run |
| Model services | At least one OpenAI-compatible endpoint with structured output (function calling) and streaming; see the next section |

### Model services

The default configuration uses three backends. Which backend serves each role is set in `config/agents.toml`
(DESIGN sections 5.2 and 5.3):

| Backend | Default | Roles it serves | Notes |
|---|---|---|---|
| `local` | An OpenAI-compatible server on the local network (llama.cpp, vLLM, Ollama, ...) with a Qwen model | Query understanding, tagging, extraction, sufficiency grading, duplicate judging, list composition; fallback of the planner and the generator | Unmetered bulk work. A single-slot server needs `max_concurrency = 1`, so that requests queue in this process |
| `gateway` | An OpenAI-compatible gateway or provider (gpt-6.1-sol, Responses API) | Answer generation, planner | Rate limit 0.5 requests a second |
| `claude_sonnet` | The `claude` CLI, Sonnet 5.5, read-only mode | Reviewer | A daily call cap; `claude` must have been signed in on this machine |

With a single OpenAI-compatible endpoint, point every role at it, and turn the reviewer off with
`[review] enabled = false` (in `config/settings.local.toml`). When the reviewer and the generator share a backend,
`sde-rag doctor` warns: the model would be judging its own answers. After changing the backend or model of any
role, check that role again (see "Checks" below).

### Optional

| Item | Used for | Without it |
|---|---|---|
| [Git LFS](https://git-lfs.com/) | The raw files of the published dataset (`dataset/raw/`) | Start crawling from an empty database |
| Google Chrome | The 1Point3Acres connector (Playwright drives the installed Chrome; no `playwright install` needed) | The source is skipped |
| A dedicated Chrome profile logged in to 1Point3Acres | The same; log in with Chrome started with `--password-store=basic`, so that systemd jobs can read the cookies | The source is skipped |
| `GITHUB_TOKEN`, or a signed-in `gh` | The GitHub API rate limit (60 requests an hour without a token) | Crawling the question repositories is slow or fails |
| `gh`, signed in | Uploading dataset releases with `sde-rag publish-data` | Only `--no-upload` |
| `notify-send` | Desktop alerts | Alerts go to the log only |
| A Telegram bot | Telegram alerts | The same |
| `loginctl enable-linger` | Scheduled jobs run without a login session | They run only while you are logged in |

## Installation

```bash
git clone https://github.com/DiamondGo/SDE-Interview-RAG.git && cd SDE-Interview-RAG
git lfs pull                                      # optional: raw files of the published dataset
uv sync                                           # dependencies, including the dev group (langgraph-cli, pytest)
cp config/agents.example.toml config/agents.toml  # model configuration: set your endpoints and model names
cp .env.example .env                              # endpoint URLs and keys referenced by agents.toml
uv run sde-rag import-data                        # optional: download and verify the published database
uv run sde-rag doctor                             # configuration problems; does every backend answer?
```

The variables in `.env` (their names come from the `*_env` fields in `agents.toml`):

| Variable | Used for |
|---|---|
| `LOCAL_BASE_URL`, `LOCAL_API_KEY` | Backend `local` |
| `GATEWAY_BASE_URL`, `GATEWAY_API_KEY` | Backend `gateway` |
| `GITHUB_TOKEN` | Optional, see above |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Optional, when `[notify] channels` includes `telegram` |

When a shared dotenv file already exists, list it in `config/settings.local.toml` instead of copying `.env`:
`env_files = ["~/path/.env"]`. Keys never go into TOML files.

## Configuration

Configuration has three layers, each overriding the one before: `config/<name>.toml` (committed), then
`config/<name>.local.toml` (this machine, not committed), then environment variables.

| File | Contents | Environment override |
|---|---|---|
| `agents.toml` (copied from the example, not committed) | Backends; each role's backend, fallback chain and timeouts; the embedding and reranking models | `SDE_BACKEND__<NAME>__<FIELD>`, `SDE_ROLE__<NAME>__<FIELD>`, `SDE_MODEL__<NAME>__<FIELD>` |
| `settings.toml` | Data directory, extra dotenv files, alert channels, HTTP politeness, per-run caps, the planner, the review quota, gates | `SDE_SETTING__<SECTION>__<FIELD>` |
| `sources.toml` | Each source: on or off (`enabled`), interval, rate limits, downloads per run; sources with `publish = false` stay out of the dataset | `SDE_SOURCE__<NAME>__<FIELD>` |
| `taxonomy.toml` | Categories, tags and their aliases | — |
| `agent_thresholds.toml` | Capability-check thresholds (relaxed per machine in `agent_thresholds.local.toml`) | — |

Common machine settings go in `config/settings.local.toml`:

```toml
env_files = ["~/path/.env"]          # optional

[notify]
channels = ["log", "desktop"]        # any of log, desktop, telegram

[onepoint3acres]
chrome_profile = "~/path/to/logged-in-profile"
```

## Checks

```bash
uv run sde-rag agents show                 # which backend and model serve each role, and its check results
uv run sde-rag doctor                      # configuration errors, index vs embedding model, does every backend answer
uv run sde-rag agent-check                 # quick level of every role (about 10 minutes)
uv run sde-rag agent-check --level full --roles router,planner
```

A check result belongs to the role's configuration fingerprint: a changed backend, model or address leaves the
role without a record until it is checked again. A role without a passing record is held back by the gates:
ingest leaves its work queued, the reviewer pauses, and a failing generator puts the UI into retrieval-only mode.

## Running

```bash
uv run sde-rag ingest                      # fetch what is new, then clean, parse, extract, tag, merge, index
uv run sde-rag ingest --sources reddit,hn --doc-limit 50
uv run sde-rag ingest --skip-llm           # rule tags only, no LLM calls
uv run sde-rag ui                          # web UI (--host, --port)
uv run sde-rag query "How do I design a rate limiter?"
uv run sde-rag query "Explain the Python GIL" --review   # wait for the reviewer's verdict
uv run sde-rag eval                        # retrieval self-check; non-zero exit when it regresses
uv run sde-rag stats                       # counts, source health, recent runs, dead letters, open alerts
uv run langgraph dev                       # the three graphs in LangGraph Studio
```

Ingest run keys: a run started by hand gets `manual-<time>`; the scheduled job (`--scheduled`) uses the current
half-day (for example `2026-11-01-AM`). Each key runs once; `--force` runs it again (and ignores the sources'
intervals). An interrupted run resumes from its checkpoint. Only one ingest runs at a time (a database lock,
taken over when the process holding it is gone).

## Scheduled runs (systemd)

```bash
uv run sde-rag install-systemd --enable --ui   # write the units and enable the timers; --ui runs the UI as a service too
uv run sde-rag install-systemd --remove        # disable them and remove the links
loginctl enable-linger                         # run without a login session
```

The units are written with this machine's paths to `deploy/systemd/` (not committed) and linked into
`~/.config/systemd/user`; existing units with the same names are not overwritten.

| Unit | When | What |
|---|---|---|
| `sde-ingest.timer` | Every day at 06:30 and 18:30 (random delay up to 2 minutes; a missed run is caught up after boot) | `sde-rag ingest --scheduled`; 1Point3Acres only in the 18:30 run |
| `sde-eval.timer` | Every day at 03:30 | Database backup, evaluation, the quick capability checks of every role (the reviewer's full check weekly) |
| `sde-reconcile.timer` | Sundays at 04:30 | Consistency repair, a full GitHub listing on the next run, heat refresh |
| `sde-watchdog.timer` | Every 5 minutes | Checks the UI's `/health`; two failures in a row restart `sde-ui.service` and alert |
| `sde-ui.service` | Always on, restarted on failure | The web UI |

```bash
systemctl --user list-timers 'sde-*'
journalctl --user -u sde-ingest.service -n 200   # run logs are in the journal
systemctl --user restart sde-ui.service          # after changing code or configuration
```

When other jobs on the machine share the local model server, schedule the ingest runs away from them, or they
queue behind each other and time out. When another process holds the 1Point3Acres profile, the run skips that
source; that is not a failure.

## Day-to-day operation

| Situation | What to do |
|---|---|
| Overall state | `uv run sde-rag stats`: recent runs (`last_runs`), consecutive failures per source, dead letters, open alerts |
| An alert arrives | Alerts are deduplicated by key (at most one reminder every 24 hours), and a recovery notice is sent automatically. `llm:<backend>`: a model service does not answer, see the next row; `source:<name>`: a source keeps failing or is blocked; `ingest:stalled`: the watchdog ended a run, with the stacks in `data/logs/stall-*.txt`; `agent:<role>`: a role is below its thresholds |
| A model service hangs or is slow | Nothing to do by hand: stalled calls are abandoned, two failed probes in a row open the circuit (10 minutes by default), and the work stays queued or goes to a fallback model; it clears itself when the service answers again. Confirm with `uv run sde-rag doctor` |
| Dead letters | `uv run sde-rag replay --list`; after fixing the cause, `uv run sde-rag replay [--stage parse] [--ids 1,2]` sends them through again on the next run |
| The reviewer | `uv run sde-rag review --list` (the queue and today's remaining calls); `--show <id>` for one job |
| A model or backend changed | `uv run sde-rag agent-check --level full --roles <role>`; the role is used again once it passes |
| 1Point3Acres asks for a login or a verification | The connector stops and alerts; it never tries to log in. Open the profile with the same flags (`--password-store=basic`), log in by hand, and close Chrome |
| Backups | Every day at 03:30 to `data/backups/rag-<time>.db.gz` (7 kept); by hand: `uv run sde-rag backup` |
| Restoring a backup | Stop the UI and the timers (`systemctl --user stop sde-ui.service sde-ingest.timer`), delete `data/rag.db-wal` and `data/rag.db-shm`, restore with `gunzip -c data/backups/rag-<time>.db.gz > data/rag.db`, and start them again. The vector snapshot no longer matches the database, so it is rebuilt from the database; nothing is embedded again |
| Publishing the dataset | `uv run sde-rag publish-data` (needs a signed-in `gh`), then commit `dataset/` |

## The data directory

| Path | Contents |
|---|---|
| `data/rag.db` | The working database: documents, questions, tags, chunks, the FTS index, embeddings, runs, alerts, reviews, query logs |
| `data/raw/` | Raw downloads (gzipped), so a changed cleaner or parser can reprocess without crawling again |
| `data/index/` | The vector snapshot; checked at startup and extended or rebuilt from the database when needed |
| `data/checkpoints.db` | LangGraph checkpoints (interrupted ingest runs resume from here; UI conversations) |
| `data/models/` | Cache of the embedding and reranking models |
| `data/backups/` | Daily backups |
| `data/logs/` | Stack dumps written by the watchdog, and similar |

`data/` is not committed. Its location is `data_dir` in `settings.toml`.

## Tests

```bash
uv run pytest                                # unit tests: no network, models faked (about 10 seconds)
uv run pytest -m integration                 # the query graph inside `langgraph dev`
uv run pytest -m agent [--agent-level full]  # capability checks against the real models
```
