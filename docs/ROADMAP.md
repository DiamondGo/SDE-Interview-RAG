# Status and roadmap

[English](ROADMAP.md) | [中文](ROADMAP.zh-CN.md)

This is the overview: where the project stands, the phases of the whole plan, and what comes next. How the
documents divide the work:

- [DESIGN.md](DESIGN.md): what is built and why (section 12 defines the milestones and their acceptance criteria);
- [PROGRESS.md](PROGRESS.md): the detailed record of every step, with measurements and the problems fixed;
- this document: a summary of the current state and the plan, updated at milestone boundaries.

How to run the project and what it needs: [RUNNING.md](RUNNING.md).

## Current state (2026-10-31)

| Milestone | State |
|---|---|
| M1 Skeleton and minimal loop | **Accepted** (2026-10-31) |
| M2 Interview experiences, reliability and the reviewer | **Everything built; acceptance in progress**: three days of scheduled crawling and the full capability checks at formal acceptance remain |
| M3 Monitoring and evaluation | Planned |
| Later (optional) | Not scheduled |

### What works today

- **Crawling**: seven sources: GitHub question repositories (41), Hacker News, dev.to, Reddit, Medium, LeetCode
  Discuss, and 1Point3Acres (headless Chrome on a dedicated profile logged in by hand). Crawling is incremental:
  only new or changed content is downloaded. systemd runs it every day at 06:30 and 18:30.
- **Processing**: cleaning; rule parsing of the Q&A in question banks; LLM extraction of questions from interview
  experiences, where a question without a matching quote from its post is discarded; rule and LLM tagging;
  merging of exact and near-duplicate questions, with occurrence counts and heat; a hybrid BM25 and vector index.
- **Querying** (web UI, CLI, or any LangGraph client):
  - "what gets asked about a topic": the most frequent questions, grouped by subtopic, every item with its source;
  - "how to answer this question": retrieval, reranking, a check that the material is enough, new search terms
    when it is not (at most 2 rounds), then a cited answer; content no source supports goes under "模型补充";
  - complex questions (several questions, comparisons, several aspects) are split by the planner into 2 to 4
    subqueries, each with checkable success criteria; unmet ones are retried or replanned, and the answer
    follows the plan's outline.
- **Reviewer**: `claude -p` (Sonnet, read-only) checks whether the knowledge and solution in answers and source
  answers are correct; verdicts appear in the UI, within a daily quota.
- **Reliability**: "slow" and "stalled" are told apart; a stall leads to a probe, then the circuit breaker, then
  a fallback model or a retrieval-only answer. Dead letters and replay; backoff for failing sources; a run
  watchdog; daily backups; weekly consistency repair; alerts (log, desktop notification, Telegram).
- **Agent capability checks**: each of the 8 roles has a golden set and thresholds. A role below its thresholds
  is kept from where it could do harm: ingest leaves its work queued, the reviewer pauses, and a failing
  generator puts the UI into retrieval-only mode.

### Scale (2026-10-31)

| Item | Count |
|---|---|
| Documents | 970: github 313, reddit 304, devto 196, leetcode 80, medium 31, hn 23, onepoint3acres 23 |
| Distinct questions | 6,773 (2,061 tagged by the LLM, the rest by rules only) |
| Embedded chunks | 8,585 (6,773 questions, 1,812 document passages) |
| Tests | 130 unit tests; 8 agent capability checks, all passing at the full level |

### Roles and models (default configuration)

| Role | Backend |
|---|---|
| Query understanding, tagging, extraction, sufficiency grading, duplicate judging, list composition | `local`: an OpenAI-compatible server on the local network (Qwen), thinking off |
| Planner | `gateway` (gpt-6.1-sol), falling back to `local` |
| Answer generation | `gateway`, falling back to `local` (thinking on) on failure or stall |
| Reviewer | `claude -p`, Sonnet 5.5 |
| Embedding / reranking | fastembed on the CPU: `multilingual-e5-large` / `bge-reranker-v2-m3` |

Which backend serves each role is set in `config/agents.toml`; after changing a model, check that role again.

## M2 acceptance

| Criterion (DESIGN section 12) | State |
|---|---|
| Scheduled crawls run stably for 3 days | **In progress**: from 06:30 on November 1 to 18:30 on November 3, six scheduled runs |
| `pytest -m agent` passes completely | Every role passed at the full level on October 31; run once more in full at formal acceptance |
| A model made to hang on purpose is detected, the system degrades and alerts | **Passed** (October 31): with the local model hung, the circuit opened and alerted after 81 s while planning and generation went on; with both backends hung, a retrieval-only answer came in 61 s; the alert was resolved after recovery |
| A complex question is split into subqueries, and an unmet criterion leads to a replan (visible in the trace) | **Passed** (October 31): the GIL / goroutine question was split into 2 subqueries, and one of them was replanned after its retry |

What to watch during the three days: every run finishes (`last_runs` in `sde-rag stats`), no new dead letters,
no open alerts, and 1Point3Acres never asks for a login or a verification.

## The whole plan

| Phase | Contents | Acceptance criteria |
|---|---|---|
| **M1 Skeleton and minimal loop** (accepted) | Database and migrations; `langgraph.json` conventions for the three graphs; configurable roles and backends; identity and permission interfaces; GitHub, HN and dev.to connectors with incremental deduplication; cleaning, rule parsing, tagging, embedding, FTS5 plus vectors, RRF; the topic and solve paths of the query graph; stall detection and model fallback for streaming generation; the Q&A page | Asking about a topic and a specific question in the UI gives cited answers; a restart re-embeds nothing; when the pipeline runs twice in a row, the second run downloads nothing again |
| **M2 Interview experiences, reliability and the reviewer** (in acceptance) | Four new sources; LLM extraction with fabrication checks; near-duplicate merging and heat; reranking; the `grade` / `rewrite` loop; the planner for complex questions; state machine, retries, dead letters, replay; full stall detection and the watchdog; the reviewer; agent capability checks and the pre-ingest gate; systemd timers | See the previous section |
| **M3 Monitoring and evaluation** | Monitoring, browse-by-topic and evaluation pages in the UI; alert rules; evaluation sets and the `evaluate` graph (today only a retrieval self-check); correctness and faithfulness evaluation, with calibration of the judge; a regression gate; blue-green index switching | An evaluation report every night; a regression introduced on purpose (for example a poor embedding model) is stopped by the gate |
| **Later (optional)** | Offline reference answers; sandboxed checks of the code in algorithm answers; general blog discovery; embedding and reranking on a GPU; enabling logins and permissions | — |

## Known limitations and open items

- The web UI has only the Q&A page; the browse, monitoring and evaluation pages come in M3. Until then,
  `sde-rag stats`, `sde-rag review --list` and the alert notifications show what is going on.
- The `evaluate` graph only checks that the index finds each question by its own text; real evaluation sets
  come in M3.
- Every run has caps: downloads per source, 600 questions tagged and 60 posts extracted by the LLM. The rest is
  taken by later runs (for example, about 220 listed LeetCode posts are not downloaded yet).
- "Common" measures how often a question gets written into posts online, which is not how often interviewers ask it.
- LeetCode GraphQL and Reddit RSS are unofficial and can change at any time, so they have health alerts and can
  be turned off one by one.
- 1Point3Acres is read with a logged-in account: low frequency, read-only, and stopping at any verification
  lower the risk of the account being restricted, but cannot remove it. Content that needs a login stays out of
  the public dataset.
- The reviewer shares one Claude quota with other uses on this machine; when it runs low, reviews of source
  answers progress more slowly.

## Next

1. Watch the six scheduled runs from November 1 to 3; fix what goes wrong and record it in PROGRESS.
2. Formal acceptance: the full capability checks of every role (`uv run pytest -m agent --agent-level full`),
   then record the M2 acceptance.
3. Start M3: the monitoring page and alert rules first, then the evaluation sets, the evaluate graph and the
   regression gate.
