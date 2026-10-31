# Progress

[English](PROGRESS.md) | [中文](PROGRESS.zh-CN.md)

Milestones are defined in [DESIGN.md](DESIGN.md) (section 12). Each milestone is reviewed before the next starts.

## Milestone 1: skeleton and minimal loop — accepted (2026-10-31)

Delivered:

- SQLite storage with versioned migrations; documents keep raw downloads and cleaned text; questions, tags,
  chunks, FTS5 index and embeddings are persisted; vector snapshot with startup consistency checks (extend
  incrementally, rebuild from the database when lost; never re-embed).
- Configurable agents: backends (OpenAI-compatible endpoints, CLI agents) and roles with fallbacks, layered
  configuration with environment overrides, capability checks, `sde-rag agents show`, `sde-rag doctor`.
- Stall detection for model calls: first-token and idle timeouts for streams, hard deadlines for structured
  calls, runaway-repetition detection, health probe and circuit breaker after a stall.
- Identity and access-control interfaces (default: single user, everything visible) wired through retrieval,
  with tests using a restrictive policy.
- Connectors: GitHub question repositories (41 repositories), Hacker News (Algolia API), dev.to (public API),
  with incremental listing, "seen" records, conditional requests, content hashing and cross-source link
  detection.
- Rule-based Q&A extraction from Markdown (heading questions, numbered-list headings, `<details>` blocks,
  question lists), rule and LLM tagging against a configurable taxonomy, exact-duplicate merging with
  frequency and heat.
- Hybrid retrieval (BM25 + vectors with reciprocal rank fusion); query graph with topic listing and cited
  answers, citation verification, degraded answers when models fail.
- Ingest graph with run keys, a database lock (taken over when its owner died), heartbeat, checkpoint resume.
- A minimal evaluate graph (retrieval self-check), Gradio web UI with `/health`, CLI.
- Public dataset export (`sde-rag publish-data`) without local usage data: raw files with Git LFS, the database as a
  GitHub release asset (only the newest releases kept), downloaded and verified by `sde-rag import-data`.
- Tests: unit tests without network; an integration test running the query graph in `langgraph dev`.

First real ingest (2026-10-31): 486 documents (314 GitHub files, 150 dev.to articles, 23 Hacker News
discussions), 6,535 distinct questions, 7,585 chunks embedded with `intfloat/multilingual-e5-large`.

Acceptance (2026-10-31), against the criteria in DESIGN.md section 12:

| Criterion | Result |
|---|---|
| "Python 常见面试题" in the UI returns a cited list | Pass: 27 questions from Python question banks, every item cited, no invalid citations (local model, about 22 s) |
| A specific question returns a cited answer | Pass: "如何设计一个限流器（rate limiter）？" (gateway model, 35 s) and "What is the difference between a process and a thread?" in the web UI (21 s) |
| A restart does not re-embed anything | Pass: the UI and the CLI load the vector snapshot; a re-index of all 521 documents kept all 1,395 document chunks and embedded nothing |
| Two runs in a row: the second downloads nothing again | Pass: two forced runs (all sources listed, intervals ignored): 0 duplicate downloads; GitHub repositories answered 304 for unchanged commits; the second run downloaded 0 documents and embedded 0 chunks in 27 s |

Fixed during acceptance:

- Invented topic-list items: a list line citing only labels outside the context is now dropped, not just stripped
  of its label; the CLI prints the verified answer when it differs from the streamed one.
- Topic ranking: generic request words ("common", "interview questions", "常见", "面试题") no longer match
  headings such as "... Interview Questions"; a "focus" ranking prefers sources dedicated to the requested tag;
  tied heat values share a rank in the fusion.
- Rule tags: words shared by several languages ("decorator", "generator", "closure", "hooks", "coroutines") are no
  longer aliases of one language; rule tags are made again when the taxonomy version changes.
- Parsing: "— Difficulty: Easy" is stored as difficulty, not as part of the question; first-person blog headings
  ("What I took away") are not questions. Documents parsed by an older parser are parsed again from the stored
  text (no download), and unchanged document chunks keep their embeddings.
- Routing: a message that is one specific question is answered even when the router labels it a topic request.

After acceptance: 532 documents (521 indexed), 6,633 distinct questions, 8,028 embedded chunks; 1,200 questions
tagged by the LLM.

Known limitations, planned for later milestones:

- "Common" can only be measured by exact repeats until near-duplicates are merged: the bank holds eight phrasings
  of "what are generators", each counted once, so a topic list leans on source focus and document order.

- Questions are merged only when their normalized text is identical (near-duplicates: milestone 2).
- Interview-experience posts are indexed as text chunks; questions are extracted from them only where they are
  written as headings (LLM extraction: milestone 2).
- The LLM tagger refines 600 questions per run; the rest keep rule tags until their turn.
- The evaluate graph only checks that the index finds each question by its own text (real evaluation sets:
  milestone 3).

## Milestone 2: interview experiences, reliability and the reviewer — in progress

Scope: LeetCode Discuss, Reddit, Medium and 1point3acres connectors; LLM question extraction with fabrication
checks; near-duplicate merging; reranking; the grade/rewrite loop; dead-letter replay; the full watchdog; the
reviewer role; agent capability tests with the pre-ingest gate; systemd timers.

Order of work: (1) reliability and scheduling; (2) LLM extraction, fabrication checks and near-duplicate merging;
(3) the new sources; (4) reranking, the grade/rewrite loop and the reviewer; (5) agent capability tests and the
gate, then three days of scheduled runs for acceptance.

### Step 1: reliability and scheduling — done (2026-10-31)

- Dead letters: `sde-rag replay --list` lists them; `sde-rag replay [--stage] [--status dead|failed] [--ids]`
  sends documents through their stage again on the next run, with the attempt count reset. The index stage now
  records failures per document too, so one bad document no longer stops the stage.
- Source backoff: a source failing several runs in a row waits 1, 2, 4 intervals before the next try, at most
  48 hours (before, it was retried every run).
- Run watchdog: a thread in the ingest process watches for progress (every finished record and every listed
  item counts). After 20 minutes with none, it writes every thread's stack to `data/logs/stall-*.txt`, marks the
  run `stalled`, releases the lock, alerts, and exits with status 3. Starting a new run marks runs still recorded
  as `running` whose process is gone as `abandoned`.
- Model stalls: each run's report counts stalled calls per role@backend and alerts at 3 or more in one run; a
  backend's circuit breaker opening raises an alert (`llm:<backend>`), resolved when it answers again.
- New in the run report: the backlog (alert when it grows three runs in a row) and new dead letters (alert when
  more than 30 are added in one run).
- Ingest graph nodes retry on SQLite write-lock timeouts (LangGraph `RetryPolicy`).
- `sde-rag backup`: SQLite online backup, gzipped (the 82 MB database becomes 45 MB in about 4 seconds); 7 kept.
- `sde-rag reconcile`: repairs references between tables (dangling "seen" records, orphaned chunks, vectors and
  full-text rows), lists every GitHub repository in full on the next run, recomputes heat (its time decay),
  deletes expired index versions and checks the full-text index. On a copy of the working database: 1.6 seconds,
  no inconsistencies.
- `sde-rag watchdog-ui`: checks the UI's `/health`; two failures in a row restart `sde-ui.service` and alert.
- `sde-rag install-systemd [--enable] [--ui] [--remove]`: writes 9 units with this machine's paths to
  `deploy/systemd/` (not committed) and links them into `~/.config/systemd/user`; existing units of the same name
  are left alone.
- Tests: 15 new (69 in total), including a hung model being detected, the fallback taking over, the circuit
  opening, the alert and the recovery. A real forced run on a copy: after reconcile, all 314 GitHub files were
  listed again and recognised as unchanged by their blob SHA; 0 downloads, no errors, no stalls, 42 seconds.

## Milestone 3: monitoring and evaluation — planned

Monitoring and evaluation pages, alert rules, evaluation sets, generation and correctness evaluation with judge
calibration, regression gate, blue-green index switch.
