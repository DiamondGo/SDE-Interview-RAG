# Progress

[English](PROGRESS.md) | [中文](PROGRESS.zh-CN.md)

Milestones are defined in [DESIGN.md](DESIGN.md) (section 12). Each milestone is reviewed before the next starts.

## Milestone 1: skeleton and minimal loop — done, in review

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
- Public dataset export (`sde-rag publish-data`, Git LFS) without local usage data.
- Tests: unit tests without network; an integration test running the query graph in `langgraph dev`.

First real ingest (2026-10-31): 486 documents (314 GitHub files, 150 dev.to articles, 23 Hacker News
discussions), 6,535 distinct questions, 7,585 chunks embedded with `intfloat/multilingual-e5-large`.

Known limitations, planned for later milestones:

- Questions are merged only when their normalized text is identical (near-duplicates: milestone 2).
- Interview-experience posts are indexed as text chunks; questions are extracted from them only where they are
  written as headings (LLM extraction: milestone 2).
- The LLM tagger refines 600 questions per run; the rest keep rule tags until their turn.
- The evaluate graph only checks that the index finds each question by its own text (real evaluation sets:
  milestone 3).

## Milestone 2: interview experiences, reliability and the reviewer — planned

LeetCode Discuss, Reddit, Medium and 1point3acres connectors; LLM question extraction with fabrication checks;
near-duplicate merging; reranking; the grade/rewrite loop; dead-letter replay; the full watchdog; the reviewer
role; agent capability tests with the pre-ingest gate; systemd timers.

## Milestone 3: monitoring and evaluation — planned

Monitoring and evaluation pages, alert rules, evaluation sets, generation and correctness evaluation with judge
calibration, regression gate, blue-green index switch.
