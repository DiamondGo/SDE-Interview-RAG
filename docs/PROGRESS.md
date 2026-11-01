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
checks; near-duplicate merging; reranking; the grade/rewrite loop; the planner for complex questions (added to the
design on 2026-10-31: subqueries with success criteria, replanning when they fail; DESIGN 5.4.1); dead-letter
replay; the full watchdog; the reviewer role; agent capability tests with the pre-ingest gate; systemd timers.

Order of work: (1) reliability and scheduling; (2) LLM extraction, fabrication checks and near-duplicate merging;
(3) the new sources; (4) reranking, the grade/rewrite loop and the reviewer; (4b) the planner; (5) agent
capability tests (including the planner) and the gate, then three days of scheduled runs for acceptance.

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

### Step 2: LLM extraction, fabrication checks, near-duplicate merging — done (2026-10-31)

- Extraction (`extract` node, `extractor` role): interview-experience posts are read in parts of about 6,000
  characters. The model reports whether the post is about interviews, the company, role, level and year, and
  each question with a canonical English wording, a verbatim quote, the answer the post gives and the round.
  A question whose quote does not match the post (similarity below 0.8, ignoring markup, case and punctuation;
  Chinese compared character by character) is discarded and counted; a run alerts when more than 15% are.
  A part the model cannot answer for is split in half and tried again. Extraction has its own status, attempts
  and dead letter (`sde-rag replay --stage extract`), so posts are indexed as text while the model is away.
- Reposts: a SimHash of the clean text; a post within 3 bits of one already read is marked `duplicate` and not
  read again.
- Near-duplicate questions (`dedup` node, `dedup_judge` role): thresholds measured on the M1 data (DESIGN 6.4).
  Automatic merges need similarity ≥ 0.95 and the same content words in about the same order; other pairs
  ≥ 0.92 go to the judge (600 pairs per run). Merged wordings become aliases; merges are logged.
- Measured on a copy of the working database: embedding all 6,633 question texts once took 5 minutes. The
  extractor reads a 6,000-character part in about 16 seconds (the 218 posts: about 1.5 to 2 hours, spread over
  runs of 40 posts). The judge decides about 3.5 pairs per second with the flat output (lists of pair numbers;
  a list of objects broke the local model's JSON in 3 of 4 batches). One run with 8 posts: 153 questions
  proposed, 148 kept, 4 discarded as unquoted (2.6%, all code-output questions whose code the model
  reformatted), 74 new questions; 63 automatic merges, 100 pairs judged, 58 merged; 7 minutes 21 seconds.
- Heat now reflects repetition across sources: after the merges, "Where do you see yourself in five years?"
  has 6 mentions from 3 sources.
- Tests: 10 new (79 in total).

### Step 3: new sources — done (2026-10-31)

- Reddit (`reddit_rss`): search RSS per subreddit (leetcode, cscareerquestions, ExperiencedDevs, csMajors,
  developersIndia) for titles containing "interview"; the post text comes with the feed, so a post costs no
  extra request. Link posts are matched to content already stored instead of being fetched. 8 seconds between
  feeds got 429; now 20.
- Medium (`medium`): tag feeds carry only a snippet and article pages answer 403, so a post found by tag is read
  from its author's feed (each feed fetched once a run). Posts in publication feeds without full text keep
  their snippet and are skipped by cleaning as too short.
- LeetCode Discuss (`leetcode_discuss`): GraphQL listing (newest first, stops at the high-water mark) and post
  bodies in Markdown; 3 seconds between requests; at most 80 posts per run.
- 1Point3Acres (`onepoint3acres`, DESIGN 3.4): headless Chrome on the logged-in profile (the same launch flags
  as the profile's other user). Only the 18:30 run, software-engineering posts only (`码农类`), at most 30 posts,
  10 to 20 seconds before every page, a 15-minute session limit. The profile in use means the run is skipped
  (neither success nor failure); a block (logged out, browser check, site prompt) stops the run with an alert.
  The visible text is stored; parts behind the points threshold are marked and not read. `publish = false`.
- Framework: per-source `max_downloads_per_run` (the rest is listed again next run, not failed), `run_hours`,
  skipped listings, `HttpClient.post` for read-only query APIs, a converter for HTML fragments from feeds, and
  quotes that skip text with "..." are checked piece by piece (the two questions discarded in the trial were real
  questions quoted that way).
- Live check on a copy of the working database: Reddit listed 200 posts (two of five feeds got 429 at 8 seconds),
  Medium 39 with full text from author feeds, LeetCode 300 (back to 2026-10-12), 1Point3Acres 50
  software-engineering threads from two list pages; 3 posts read in 63 seconds, all with a hidden part; the
  profile lock was released afterwards. Extraction on them: the eBay phone-screen question (Meeting Rooms II)
  from the visible part of a 1Point3Acres post, with company and round from the board's labels; 11 questions
  from a Medium Flipkart experience; nothing from a Netflix post whose questions were all hidden.
- Extraction: 60 posts per run (about 25 seconds a post); the new sources add several hundred posts, so the
  backlog takes a week or two of scheduled runs.
- Tests: 10 new (89 in total).

### Step 4: reranking, the grade/rewrite loop, the reviewer — done (2026-10-31)

- `rerank`: 30 candidates are scored by `BAAI/bge-reranker-v2-m3-int8` (fastembed, CPU) against the question;
  6 go into the context, at most 3 paragraphs of one post (in the first live check, one long article had taken
  all six places). A source answer the reviewer judged incorrect loses 3 points of score, and the reviewer's
  corrections travel with it in the context. A failing reranker keeps the retrieval order.
- `grade` / `rewrite`: the local grader judges whether the context covers the question; when not, its new
  search queries are added and retrieval runs again, at most twice. A failing grader counts as "enough".
- The reviewer (`reviewer` role, `claude -p` with Sonnet, read-only, JSON schema): queue `review_jobs` with
  priorities (1: the user pressed 请评审员复核, 2: a `solve` answer, 4: source answers), a daily quota counted per
  call (failed and retried calls included; a batch is one call), 40 calls kept for interactive use, retries,
  and an alert after three failures in a row. The UI process runs the queue in a background thread (pausing a
  minute after a failure, so a broken reviewer does not use up the quota); the page shows "评审中" and then the
  verdict with the issues and corrections; the ingest run reviews 10 source answers of the hottest questions
  in batches of 5. Reviews of query answers hold the questions people asked: they are never published.
- `sde-rag review --list | --run N | --sources N | --show ID`; `sde-rag query --review` waits for the verdict.
- Live check on the working database ("如何设计一个限流器（rate limiter）？"): reranked, the grader found the
  context sufficient on the first round, answer with citations, then the reviewer: correct (5/5, 5/5), one minor
  issue (a token bucket with B = 60 can pass about 120 requests in a minute, more than "60 per minute" says).
  The reviewer answers in about 5 to 60 seconds; the first query also downloaded the reranker (0.57 GB).
- Tests: 9 new (98 in total).

### Step 4b: the planner for complex questions — done (2026-10-31)

- `understand` reports `complexity`; two or more question marks make a question complex even when the router
  says simple, and rules decide when the router fails. Complex questions go to `plan` (the `planner` role:
  local model with thinking, the gateway as fallback); simple ones keep the existing path.
- `graphs/planner.py`: plan validation (2 to 4 subqueries, every subquery with a code-checked criterion,
  English search terms, no dependency cycles; one subquery means "not needed"); subqueries run as LangGraph
  `Send` branches, dependent ones in a later wave with `{s1}` replaced by what s1 found; criteria `min_items`,
  `has_kind`, `mentions` by code and `covers` in one grader call; one retry with new search terms; `check_plan`
  and `replan` (met and unchanged subqueries keep their results, a replan that changes nothing stops); budget of
  time and calls; `synthesize` takes items from every part in turns and answers by the outline, saying which
  parts found no source. Every failure falls back to the simple path or to what was found.
- First live run ("Python 的 GIL 对多线程有什么影响？Go 的 goroutine 之间又是怎么通信的？"): planned into two
  parts, both met, answered in 46 seconds, but the planner had labeled both parts `topic`, so the answer was two
  lists of interview questions. Fixed: parts of a question that wants an answer become `solve` (and the prompt
  says so). Second run: both parts retried, part 1 replanned, stopped by the budget; a good answer (CPU-bound
  versus I/O-bound threads, the GIL is no thread-safety guarantee, channels and select) in 3 minutes 21 seconds.
  Part 1 had failed `min_items` because subqueries were reranked against their goal, written in Chinese:
  reranking against the English search query gives the same items scores above the threshold (1.13 and 0.91
  instead of −0.28 and −0.95). Most of the latency is the planner's thinking and the extra grader calls.
- Tests: 17 new (115 in total).

### Fixes after the first scheduled run (2026-10-31)

- The 18:30 run did nothing ("already done for this run key"): a run started by hand earlier that afternoon had
  used the default half-day key. Now `sde-rag ingest` started by hand gets its own key (`manual-<time>`); the
  timer's unit passes `--scheduled` for the half-day key.
- A manual run at 18:59 instead. 1Point3Acres (allowed in the 18:00 hour) held the browser profile from 18:59 to
  19:06, across the full hour, when another scheduled job on this machine may use the same profile; that time it
  did not need it. Lesson: `run_hours` alone does not keep the session clear of the hour; start runs that
  include it before about 18:40.
- In that run, one thread needing read permission above 80 was taken for a block, stopped the rest of the source
  and raised an alert. Now a prompt about one thread skips that thread (`SkipDoc`: remembered as seen, never
  retried) and the run goes on.
- The run (stopped at 19:25 and resumed from its checkpoint): Medium 31 posts, Reddit 309, LeetCode 80 (220
  left for later runs), 1Point3Acres 23; extraction of 60 posts kept 477 questions (382 new), 7 discarded as
  unquoted (1.4%).
- Tests: 2 new (117 in total).

### Step 5: agent capability checks and gates — done (2026-10-31)

- `sde_rag/agent_checks.py`: one scorer per role (extractor, tagger, router, grader, planner, generator,
  reviewer, and a whole-graph smoke test) through the pipeline's own prompts, schemas and parsing; thresholds in
  `config/agent_thresholds.toml` (relaxed per machine in `agent_thresholds.local.toml`, marked by
  `sde-rag agents show`); every model call's latency and stalls measured; results stored per configuration
  fingerprint (`agent_checks`, local only). `sde-rag agent-check [--level quick|full] [--roles]`,
  `pytest -m agent [--agent-level full]`.
- Golden sets in `evals/agent/` (written for the sets, modeled on crawled content): extractor 12 posts, tagger 52
  questions, router 32 queries, grader 16 + 16, planner 12 complex + 8 simple + 5 replanning, generator 8, reviewer
  12 (6 with a planted error), smoke test 6.
- Gates: ingest checks the extractor and the tagger when their configuration has no result from the last 24
  hours, and leaves their work queued when they fail; the reviewer pauses without a passing result; a failing
  generator puts the UI into retrieval-only mode with a banner; the UI checks the router and the generator in the
  background at start; the nightly timer runs the quick level of every role and the reviewer's full level weekly.
- First real run (quick): everything passed except the router (filter F1 0.75; one case with an empty object
  from the local model). Fixes: the router prompt lists the taxonomy's tags (it had invented "process",
  "operating-systems"); the category implied by the router's tags wins over its guess, in production too (it had
  put Kubernetes under system design, which made the category filter exclude the Kubernetes questions); a second
  try on the same backend; `max_tokens` 800 → 1200. Router full: intent 0.969, filter F1 0.889, key terms 1.0.
- Full level of every role, all passed: extractor recall 0.96, fabrication 0; tagger category 0.962, tag F1 0.812;
  grader 0.938 / 1.0; planner 1.0 on every metric (964 s for 25 cases: about 72 s a call with thinking on);
  generator citations, "no source" and language all 1.0; reviewer detection 1.0, false positives 0, pinpointed 1.0;
  smoke test 6 of 6, p95 232 s (the planned Kafka question).
- Tests: 12 new (129 in total, plus the 8 capability checks under `-m agent`).

## Milestone 3: monitoring and evaluation — planned

Monitoring and evaluation pages, alert rules, evaluation sets, generation and correctness evaluation with judge
calibration, regression gate, blue-green index switch.
