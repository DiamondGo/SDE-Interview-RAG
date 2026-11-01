-- LLM question extraction from interview-experience posts: its own track next to parse/index, so posts are
-- indexed as text while the extractor is unavailable.
ALTER TABLE documents ADD COLUMN extract_status TEXT;      -- NULL (pending) | done | not_interview | duplicate | failed | dead
ALTER TABLE documents ADD COLUMN extract_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN extract_error TEXT;
ALTER TABLE documents ADD COLUMN extractor TEXT;           -- llm:<role fingerprint> that did the extraction
ALTER TABLE documents ADD COLUMN extract_stats TEXT;       -- JSON: questions proposed, kept, discarded as unquoted
ALTER TABLE documents ADD COLUMN simhash INTEGER;          -- of the clean text, signed 64-bit
ALTER TABLE documents ADD COLUMN dup_of INTEGER;           -- near-duplicate of this document: extracted once, there
CREATE INDEX documents_extract ON documents (kind, extract_status);

-- Near-duplicate questions merged into one canonical question. The dropped question's normalized-text hash
-- stays here, so parsing the same wording again attaches to the kept question instead of recreating it.
CREATE TABLE question_aliases (
    norm_hash TEXT PRIMARY KEY,
    question_id INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX question_aliases_q ON question_aliases (question_id);

-- Embeddings of the question text alone (no answer), for finding near-duplicates. Computed once per wording.
CREATE TABLE question_vectors (
    question_id INTEGER PRIMARY KEY,
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vector BLOB NOT NULL
);

-- Candidate pairs in the uncertain similarity band, judged by the dedup_judge role within a per-run budget.
CREATE TABLE dedup_pairs (
    a INTEGER NOT NULL,                         -- a < b
    b INTEGER NOT NULL,
    similarity REAL NOT NULL,
    verdict TEXT,                               -- NULL (pending) | same | different
    judged_by TEXT,
    created_at TEXT NOT NULL,
    judged_at TEXT,
    PRIMARY KEY (a, b)
);
CREATE INDEX dedup_pairs_pending ON dedup_pairs (verdict);

-- Audit trail of merges (what was folded into what, and why), so a bad threshold can be found and undone.
CREATE TABLE question_merges (
    id INTEGER PRIMARY KEY,
    kept_id INTEGER NOT NULL,
    dropped_id INTEGER NOT NULL,
    dropped_text TEXT NOT NULL,
    similarity REAL,
    how TEXT NOT NULL,                          -- auto | judge
    created_at TEXT NOT NULL
);

ALTER TABLE questions ADD COLUMN dedup_checked INTEGER NOT NULL DEFAULT 0;
