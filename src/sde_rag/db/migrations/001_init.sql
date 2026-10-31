CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per configured source: incremental state and health.
CREATE TABLE sources (
    name TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT '{}',          -- connector cursor / high-water marks (JSON)
    last_run_at TEXT,
    last_success_at TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);

-- Everything a connector has listed, so nothing is downloaded twice.
CREATE TABLE seen_items (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url_norm TEXT,
    title TEXT,
    version TEXT,                              -- blob sha / edited_at: a different value means changed
    etag TEXT,
    last_modified TEXT,
    content_hash TEXT,
    doc_id INTEGER,                            -- document holding the content (may belong to another source)
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    UNIQUE (source, external_id)
);
CREATE INDEX seen_items_url ON seen_items (url_norm);

CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    origin TEXT NOT NULL,                      -- finer than source, e.g. the GitHub repo; counted for "distinct sources"
    external_id TEXT NOT NULL,
    url TEXT NOT NULL,
    url_norm TEXT NOT NULL,
    alt_url_norm TEXT,                         -- canonical URL declared by the page, if different
    title TEXT NOT NULL DEFAULT '',
    author TEXT,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    kind TEXT NOT NULL,                        -- prep | experience
    content_type TEXT NOT NULL,                -- markdown | html
    raw_path TEXT,
    raw_hash TEXT,
    clean_text TEXT,
    clean_hash TEXT,
    cleaner_version INTEGER,
    lang TEXT,
    hints TEXT NOT NULL DEFAULT '{}',          -- category/tag hints from the source config (JSON)
    citations INTEGER NOT NULL DEFAULT 0,      -- times another source linked to this content
    status TEXT NOT NULL,                      -- fetched | cleaned | parsed | indexed | unchanged | skipped | failed | dead
    stage TEXT,                                -- stage that failed, for failed/dead
    attempts INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    last_error TEXT,
    skip_reason TEXT,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    visibility TEXT NOT NULL DEFAULT 'public',
    acl TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE (source, external_id)
);
CREATE INDEX documents_status ON documents (status);
CREATE INDEX documents_url ON documents (url_norm);
CREATE INDEX documents_alt_url ON documents (alt_url_norm);

-- Canonical questions: every mention of the same question points here.
CREATE TABLE questions (
    id INTEGER PRIMARY KEY,
    norm_hash TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',            -- representative source answer
    section_path TEXT NOT NULL DEFAULT '',
    category TEXT,
    difficulty TEXT,
    mention_count INTEGER NOT NULL DEFAULT 0,
    distinct_sources INTEGER NOT NULL DEFAULT 0,
    companies TEXT NOT NULL DEFAULT '[]',
    first_seen TEXT,
    last_seen TEXT,
    heat REAL NOT NULL DEFAULT 0,
    tag_status TEXT,                            -- NULL (untagged) | rule | llm
    tag_version INTEGER,
    stats_dirty INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX questions_tag_status ON questions (tag_status);
CREATE INDEX questions_category ON questions (category);

CREATE TABLE question_mentions (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES questions (id),
    doc_id INTEGER NOT NULL REFERENCES documents (id),
    mention_hash TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    section_path TEXT NOT NULL DEFAULT '',
    ordinal INTEGER NOT NULL,
    company TEXT,
    role TEXT,
    level TEXT,
    round TEXT,
    year INTEGER,
    extractor TEXT NOT NULL,                    -- rules-v1 | llm:<fingerprint>
    created_at TEXT NOT NULL
);
CREATE INDEX question_mentions_q ON question_mentions (question_id);
CREATE INDEX question_mentions_doc ON question_mentions (doc_id);

CREATE TABLE question_tags (
    question_id INTEGER NOT NULL REFERENCES questions (id),
    tag TEXT NOT NULL,
    origin TEXT NOT NULL,                       -- rule | hint | llm
    PRIMARY KEY (question_id, tag)
);
CREATE INDEX question_tags_tag ON question_tags (tag);

CREATE TABLE tag_candidates (
    keyword TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);

CREATE TABLE index_versions (
    version TEXT PRIMARY KEY,
    embed_model TEXT NOT NULL,
    dim INTEGER,
    chunker_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL                        -- building | active | retired
);

-- Retrieval units. kind=question: one per canonical question; kind=doc: paragraphs of a document.
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    index_version TEXT NOT NULL,
    kind TEXT NOT NULL,
    question_id INTEGER NOT NULL DEFAULT 0,     -- 0, not NULL, so the UNIQUE constraint applies
    doc_id INTEGER NOT NULL DEFAULT 0,
    ordinal INTEGER NOT NULL DEFAULT 0,
    embed_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    category TEXT,
    tags TEXT NOT NULL DEFAULT '',              -- space-separated
    tenant_id TEXT NOT NULL DEFAULT 'default',
    visibility TEXT NOT NULL DEFAULT 'public',
    acl TEXT,
    embedded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE (index_version, kind, question_id, doc_id, ordinal)
);
CREATE INDEX chunks_question ON chunks (question_id);
CREATE INDEX chunks_doc ON chunks (doc_id);
CREATE INDEX chunks_pending ON chunks (index_version, embedded);

CREATE VIRTUAL TABLE chunk_fts USING fts5(body, tags, tokenize = 'porter unicode61');

CREATE TABLE embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks (id),
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE ingest_runs (
    id INTEGER PRIMARY KEY,
    run_key TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    heartbeat_at TEXT,
    status TEXT NOT NULL,                       -- running | done | failed | stalled
    summary TEXT
);
CREATE INDEX ingest_runs_key ON ingest_runs (run_key);

CREATE TABLE run_locks (
    name TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE query_logs (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    intent TEXT,
    filters TEXT,
    answer TEXT,
    citations TEXT,
    degraded INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER,
    trace TEXT,
    feedback INTEGER
);

CREATE TABLE alerts (
    key TEXT PRIMARY KEY,
    message TEXT NOT NULL,
    first_at TEXT NOT NULL,
    last_notified_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE eval_runs (
    id INTEGER PRIMARY KEY,
    suite TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config TEXT,
    metrics TEXT NOT NULL,
    passed INTEGER
);
