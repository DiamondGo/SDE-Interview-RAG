-- The reviewer: is an answer's domain knowledge accurate and its solution correct? Jobs are queued, run by
-- priority within a daily quota, and retried when a call fails.
CREATE TABLE review_jobs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                         -- query_answer | source_answer
    subject_id INTEGER NOT NULL,                -- query_logs.id | questions.id
    priority INTEGER NOT NULL,                  -- 1: asked for by the user, 2: query answer, 4: source answer
    status TEXT NOT NULL,                       -- pending | running | done | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    context TEXT,                               -- what the answer was written from (for query answers)
    verdict TEXT,                               -- correct | minor_issues | incorrect | unverifiable
    result TEXT,                                -- the full review (JSON)
    reviewer TEXT,                              -- role fingerprint
    error TEXT,
    human_label TEXT,                           -- agree | disagree (calibration, section 7.4)
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX review_jobs_queue ON review_jobs (status, priority, id);
CREATE INDEX review_jobs_subject ON review_jobs (kind, subject_id);

-- The latest review of a question's representative source answer.
ALTER TABLE questions ADD COLUMN answer_verdict TEXT;
ALTER TABLE questions ADD COLUMN answer_review_id INTEGER;

ALTER TABLE query_logs ADD COLUMN review_job_id INTEGER;
