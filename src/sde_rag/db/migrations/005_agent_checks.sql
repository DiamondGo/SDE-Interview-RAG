-- Agent capability checks (DESIGN 8): can each LLM role, as configured on this machine, still do its job?
-- A result belongs to the role's configuration fingerprint: a different model, backend or behavior-affecting
-- parameter has no passing record until it is checked again.
CREATE TABLE agent_checks (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    level TEXT NOT NULL,                        -- quick | full
    cases INTEGER NOT NULL,
    metrics TEXT NOT NULL,                      -- JSON: metric -> value
    checks TEXT NOT NULL,                       -- JSON: [{metric, value, op, threshold, ok, relaxed}]
    passed INTEGER NOT NULL,
    failures TEXT,                              -- JSON: examples of the cases that went wrong
    duration_s REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX agent_checks_role ON agent_checks (role, fingerprint, id);
