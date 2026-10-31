-- The parser that produced a document's questions; documents parsed by an older parser are parsed again
-- from their stored clean text (no download).
ALTER TABLE documents ADD COLUMN parser_version TEXT;
UPDATE documents SET parser_version = 'rules-v1' WHERE status IN ('parsed', 'indexed');
