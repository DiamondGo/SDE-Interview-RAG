import sqlite3

from sde_rag import db
from sde_rag.pipeline import dataset, stages, tagging
from sde_rag.pipeline.fetch import sync_source
from sde_rag.pipeline.indexing import index_stage
from tests.fakes import GO_MD, PREP_MD, FakeConnector


def _build():
    sync_source("public", FakeConnector("public", {"py": {"version": "1", "content": PREP_MD, "origin": "repo/a"}}))
    sync_source("login", FakeConnector("login", {"go": {"version": "1", "content": GO_MD, "origin": "forum"}}))
    stages.clean_stage(10)
    stages.parse_stage(10)
    tagging.rule_tag_stage()
    stages.merge_stage()
    index_stage(10)
    db.conn().execute("INSERT INTO query_logs (user_id, created_at, question) VALUES ('me', 'now', 'private question')")


def test_publish_excludes_local_usage_and_unpublishable_sources(tmp_path, monkeypatch):
    _build()
    monkeypatch.setattr(dataset, "source_configs", lambda: {"public": {}, "login": {"publish": False}})
    out = tmp_path / "dataset"
    m = dataset.publish_data(out)
    c = sqlite3.connect(out / "rag.db")
    assert c.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0] == 0
    assert {r[0] for r in c.execute("SELECT DISTINCT source FROM documents")} == {"public"}
    assert c.execute("SELECT COUNT(*) FROM seen_items WHERE source = 'login'").fetchone()[0] == 0
    # The GIL question was in both sources: it stays, recounted to one mention.
    assert c.execute("SELECT mention_count FROM questions WHERE text LIKE '%GIL%'").fetchone()[0] == 1
    # Questions only the login source had are gone, with their chunks and vectors.
    assert c.execute("SELECT COUNT(*) FROM questions WHERE text LIKE '%goroutine%'").fetchone()[0] == 0
    orphans = c.execute("SELECT COUNT(*) FROM chunks WHERE kind = 'question' AND question_id NOT IN "
                        "(SELECT id FROM questions)").fetchone()[0]
    assert orphans == 0
    assert c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == m["embeddings"]
    assert m["unpublished_sources"] == {"login": 1}
    assert [p.parent.name for p in (out / "raw").rglob("*.gz")] == ["public"]
    assert (out / "NOTICE.md").exists() and (out / "NOTICE.zh-CN.md").exists()
    # The working database still has everything.
    assert db.conn().execute("SELECT COUNT(*) FROM query_logs").fetchone()[0] == 1


def test_import_starts_a_fresh_data_dir(tmp_path, monkeypatch):
    _build()
    monkeypatch.setattr(dataset, "source_configs", lambda: {})
    out = tmp_path / "dataset"
    dataset.publish_data(out)
    monkeypatch.setattr(dataset, "dataset_dir", lambda: out)
    monkeypatch.setenv("SDE_DATA_DIR", str(tmp_path / "fresh"))
    db.close_all()
    res = dataset.import_data()
    assert res["raw_files"] == 2
    assert db.conn().execute("SELECT COUNT(*) FROM questions").fetchone()[0] > 0
    assert db.conn().execute("SELECT COUNT(*) FROM query_logs").fetchone()[0] == 0
