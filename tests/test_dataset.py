import gzip
import json
import shutil
import sqlite3

import pytest

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
    m = dataset.publish_data(out, upload=False)
    assert m["database"]["release"] is None
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


class FakeGh:
    """Releases on a fake GitHub: created ones are listed newest first, deletions are recorded."""

    def __init__(self, existing: list[str]):
        self.releases = [{"tagName": t, "createdAt": f"2026-01-0{i + 1}T00:00:00Z"} for i, t in enumerate(existing)]
        self.calls: list[tuple] = []

    def __call__(self, *args):
        self.calls.append(args)
        if args[:2] == ("release", "create"):
            self.releases.append({"tagName": args[2], "createdAt": "2026-12-31T00:00:00Z"})
        elif args[:2] == ("release", "list"):
            return json.dumps(self.releases)
        elif args[:2] == ("release", "delete"):
            self.releases = [r for r in self.releases if r["tagName"] != args[2]]
        return ""


def test_publish_uploads_a_release_and_keeps_only_the_newest(tmp_path, monkeypatch):
    _build()
    monkeypatch.setattr(dataset, "source_configs", lambda: {})
    gh = FakeGh(["dataset-a", "dataset-b", "dataset-c", "v1.0"])
    monkeypatch.setattr(dataset, "_gh", gh)
    m = dataset.publish_data(tmp_path / "dataset")
    tag = m["database"]["release"]
    assert tag.startswith("dataset-") and m["database"]["url"].endswith(f"/{tag}/rag.db.gz")
    assert m["releases_deleted"] == ["dataset-a"]                 # 3 kept: the new one, c and b; v1.0 untouched
    assert {r["tagName"] for r in gh.releases} == {"dataset-b", "dataset-c", tag, "v1.0"}
    manifest = json.loads((tmp_path / "dataset" / "MANIFEST.json").read_text())
    assert manifest["database"]["sha256"] == m["database"]["sha256"]


def test_import_downloads_and_verifies_the_database(tmp_path, monkeypatch):
    _build()
    monkeypatch.setattr(dataset, "source_configs", lambda: {})
    monkeypatch.setattr(dataset, "_gh", FakeGh([]))
    out = tmp_path / "dataset"
    dataset.publish_data(out)
    asset = tmp_path / "asset.gz"
    (out / "rag.db.gz").replace(asset)
    (out / "rag.db").unlink()                                      # a fresh clone has no database
    downloads = []
    monkeypatch.setattr(dataset, "_download", lambda url, dst: downloads.append(url) or shutil.copy(asset, dst))
    monkeypatch.setattr(dataset, "dataset_dir", lambda: out)
    with monkeypatch.context() as m:
        m.setenv("SDE_DATA_DIR", str(tmp_path / "fresh"))
        db.close_all()
        res = dataset.import_data()
        assert len(downloads) == 1 and res["raw_files"] == 2
        assert db.conn().execute("SELECT COUNT(*) FROM questions").fetchone()[0] > 0
        assert db.conn().execute("SELECT COUNT(*) FROM query_logs").fetchone()[0] == 0
        # A corrupted download is refused.
        asset.write_bytes(gzip.compress(b"not the database"))
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            dataset.import_data(force=True)
        db.close_all()
