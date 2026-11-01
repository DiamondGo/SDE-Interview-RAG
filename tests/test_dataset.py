import gzip
import io
import json
import shutil
import sqlite3
import tarfile

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
    assert not (out / "raw").exists()
    with tarfile.open(out / dataset.RAW_ASSET) as archive:
        assert [m.name.split("/")[1] for m in archive.getmembers()] == ["public"]
    assert m["raw_archive"]["sha256"] == dataset._sha256(out / dataset.RAW_ASSET)
    assert m["raw_archive"]["release"] is None
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
    assert m["raw_archive"]["release"] == tag
    assert m["raw_archive"]["url"].endswith(f"/{tag}/raw.tar.gz")
    create = next(args for args in gh.calls if args[:2] == ("release", "create"))
    assert create[3:5] == (str(tmp_path / "dataset" / dataset.ASSET), str(tmp_path / "dataset" / dataset.RAW_ASSET))
    assert m["releases_deleted"] == ["dataset-a"]                 # 3 kept: the new one, c and b; v1.0 untouched
    assert {r["tagName"] for r in gh.releases} == {"dataset-b", "dataset-c", tag, "v1.0"}
    manifest = json.loads((tmp_path / "dataset" / "MANIFEST.json").read_text())
    assert manifest["database"]["sha256"] == m["database"]["sha256"]


@pytest.mark.parametrize("with_raw", [False, True])
def test_import_downloads_and_verifies_the_database(tmp_path, monkeypatch, with_raw):
    _build()
    monkeypatch.setattr(dataset, "source_configs", lambda: {})
    monkeypatch.setattr(dataset, "_gh", FakeGh([]))
    out = tmp_path / "dataset"
    dataset.publish_data(out)
    asset = tmp_path / "asset.gz"
    (out / "rag.db.gz").replace(asset)
    raw_asset = tmp_path / "raw-asset.tar.gz"
    (out / dataset.RAW_ASSET).replace(raw_asset)
    (out / "rag.db").unlink()                                      # a fresh clone has no database
    downloads = []
    def download(url, dst):
        downloads.append(url)
        shutil.copy(raw_asset if url.endswith(dataset.RAW_ASSET) else asset, dst)
    monkeypatch.setattr(dataset, "_download", download)
    monkeypatch.setattr(dataset, "dataset_dir", lambda: out)
    with monkeypatch.context() as m:
        m.setenv("SDE_DATA_DIR", str(tmp_path / "fresh"))
        db.close_all()
        res = dataset.import_data(with_raw=with_raw)
        assert len(downloads) == (2 if with_raw else 1)
        assert res["raw_files"] == (2 if with_raw else 0)
        assert (tmp_path / "fresh" / "raw").exists() == with_raw
        assert db.conn().execute("SELECT COUNT(*) FROM questions").fetchone()[0] > 0
        assert db.conn().execute("SELECT COUNT(*) FROM query_logs").fetchone()[0] == 0
        if with_raw:
            for row in db.conn().execute("SELECT raw_path FROM documents"):
                with gzip.open(tmp_path / "fresh" / row[0], "rt") as f:
                    assert f.read() in (PREP_MD, GO_MD)
        with pytest.raises(FileExistsError):
            dataset.import_data()
        # A corrupted download is refused.
        asset.write_bytes(gzip.compress(b"not the database"))
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            dataset.import_data(force=True)
        assert db.conn().execute("SELECT COUNT(*) FROM questions").fetchone()[0] > 0
        db.close_all()


def _local_dataset(tmp_path, monkeypatch):
    _build()
    monkeypatch.setattr(dataset, "source_configs", lambda: {})
    out = tmp_path / "dataset"
    dataset.publish_data(out, upload=False)
    monkeypatch.setattr(dataset, "dataset_dir", lambda: out)
    return out


def test_database_only_import_ignores_legacy_raw_files(tmp_path, monkeypatch):
    out = _local_dataset(tmp_path, monkeypatch)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    manifest.pop("raw_archive")
    (out / "MANIFEST.json").write_text(json.dumps(manifest))
    legacy = out / "raw" / "public" / "old.gz"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("version https://git-lfs.github.com/spec/v1\n")
    with monkeypatch.context() as m:
        m.setenv("SDE_DATA_DIR", str(tmp_path / "fresh"))
        db.close_all()
        assert dataset.import_data()["raw_files"] == 0
        assert not (tmp_path / "fresh" / "raw").exists()
        db.close_all()


def test_import_requires_release_metadata_for_raw_files(tmp_path, monkeypatch):
    out = _local_dataset(tmp_path, monkeypatch)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    manifest.pop("raw_archive")
    (out / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="no raw archive"):
        dataset.import_data(force=True, with_raw=True)
    assert db.conn().execute("SELECT question FROM query_logs").fetchone()[0] == "private question"


def test_import_rejects_corrupt_raw_download_before_replacing_database(tmp_path, monkeypatch):
    out = _local_dataset(tmp_path, monkeypatch)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    manifest["raw_archive"]["url"] = "https://example.invalid/raw.tar.gz"
    (out / "MANIFEST.json").write_text(json.dumps(manifest))
    (out / dataset.RAW_ASSET).unlink()
    monkeypatch.setattr(dataset, "_download", lambda url, dst: dst.write_bytes(b"corrupt"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dataset.import_data(force=True, with_raw=True)
    assert db.conn().execute("SELECT question FROM query_logs").fetchone()[0] == "private question"


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "device", "duplicate", "extra", "missing"])
def test_import_rejects_unsafe_or_mismatched_archive(tmp_path, monkeypatch, kind):
    out = _local_dataset(tmp_path, monkeypatch)
    raw_file = out / dataset.RAW_ASSET
    with tarfile.open(raw_file) as archive:
        originals = [(m.name, archive.extractfile(m).read()) for m in archive.getmembers()]
    with tarfile.open(raw_file, "w:gz") as archive:
        for name, body in originals[1:] if kind == "missing" else originals:
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
        if kind != "missing":
            name = {"traversal": "raw/../../escape.gz", "absolute": str(tmp_path / "escape.gz"),
                    "duplicate": originals[0][0]}.get(kind, "raw/public/extra.gz")
            member = tarfile.TarInfo(name)
            if kind in ("symlink", "hardlink"):
                member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                member.linkname = str(tmp_path / "escape.gz")
            elif kind == "device":
                member.type = tarfile.CHRTYPE
            archive.addfile(member, io.BytesIO(b""))
    manifest = json.loads((out / "MANIFEST.json").read_text())
    manifest["raw_archive"]["sha256"] = dataset._sha256(raw_file)
    (out / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        dataset.import_data(force=True, with_raw=True)
    assert not (tmp_path / "escape.gz").exists()
    assert db.conn().execute("SELECT question FROM query_logs").fetchone()[0] == "private question"


@pytest.mark.parametrize("name", ["../private.gz", "raw/../private.gz", "/raw/public/test.gz", "raw\\public\\test.gz"])
def test_pack_rejects_unsafe_raw_paths(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe raw file path"):
        dataset._pack_raw({name}, tmp_path, tmp_path / dataset.RAW_ASSET)
    assert not (tmp_path / dataset.RAW_ASSET).exists()
