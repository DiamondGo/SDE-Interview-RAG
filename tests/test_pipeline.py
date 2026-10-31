import shutil

import numpy as np

from sde_rag import db
from sde_rag.index import embedder as embedder_mod
from sde_rag.index.vectors import get_index, load_vectors, reset_indexes, snapshot_dir
from sde_rag.index.versions import active_version
from sde_rag.pipeline import stages, tagging
from sde_rag.pipeline.fetch import sync_source
from sde_rag.pipeline.indexing import index_stage, split_paragraphs
from tests.fakes import EXPERIENCE_MD, GO_MD, PREP_MD, FakeConnector


def _run_stages():
    stages.clean_stage(100)
    stages.parse_stage(100)
    tagging.rule_tag_stage()
    stages.merge_stage()
    return index_stage(100)


def test_second_run_downloads_nothing():
    docs = {"py": {"version": "v1", "content": PREP_MD}, "go": {"version": "v1", "content": GO_MD}}
    conn = FakeConnector("repo", docs)
    first = sync_source("repo", conn)
    assert first["downloaded"] == 2 and first["new"] == 2
    conn.fetched.clear()
    second = sync_source("repo", conn)
    assert second["seen"] == 2 and second["downloaded"] == 0 and conn.fetched == []


def test_changed_version_redownloads_and_same_content_counts_as_duplicate():
    docs = {"py": {"version": "v1", "content": PREP_MD}}
    conn = FakeConnector("repo", docs)
    sync_source("repo", conn)
    docs["py"]["version"] = "v2"                       # version moved, content identical
    st = sync_source("repo", conn)
    assert st["downloaded"] == 1 and st["duplicate_downloads"] == 1 and st["changed"] == 0
    docs["py"].update(version="v3", content=PREP_MD + "\n## 4. What is a metaclass?\nA class of a class.\n")
    st = sync_source("repo", conn)
    assert st["changed"] == 1
    assert db.conn().execute("SELECT status FROM documents").fetchone()[0] == "fetched"


def test_links_to_known_content_are_citations_not_downloads_of_the_article():
    sync_source("devto", FakeConnector("devto", {"a1": {"version": "1", "content": EXPERIENCE_MD, "kind": "experience",
                                                       "url": "https://dev.to/x/my-onsite"}}))
    hn = FakeConnector("hn", {"h1": {"version": "1", "content": "# discussion\n" + EXPERIENCE_MD, "kind": "experience",
                                     "outbound_url": "https://www.dev.to/x/my-onsite/?utm_source=hn"}})
    st = sync_source("hn", hn)
    assert st["cross_source"] == 1
    assert db.conn().execute("SELECT citations FROM documents WHERE source = 'devto'").fetchone()[0] == 1


def test_failed_and_not_modified_documents():
    conn = FakeConnector("repo", {"bad": {"version": "1", "fail": True}, "nm": {"version": "1", "not_modified": True}})
    st = sync_source("repo", conn)
    assert st["failed"] == 1 and st["not_modified"] == 1 and conn.finalized_failed == ["bad"]
    assert db.conn().execute("SELECT consecutive_failures FROM sources WHERE name = 'repo'").fetchone()[0] == 0


def test_stages_build_questions_merge_across_sources_and_index():
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD, "origin": "repo/a",
                                                "hints": {"category": "language", "tags": ["python"]}}}))
    sync_source("b", FakeConnector("b", {"go": {"version": "1", "content": GO_MD, "origin": "repo/b"},
                                         "exp": {"version": "1", "content": EXPERIENCE_MD, "kind": "experience"}}))
    stats = _run_stages()
    c = db.conn()
    gil = c.execute("SELECT * FROM questions WHERE text LIKE '%GIL%'").fetchone()
    assert gil["mention_count"] == 2 and gil["distinct_sources"] == 2     # merged across repos
    assert gil["category"] == "language"
    assert set(tagging.question_tags(gil["id"])) >= {"python"}
    goroutine = c.execute("SELECT id, category FROM questions WHERE text = 'What is a goroutine?'").fetchone()
    assert "golang" in tagging.question_tags(goroutine["id"]) and goroutine["category"] == "language"
    assert stats["doc_chunks"] >= 1 and stats["embedded"] == stats["vectors"]
    assert c.execute("SELECT COUNT(*) FROM documents WHERE status = 'indexed'").fetchone()[0] == 3
    # Re-parsing a changed document drops its old questions.
    c.execute("UPDATE documents SET clean_text = ?, status = 'cleaned' WHERE external_id = 'go'",
              ("# Go\n## What is a goroutine?\nStill here.\n",))
    stages.parse_stage(100)
    assert stages.merge_stage()["removed"] == 1           # "How do goroutines communicate?" is gone
    assert c.execute("SELECT mention_count FROM questions WHERE id = ?", (gil["id"],)).fetchone()[0] == 1


def test_restart_does_not_reembed_and_snapshot_recovers(monkeypatch):
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    _run_stages()
    version = active_version()
    ids, mat, how = load_vectors(version)
    assert how == "snapshot" and len(ids) == 3

    calls = []
    real = embedder_mod.get_embedder()
    with monkeypatch.context() as m:
        m.setattr(real, "embed_passages", lambda texts: calls.append(len(texts)) or np.zeros((0, 64)))
        assert index_stage(100)["embedded"] == 0 and calls == []      # nothing pending: no embedding at all

    # A new question: the snapshot is extended with just that row.
    c = db.conn()
    c.execute("UPDATE documents SET clean_text = clean_text || ?, status = 'cleaned'",
              ("\n## 4. What is a metaclass?\nA class of a class.\n",))
    stages.parse_stage(100)
    tagging.rule_tag_stage()
    stages.merge_stage()
    assert index_stage(100)["snapshot"] == "extended"

    # Lost or corrupted snapshot: rebuilt from the database, not re-embedded.
    shutil.rmtree(snapshot_dir(version))
    ids, _, how = load_vectors(version)
    assert how == "rebuilt" and len(ids) == 4
    (snapshot_dir(version) / "manifest.json").write_text("{broken")
    assert load_vectors(version)[2] == "rebuilt"
    reset_indexes()
    assert len(get_index(version).ids) == 4


def test_parser_upgrade_reparses_stored_text_without_downloading_or_reembedding(monkeypatch):
    conn = FakeConnector("a", {"py": {"version": "1", "content": PREP_MD},
                               "exp": {"version": "1", "content": EXPERIENCE_MD, "kind": "experience"}})
    sync_source("a", conn)
    _run_stages()
    c = db.conn()
    before = {r[0] for r in c.execute("SELECT id FROM chunks")}
    c.execute("UPDATE documents SET parser_version = 'rules-v0'")
    conn.fetched.clear()
    calls = []
    real = embedder_mod.get_embedder()
    with monkeypatch.context() as m:
        m.setattr(real, "embed_passages", lambda texts: calls.append(len(texts)) or np.zeros((0, 64)))
        assert stages.parse_stage(100)["reparse_queued"] == 2
        tagging.rule_tag_stage()
        stages.merge_stage()
        st = index_stage(100)
    assert conn.fetched == [] and calls == [] and st["embedded"] == 0 and st["doc_chunks_kept"] >= 1
    assert {r[0] for r in c.execute("SELECT id FROM chunks")} == before
    assert c.execute("SELECT COUNT(*) FROM documents WHERE status = 'indexed'").fetchone()[0] == 2
    assert stages.parse_stage(100)["reparse_queued"] == 0


def test_taxonomy_change_retags_rule_tags_and_keeps_llm_tags():
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    _run_stages()
    c = db.conn()
    llm_q, *rule_qs = [r[0] for r in c.execute("SELECT id FROM questions ORDER BY id")]
    c.execute("UPDATE questions SET tag_status = 'llm' WHERE id = ?", (llm_q,))
    c.execute("UPDATE questions SET tag_version = 0")
    assert tagging.rule_tag_stage()["rule_tagged"] == len(rule_qs)
    assert tagging.rule_tag_stage()["rule_tagged"] == 0


def test_failures_go_to_dead_letter_after_max_attempts(monkeypatch):
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    doc_id = db.conn().execute("SELECT id FROM documents").fetchone()[0]
    statuses = [stages.mark_failed(doc_id, "clean", "boom") for _ in range(5)]
    assert statuses == ["failed"] * 4 + ["dead"]


def test_split_paragraphs_keeps_code_blocks_whole():
    text = "intro\n\n```\n" + "line\n\n" * 50 + "```\n\nafter"
    parts = split_paragraphs(text, 100)
    assert any(p.count("```") == 2 for p in parts)
