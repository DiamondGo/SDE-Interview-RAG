from sde_rag import auth, db, retrieval
from sde_rag.auth import ChunkFilter, Principal, safe_can_read
from sde_rag.index.vectors import reset_indexes
from sde_rag.pipeline import stages, tagging
from sde_rag.pipeline.fetch import sync_source
from sde_rag.pipeline.indexing import index_stage
from tests.fakes import GO_MD, PREP_MD, FakeConnector


class TenantPolicy:
    """Users see only their tenant's chunks; one chunk id may additionally be denied at read time."""

    def __init__(self, deny_chunk: int | None = None):
        self.deny_chunk = deny_chunk

    def visible_filter(self, p: Principal):
        return ChunkFilter(tenant_ids=frozenset({p.tenant_id}))

    def can_read(self, p, chunks):
        return [c["chunk_id" if "chunk_id" in c else "id"] != self.deny_chunk for c in chunks]


class BrokenPolicy:
    def visible_filter(self, p):
        return None

    def can_read(self, p, chunks):
        raise RuntimeError("policy service down")


def _index_two_tenants():
    sync_source("a", FakeConnector("a", {"py": {"version": "1", "content": PREP_MD}}))
    sync_source("b", FakeConnector("b", {"go": {"version": "1", "content": GO_MD}}))
    stages.clean_stage(10)
    stages.parse_stage(10)
    tagging.rule_tag_stage()
    stages.merge_stage()
    index_stage(10)
    c = db.conn()
    # Questions only in the Go doc belong to tenant "t2".
    c.execute("UPDATE chunks SET tenant_id = 't2' WHERE question_id IN (SELECT question_id FROM question_mentions m "
              "JOIN documents d ON d.id = m.doc_id GROUP BY question_id HAVING MAX(d.source) = 'b' AND MIN(d.source) = 'b')")
    db.set_meta("index_epoch", "changed")
    reset_indexes()


def test_prefilter_applies_in_both_bm25_and_vector_search():
    _index_two_tenants()
    auth.configure(policy=TenantPolicy())
    hits = retrieval.hybrid(Principal(tenant_id="default"), ["goroutine channels"], k=20)
    tenants = {r[0] for r in db.conn().execute(
        f"SELECT tenant_id FROM chunks WHERE id IN ({','.join(str(h.chunk_id) for h in hits) or 'NULL'})")}
    assert tenants <= {"default"}
    other = retrieval.hybrid(Principal(tenant_id="t2"), ["goroutine channels"], k=20)
    assert other and all(db.conn().execute("SELECT tenant_id FROM chunks WHERE id = ?", (h.chunk_id,)).fetchone()[0]
                         == "t2" for h in other)


def test_read_check_drops_denied_chunks_and_fails_closed():
    _index_two_tenants()
    hits = retrieval.hybrid(Principal(), ["GIL python"], k=5)
    denied = hits[0].chunk_id
    auth.configure(policy=TenantPolicy(deny_chunk=denied))
    trace = {}
    items = retrieval.load_context(Principal(), hits, 5, trace=trace)
    assert denied not in [i["chunk_id"] for i in items] and trace["denied_by_policy"] == 1
    auth.configure(policy=BrokenPolicy())
    assert safe_can_read(Principal(), [{"id": 1}, {"id": 2}]) == [False, False]
    assert retrieval.load_context(Principal(), hits, 5) == []


def test_scope_keys_differ_by_principal():
    assert Principal(user_id="a").scope_key() != Principal(user_id="b").scope_key()
    assert Principal(groups=frozenset({"x", "y"})).scope_key() == Principal(groups=frozenset({"y", "x"})).scope_key()


def test_chunk_filter_acl():
    f = ChunkFilter(visibilities=frozenset({"public"}), principals=frozenset({"team-a"}))
    assert f.allows("default", "public", None)
    assert f.allows("default", "private", '["team-a"]')
    assert not f.allows("default", "private", '["team-b"]')
