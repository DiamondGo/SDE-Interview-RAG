"""BM25 keyword search (SQLite FTS5). Filters, including access control, are applied inside the query."""

import re

from sde_rag import db
from sde_rag.auth import ChunkFilter

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*[A-Za-z0-9+#]|[A-Za-z0-9]|[\u4e00-\u9fff]+")
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in
into is it its me my of on or our should so than that the their them then there these they this to was we were
what when where which who whom why will with would you your about explain describe difference between vs versus
question questions interview interviews common asked ask tell give example""".split())


def match_expression(queries: list[str]) -> str:
    seen, terms = set(), []
    for q in queries:
        for tok in _TOKEN.findall(q.lower()):
            tok = tok.strip(".-")
            if not tok or tok in STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            terms.append('"' + tok.replace('"', '""') + '"')
    return " OR ".join(terms[:32])


def filter_sql(*, kinds=None, category=None, chunk_filter: ChunkFilter | None = None,
               alias: str = "c") -> tuple[str, list]:
    clauses, args = [], []
    if kinds:
        clauses.append(f"{alias}.kind IN ({','.join('?' * len(kinds))})")
        args += list(kinds)
    if category:
        clauses.append(f"{alias}.category = ?")
        args.append(category)
    if chunk_filter is not None:
        if chunk_filter.tenant_ids is not None:
            clauses.append(f"{alias}.tenant_id IN ({','.join('?' * len(chunk_filter.tenant_ids)) or 'NULL'})")
            args += sorted(chunk_filter.tenant_ids)
        if chunk_filter.visibilities is not None:
            vis = f"{alias}.visibility IN ({','.join('?' * len(chunk_filter.visibilities)) or 'NULL'})"
            args += sorted(chunk_filter.visibilities)
            if chunk_filter.principals:
                vis = (f"({vis} OR EXISTS (SELECT 1 FROM json_each(COALESCE({alias}.acl, '[]')) "
                       f"WHERE value IN ({','.join('?' * len(chunk_filter.principals))})))")
                args += sorted(chunk_filter.principals)
            clauses.append(vis)
    return (" AND " + " AND ".join(clauses)) if clauses else "", args


def search(version: str, queries: list[str], k: int, *, kinds=None, category=None, tags=None,
           chunk_filter: ChunkFilter | None = None) -> list[tuple[int, float]]:
    expr = match_expression(queries)
    if not expr:
        return []
    if tags:
        tag_expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in tags)
        expr = f"({expr}) AND tags : ({tag_expr})"
    where, args = filter_sql(kinds=kinds, category=category, chunk_filter=chunk_filter)
    rows = db.conn().execute(
        f"SELECT c.id, bm25(chunk_fts, 1.0, 0.3) AS score FROM chunk_fts JOIN chunks c ON c.id = chunk_fts.rowid "
        f"WHERE chunk_fts MATCH ? AND c.index_version = ?{where} ORDER BY score LIMIT ?",
        [expr, version, *args, k]).fetchall()
    return [(r[0], -float(r[1])) for r in rows]
