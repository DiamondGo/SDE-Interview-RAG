"""Index versions. Vectors from different embedding models never share a version (they are not comparable)."""

from sde_rag import db
from sde_rag.index.embedder import configured_model_name
from sde_rag.settings import section


class IndexModelMismatch(RuntimeError):
    pass


def active_version() -> str:
    """The active version, created on first use for the configured embedding model."""
    v = db.get_meta("active_index_version")
    if v:
        return v
    with db.tx() as t:
        v = db.get_meta("active_index_version")
        if v:
            return v
        t.execute("INSERT INTO index_versions (version, embed_model, chunker_version, created_at, status) "
                  "VALUES ('v1', ?, ?, ?, 'active') ON CONFLICT(version) DO NOTHING",
                  (configured_model_name(), int(section("index").get("chunker_version", 1)), db.now_iso()))
        db.set_meta("active_index_version", "v1")
    return "v1"


def version_info(version: str):
    return db.conn().execute("SELECT * FROM index_versions WHERE version = ?", (version,)).fetchone()


def check_model(version: str | None = None) -> str:
    version = version or active_version()
    info = version_info(version)
    configured = configured_model_name()
    if info and info["embed_model"] != configured:
        raise IndexModelMismatch(
            f"index {version} was built with {info['embed_model']!r} but the configured embedding model is "
            f"{configured!r}. Mixing them would compare incompatible vectors: rebuild the index into a new version "
            f"(blue-green, milestone 3) or switch the embedder back.")
    return version
