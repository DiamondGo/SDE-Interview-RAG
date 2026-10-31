"""Identity and access control. Not enforced yet, but every retrieval and log path goes through it.

Today: NoAuthProvider (everyone is the default user) and AllowAllPolicy (everything is visible).
Swapping in real implementations must not require touching retrieval code.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Principal:
    user_id: str = "local"
    tenant_id: str = "default"
    groups: frozenset[str] = field(default_factory=frozenset)

    def scope_key(self) -> str:
        """Stable key for caches and logs: two principals with the same key see the same content."""
        return f"{self.tenant_id}:{self.user_id}:{','.join(sorted(self.groups))}"


@dataclass(frozen=True)
class ChunkFilter:
    """Pre-filter applied inside retrieval. None fields mean "no restriction"."""
    tenant_ids: frozenset[str] | None = None
    visibilities: frozenset[str] | None = None
    principals: frozenset[str] | None = None        # matched against each chunk's ACL list

    def allows(self, tenant_id: str, visibility: str, acl: str | None) -> bool:
        if self.tenant_ids is not None and tenant_id not in self.tenant_ids:
            return False
        if self.visibilities is not None and visibility not in self.visibilities:
            if self.principals is None or not acl:
                return False
            return bool(self.principals & set(json.loads(acl)))
        return True


class AuthProvider(Protocol):
    def authenticate(self, request: Any) -> Principal: ...


class AccessPolicy(Protocol):
    def visible_filter(self, p: Principal) -> ChunkFilter | None: ...
    def can_read(self, p: Principal, chunks: list[dict]) -> list[bool]: ...


class NoAuthProvider:
    def authenticate(self, request: Any = None) -> Principal:
        user = getattr(request, "user_id", None) if request is not None else None
        return Principal(user_id=user or "local")


class AllowAllPolicy:
    def visible_filter(self, p: Principal) -> ChunkFilter | None:
        return None

    def can_read(self, p: Principal, chunks: list[dict]) -> list[bool]:
        return [True] * len(chunks)


_provider: AuthProvider = NoAuthProvider()
_policy: AccessPolicy = AllowAllPolicy()


def auth_provider() -> AuthProvider:
    return _provider


def access_policy() -> AccessPolicy:
    return _policy


def configure(provider: AuthProvider | None = None, policy: AccessPolicy | None = None) -> None:
    global _provider, _policy
    if provider is not None:
        _provider = provider
    if policy is not None:
        _policy = policy


def reset() -> None:
    configure(NoAuthProvider(), AllowAllPolicy())


def safe_can_read(p: Principal, chunks: list[dict]) -> list[bool]:
    """Fail closed: a policy error hides everything instead of skipping the check."""
    try:
        ok = access_policy().can_read(p, chunks)
        return ok if len(ok) == len(chunks) else [False] * len(chunks)
    except Exception:
        return [False] * len(chunks)
