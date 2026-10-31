class AgentError(RuntimeError):
    pass


class ConfigError(AgentError):
    pass


class StallError(AgentError):
    """No progress within the allowed time: the call is abandoned, not merely slow."""


class RunawayError(AgentError):
    """Output exceeded its budget or started repeating itself."""


class BackendUnavailable(AgentError):
    """Misconfigured backend, or circuit open after failed health probes."""


class AllAttemptsFailed(AgentError):
    def __init__(self, role: str, errors: list[str]):
        self.role = role
        self.errors = errors
        super().__init__(f"role {role!r}: every backend failed: " + " | ".join(errors))
