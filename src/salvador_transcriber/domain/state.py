from enum import StrEnum


class Role(StrEnum):
    GUEST = "guest"
    OPERATOR = "operator"


class CallState(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class ChannelState(StrEnum):
    CONNECTING = "connecting"
    READY = "ready"
    DEGRADED = "degraded"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"
