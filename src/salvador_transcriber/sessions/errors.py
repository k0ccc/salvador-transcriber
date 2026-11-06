class SessionError(RuntimeError):
    """Base class for expected session lifecycle errors."""


class SessionStateError(SessionError):
    pass


class CallNotFoundError(SessionError):
    pass


class CallAlreadyExistsError(SessionError):
    pass


class HotelMismatchError(SessionError):
    pass


class RoleOwnershipError(SessionError):
    pass
