from enum import Enum


class TerminateKind(Enum):
    """How far a `terminate()` call propagates through a lane chain.

    Passed to `Lane.terminate(kind=...)`; read back via the `terminated` property.
    """

    NO = "no"

    """Not terminated (the default state)."""

    SELF = "self"

    """
    This lane and its dependencies are terminated.
    """

    NEIGHBOR = "neighbor"

    """
    The neighbor lanes are terminated.
    """

    ALL = "all"

    """
    All lanes are terminated.
    """
