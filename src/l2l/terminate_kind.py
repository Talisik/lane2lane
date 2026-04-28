from enum import Enum


class TerminateKind(Enum):
    NO = "no"
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
