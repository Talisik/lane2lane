from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Type,
    Union,
)

if TYPE_CHECKING:
    from .lane import Lane
    from .mock import Mock

ProcessModeType = Union[Literal["all", "one"], int]
LaneReferenceType = Union[
    Type["Lane"],
    str,
    "LaneDictType",
    "Mock",
    None,
]
LaneDictType = Dict[
    int,
    LaneReferenceType,
]
