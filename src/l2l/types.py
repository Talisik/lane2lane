from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Type,
    Union,
)

if TYPE_CHECKING:
    from ._lane_core import _LaneCore
    from .mock import Mock

ProcessModeType = Union[Literal["all", "one"], int]
LaneReferenceType = Union[
    # the shared base so both Lane and AsyncLane subclasses are accepted
    Type["_LaneCore"],
    str,
    "LaneDictType",
    "Mock",
    None,
]
LaneDictType = Dict[
    int,
    LaneReferenceType,
]
