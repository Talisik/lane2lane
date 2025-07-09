from typing import (
    TYPE_CHECKING,
    Dict,
    Type,
    Union,
)

if TYPE_CHECKING:
    from .directions import LaneDirection
    from .lane import Lane

LaneReferenceType = Union[Type["Lane"], str, "LaneDirection", "LaneDictType", None]
LaneDictType = Dict[
    int,
    LaneReferenceType,
]
