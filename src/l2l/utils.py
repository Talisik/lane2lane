from typing import TYPE_CHECKING, Type, Union

from .directions import Fork, LaneDirection
from .types import LaneReferenceType

if TYPE_CHECKING:
    from .lane import Lane


def get_lane(value: Union[Type["Lane"], str, None]):
    if value is None:
        return None

    if isinstance(value, str):
        return Lane.get_lane(value)

    return value


def from_lane_reference(lane: LaneReferenceType):
    if isinstance(lane, LaneDirection):
        return lane

    if isinstance(lane, dict):
        return Fork(
            lanes=lane,
            isolated=False,
        )

    return get_lane(lane)
