from typing import TYPE_CHECKING, Type, Union

from .mock import Mock
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
    if isinstance(lane, dict):
        return Mock(
            lanes=lane,
        )

    if isinstance(lane, Mock):
        return lane

    return get_lane(lane)
