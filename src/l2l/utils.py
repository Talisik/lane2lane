from typing import TYPE_CHECKING, Type, Union

from .mock import Mock
from .types import LaneReferenceType

if TYPE_CHECKING:
    from .lane import Lane


def get_lane(value: Union[Type["Lane"], str, None]):
    """Resolves a lane reference to a class: ``None``→``None``, ``str``→looked-up
    `Lane`, a class→itself."""
    if value is None:
        return None

    if isinstance(value, str):
        return Lane.get_lane(value)

    return value


def from_lane_reference(lane: LaneReferenceType):
    """Normalizes a `lanes`-dict value to a runnable: ``dict``→`Mock`, `Mock`→
    itself, else resolved via `get_lane`."""
    if isinstance(lane, dict):
        return Mock(
            lanes=lane,
        )

    if isinstance(lane, Mock):
        return lane

    return get_lane(lane)
