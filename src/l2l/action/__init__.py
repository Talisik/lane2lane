from typing import TYPE_CHECKING, Type, Union

from ..types import UNDEFINED
from .action import Action

if TYPE_CHECKING:
    from ..lane import Lane


def GOTO(
    lane: Union[Type["Lane"], str, None],
    value=UNDEFINED,
):
    if isinstance(lane, str):
        lane = Lane.get_lane(lane)

    if lane is None:
        return

    return Action(
        name="goto",
        kwargs={
            "lane": lane,
            "value": value,
        },
    )


STOP = Action(name="stop")
