from typing import Type, Union

from l2l.types import UNDEFINED

from ..lane import Lane
from .action import Action


def GOTO(
    lane: Union[Type[Lane], str, None],
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
