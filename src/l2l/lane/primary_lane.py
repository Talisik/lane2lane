from abc import ABC
from typing import TypeVar

from .lane import Lane

T = TypeVar("T")


class PrimaryLane(Lane[T], ABC):
    @classmethod
    def primary(cls) -> bool:
        return True
