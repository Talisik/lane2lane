from abc import ABC

from .lane import Lane


class PrimaryLane(Lane, ABC):
    @classmethod
    def primary(cls) -> bool:
        return True
