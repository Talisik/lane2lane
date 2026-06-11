from abc import ABC

from .async_lane import AsyncLane


class AsyncPrimaryLane(AsyncLane, ABC):
    @classmethod
    def primary(cls) -> bool:
        return True
