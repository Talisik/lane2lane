from abc import ABC, abstractmethod
from typing import Iterable, final

from .lane import Lane


class Subscriber(Lane, ABC):
    @abstractmethod
    def get_payloads(self) -> Iterable:
        pass

    @final
    def process(self, **kwargs):
        self.set(
            "payloads",
            [*self.get_payloads()],
        )
