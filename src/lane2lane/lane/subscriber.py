from abc import ABC, abstractmethod
from typing import Iterable, final

from ..constants import LOGGER
from .lane import Lane


class Subscriber(Lane, ABC):
    @abstractmethod
    def get_payloads(self) -> Iterable:
        pass

    @final
    def process(self, value):
        payloads = list(self.get_payloads())

        if not payloads:
            self.terminate()

        else:
            LOGGER().info(
                "Got %d payload(s).",
                len(payloads),
            )

        yield from payloads
