from abc import abstractmethod
from typing import Iterable, final

from ..constants import LOGGER
from .trail import Trail


class Subscriber(Trail):
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
