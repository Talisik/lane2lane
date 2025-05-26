from abc import ABC, abstractmethod
from typing import Iterable, TypeVar, final

from loguru import logger

from .lane import Lane

T = TypeVar("T")


class Subscriber(Lane[T], ABC):
    @abstractmethod
    def get_payloads(self, value) -> Iterable:
        pass

    @final
    def process(self, value):
        payloads = list(self.get_payloads(value))

        if not payloads:
            self.terminate()

        else:
            logger.info(
                "Got {0} payload(s).",
                len(payloads),
            )

        yield from payloads
