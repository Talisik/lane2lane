from abc import ABC, abstractmethod
from typing import Iterable

from .logger import logger

from .async_lane import AsyncLane


class AsyncSubscriber(AsyncLane, ABC):
    @abstractmethod
    async def get_payloads(self, value) -> Iterable:
        pass

    async def process(self, value):
        payloads = list(await self.get_payloads(value))

        if not payloads:
            self.terminate()

        else:
            logger.info(
                "Got {0} payload(s).",
                len(payloads),
            )

        for payload in payloads:
            yield payload
