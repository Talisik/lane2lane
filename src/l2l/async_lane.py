import traceback
from inspect import isasyncgen, isawaitable, isgenerator
from time import perf_counter
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Iterable,
    Optional,
    Type,
    Union,
    final,
)

from ._lane_core import _LaneCore
from .errors import LaneNotFoundError
from .events import events
from .logger import logger
from .mock import Mock
from .terminate_kind import TerminateKind


async def _aiter(value):
    """Iterates a value that may be either a sync or an async iterable."""
    if isasyncgen(value):
        async for item in value:
            yield item

    else:
        for item in value:
            yield item


class AsyncLane(_LaneCore):
    """Asynchronous processing lane.

    Mirrors :class:`l2l.Lane` but runs ``process``/``run``/``start`` as
    coroutines, awaiting each value sequentially. Keeps its own lane registry
    (rooted at ``AsyncLane``) so async and sync lanes never mix inside a single
    chain. Shared machinery lives in :class:`l2l._lane_core._LaneCore`.

    Inputs may be plain values, sync generators, or async generators. The
    ``process`` method may be an ``async def`` (returning a value, a sync
    generator, or an async generator) or an ``async def`` with ``yield``.
    """

    def process(
        self,
        value,
    ) -> Union[Awaitable[Any], AsyncIterator[Any]]:
        """Processes a value through this lane's core logic.

        Override with either an ``async def`` (returning a plain value, a sync
        generator, or an async generator) or an ``async def`` with ``yield``
        (an async generator). The default awaitable returns the input unchanged.

        Declared as a plain method returning ``Awaitable | AsyncIterator`` so
        both override shapes are type-compatible; the runtime dispatches on the
        actual kind in :meth:`__invoke_process`.
        """
        return self.__default_process(value)

    async def __default_process(self, value):
        return value

    async def __invoke_process(self, value):
        """Calls ``process`` whether it is a coroutine or async-generator.

        ``process`` may be defined as ``async def`` (returns a coroutine) or as
        an ``async def`` with ``yield`` (returns an async generator). The former
        must be awaited; the latter must not.
        """
        events.emit(
            "lane_active",
            run_id=id(self),
            name=self.first_name(),
            parent_id=id(self.primary_lane) if self.primary_lane else None,
        )
        start = perf_counter()

        try:
            result = self.process(value)

            if isawaitable(result):
                result = await result

            return result
        finally:
            # If process is an async-generator, only its creation is timed
            # here (the work runs lazily as it is iterated downstream).
            self._work_seconds += perf_counter() - start
            events.emit(
                "lane_idle",
                run_id=id(self),
                name=self.first_name(),
                work=self._work_seconds,
            )

    async def __yield_result(self, result):
        if isasyncgen(result):
            async for item in result:
                yield item

        elif isgenerator(result):
            for item in result:
                yield item

        else:
            yield result

    async def __process_batch(
        self,
        value,
        max_count: int,
    ) -> AsyncGenerator[Any, None]:
        count = 0
        result = []

        async for subvalue in _aiter(value):
            result.append(subvalue)

            count += 1

            if count < max_count:
                continue

            yield result

            result = []
            count = 0

        if result:
            yield result

    async def __process_generator(self, value):
        if self.process_mode == "all":
            data: Any = [item async for item in _aiter(value)]
            result = await self.__invoke_process(data)

            if self.terminated != TerminateKind.NO:
                return

            async for item in self.__yield_result(result):
                yield item

        elif self.process_mode == "one":
            async for subvalue in _aiter(value):
                result = await self.__invoke_process(subvalue)

                if self.terminated != TerminateKind.NO:
                    return

                async for item in self.__yield_result(result):
                    yield item

                if self.terminated != TerminateKind.NO:
                    return

        else:
            async for batch in self.__process_batch(
                value,
                self.process_mode,
            ):
                result = await self.__invoke_process(batch)

                if self.terminated != TerminateKind.NO:
                    return

                async for item in self.__yield_result(result):
                    yield item

                if self.terminated != TerminateKind.NO:
                    return

    async def __process(
        self,
        value,
        processes: Optional[int],
    ):
        self._start_time = perf_counter()

        logger.debug(
            "N-{0} {1} started.",
            self._run_index,
            self.first_name(),
        )

        events.emit(
            "lane_started",
            run_id=id(self),
            name=self.first_name(),
            parent_id=id(self.primary_lane) if self.primary_lane else None,
        )

        try:
            if isgenerator(value) or isasyncgen(value):
                async for item in self.__process_generator(value):
                    yield item

            else:
                result = await self.__invoke_process(value)

                async for item in self.__yield_result(result):
                    yield item

        except Exception as e:
            self._add_error(
                e,
                traceback.format_exc(),
            )

            logger.exception(e)

            if self.terminate_on_error():
                self.terminate()

        logger.debug(
            "N-{0} {1} done in {2:.2f}s.",
            self._run_index,
            self.first_name(),
            self._work_seconds,
        )

        events.emit(
            "lane_done",
            run_id=id(self),
            name=self.first_name(),
            duration=self.duration,
            work=self._work_seconds,
            terminated=self.terminated != TerminateKind.NO,
        )

    @final
    async def goto(
        self,
        lane: Union[str, Type["AsyncLane"]],
        value: Any,
    ):
        cls = self._get_lane_ref(lane)

        if not cls:
            raise LaneNotFoundError(lane)

        result = await cls().run(value)

        if isasyncgen(result):
            async for item in result:
                yield item

        else:
            yield result

    async def __process_sub_lanes(
        self,
        value,
        sub_lanes: Iterable[Union["Mock", Type["AsyncLane"]]],
        processes: Optional[int],
    ):
        new_value = value

        for sub_lane in sub_lanes:
            if self.terminated != TerminateKind.NO:
                break

            instance = (
                AsyncLane.from_mock(sub_lane)
                if isinstance(sub_lane, Mock)
                else sub_lane(self.primary_lane or self)
            )
            original_value = new_value

            if instance.isolated:
                deconstructed_value = [item async for item in _aiter(new_value)]
                original_value = (value for value in deconstructed_value)
                new_value = (value for value in deconstructed_value)

            result = await instance.run(
                value=new_value,
                processes=processes,
            )

            if instance.isolated:
                new_value = original_value

                if isasyncgen(result):
                    async for _ in result:
                        pass

                continue

            new_value = result

        return new_value

    @final
    async def run(
        self,
        value: Any = None,
        processes: Optional[int] = None,
    ):
        """Executes this lane with the given input value.

        Async counterpart of :meth:`l2l.Lane.run`. Returns either a plain value
        or an async generator that yields the processed values.
        """
        self.__class__._run_count += 1
        self._run_index = self.__class__._run_count

        value = await self.__process_sub_lanes(
            value=value,
            sub_lanes=self.get_before_lanes(self),
            processes=processes,
        )

        if self.terminated != TerminateKind.NO:
            return value

        value = self.__process(
            value=value,
            processes=processes,
        )

        if self.terminated != TerminateKind.NO:
            return value

        return await self.__process_sub_lanes(
            value=value,
            sub_lanes=self.get_after_lanes(self),
            processes=processes,
        )

    @classmethod
    @final
    async def start(
        cls,
        name: str,
        print_lanes=True,
        print_indent=2,
        processes: Optional[int] = None,
    ):
        """Starts all primary async lanes matching ``name`` and yields results.

        Async counterpart of :meth:`l2l.Lane.start`. This is an async generator;
        iterate it with ``async for`` (or drain it inside ``asyncio.run``).
        """
        cls._reset_global_errors()

        lanes = [*cls.get_primary_lanes(name)]
        active_lanes = filter(
            lambda lane: not lane.passive(),
            lanes,
        )

        if not any(active_lanes):
            raise ValueError(f"No lanes found for '{name}'!")

        if print_lanes:
            cls.print_available_lanes(
                name,
                print_indent,
            )

            cls._print_load_order(
                lanes,
            )

        for lane in lanes:
            result = await lane.run(
                value=None,
                processes=processes,
            )

            if isasyncgen(result):
                async for item in result:
                    yield item

                if lane.terminated == TerminateKind.ALL:
                    break

                continue

            yield result

            if lane.terminated == TerminateKind.ALL:
                break


AsyncLane._registry = AsyncLane
