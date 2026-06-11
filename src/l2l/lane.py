import traceback
from collections import deque
from inspect import isgenerator
from multiprocessing.pool import ThreadPool
from time import perf_counter
from typing import (
    Any,
    Generator,
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


class Lane(_LaneCore):
    """Synchronous processing lane.

    Override :meth:`process` to implement per-value logic, then run a lane (or a
    chain of before/after lanes) with :meth:`run` / :meth:`start`. Shared
    machinery (naming, discovery, termination, errors, printing) lives in
    :class:`l2l._lane_core._LaneCore`.

    See :class:`l2l.AsyncLane` for the asynchronous counterpart.
    """

    def process(self, value) -> Any:
        """Processes a value through this lane's core logic.

        Override this method to implement specific processing. The default
        returns the input unchanged. May return a value or a generator (whose
        yielded values are collected by :meth:`run`).
        """
        return value

    def __timed_process(self, value):
        """Calls ``process`` while accumulating its own compute time.

        Note: if ``process`` returns a generator, only its creation is timed
        (the work runs lazily as the result is iterated downstream).
        """
        start = perf_counter()

        try:
            return self.process(value)
        finally:
            self._work_seconds += perf_counter() - start

    def __process_batch(
        self,
        value,
        max_count: int,
    ) -> Generator[Any, None, None]:
        count = 0
        result = []

        for subvalue in value:
            result.append(subvalue)

            count += 1

            if count < max_count:
                continue

            yield result

            result = []
            count = 0

        if result:
            yield result

    def __process_generator(self, value):
        if self.process_mode == "all":
            data: Any = [*value]
            result = self.__timed_process(data)

            if self.terminated != TerminateKind.NO:
                return

            if isgenerator(result):
                yield from result

            else:
                yield result

        elif self.process_mode == "one":
            for subvalue in value:
                result = self.__timed_process(subvalue)

                if self.terminated != TerminateKind.NO:
                    return

                if isgenerator(result):
                    yield from result

                else:
                    yield result

                if self.terminated != TerminateKind.NO:
                    return

        else:
            for result in self.__process_batch(
                value,
                self.process_mode,
            ):
                result = self.__timed_process(result)

                if self.terminated != TerminateKind.NO:
                    return

                if isgenerator(result):
                    yield from result

                else:
                    yield result

                if self.terminated != TerminateKind.NO:
                    return

    def __process(
        self,
        value,
        processes: Optional[int],
    ):
        self._start_time = perf_counter()

        logger.debug(
            "N-{0} {1} started.",
            self.__class__._run_count,
            self.first_name(),
        )

        events.emit(
            "lane_started",
            run_id=id(self),
            name=self.first_name(),
            parent_id=id(self.primary_lane) if self.primary_lane else None,
        )

        try:
            if isgenerator(value):
                if processes is None or not self.multiprocessing:
                    yield from self.__process_generator(value)

                else:
                    with ThreadPool(processes=processes) as pool:
                        for result in pool.map(
                            self.__process_generator,
                            value,
                        ):
                            if isgenerator(result):
                                yield from result

                            else:
                                yield result

            else:
                result = self.__timed_process(value)

                if isgenerator(result):
                    yield from result

                else:
                    yield result

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
            self.__class__._run_count,
            self.first_name(),
            self.duration,
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
    def goto(
        self,
        lane: Union[str, Type["Lane"]],
        value: Any,
    ):
        cls = self._get_lane_ref(lane)

        if not cls:
            raise LaneNotFoundError(lane)

        result = cls().run(value)

        if isgenerator(result):
            yield from result

        else:
            return result

    def __process_sub_lanes(
        self,
        value,
        sub_lanes: Iterable[Union["Mock", Type["Lane"]]],
        processes: Optional[int],
    ):
        new_value = value

        for sub_lane in sub_lanes:
            if self.terminated != TerminateKind.NO:
                break

            instance = (
                Lane.from_mock(sub_lane)
                if isinstance(sub_lane, Mock)
                else sub_lane(self.primary_lane or self)
            )
            original_value = new_value

            if instance.isolated:
                deconstructed_value = [*new_value]
                original_value = (value for value in deconstructed_value)
                new_value = (value for value in deconstructed_value)

            result = instance.run(
                value=new_value,
                processes=processes,
            )

            if instance.isolated:
                new_value = original_value

                if isgenerator(result):
                    deque(result, maxlen=0)

                continue

            new_value = result

        return new_value

    @final
    def run(
        self,
        value: Any = None,
        processes: Optional[int] = None,
    ):
        """Executes this lane with the given input value.

        Runs all 'before' lanes, this lane's :meth:`process`, then all 'after'
        lanes, in priority order, stopping early if terminated. Returns the
        final value, which may be a generator if processing yields.
        """
        self.__class__._run_count += 1

        value = self.__process_sub_lanes(
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

        return self.__process_sub_lanes(
            value=value,
            sub_lanes=self.get_after_lanes(self),
            processes=processes,
        )

    @classmethod
    @final
    def start(
        cls,
        name: str,
        print_lanes=True,
        print_indent=2,
        processes: Optional[int] = None,
    ):
        """Starts all primary lanes matching ``name`` and yields their results.

        Clears global errors, finds matching primary lanes, optionally prints
        the available lanes and load order, then runs each and yields results.
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
            result = lane.run(
                value=None,
                processes=processes,
            )

            if isgenerator(result):
                yield from result

                if lane.terminated == TerminateKind.ALL:
                    break

                continue

            yield result

            if lane.terminated == TerminateKind.ALL:
                break


Lane._registry = Lane
