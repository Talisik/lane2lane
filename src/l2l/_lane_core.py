import asyncio
import re
from abc import ABC
from time import perf_counter
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    final,
)

from fun_things import categorizer, get_all_descendant_classes, load_modules

from ._style import style
from .events import _AsyncGate, _SyncGate, events
from .logger import logger
from .mock import Mock
from .terminate_kind import TerminateKind
from .types import LaneDictType, ProcessModeType


class _LaneCore:
    """Shared machinery for :class:`l2l.Lane` and :class:`l2l.AsyncLane`.

    Holds everything that is identical between the synchronous and the
    asynchronous lane: naming, conditions, priorities, lane discovery,
    termination, error tracking and the diagnostic printing helpers.

    The execution core (``process``/``run``/``start`` and their internals)
    lives in the concrete subclasses, since those genuinely differ between the
    sync and async models. Each concrete class also sets ``_registry`` to itself
    so discovery (``all_lanes`` etc.) stays scoped to that lane family.

    Shared mutable state uses single-underscore names on purpose: name mangling
    is per-class, and these attributes must be visible from both this base and
    the concrete subclasses that read/write them.
    """

    #: The concrete root class used for lane discovery. Set by each subclass.
    _registry: Optional[type] = None

    isolated: bool = False
    """
    If True, the lane will not return its values for the next lane.
    """
    process_mode: ProcessModeType = "one"
    """
    Controls how input values are processed.

    If set to `"all"`, the entire input collection is processed as a single batch.

    If set to `integer`, input is processed in batches of that size.

    If set to `"one"`, each item is processed individually.
    """

    multiprocessing: bool = True
    """
    Determines whether to use multiprocessing for handling generator inputs.

    Only meaningful for the synchronous :class:`l2l.Lane` (ThreadPool). The
    asynchronous :class:`l2l.AsyncLane` processes values sequentially with
    ``await`` and ignores this flag; it is kept here so a shared ``Mock`` can
    drive both lane types.
    """
    use_filename: bool = False
    """
    Determines if the filename should be used as the lane name.

    If True, the filename will be used as the lane name.
    If False, the class name will be used as the lane name.
    """

    lanes: LaneDictType = {}
    """
    A dictionary of lane classes, indexed by their priority number.

    The keys are the priority numbers, and the values are the lane class
    instances or class names. Negative priorities run before this lane (more
    negative first); non-negative priorities run after (higher first). ``None``
    removes a lane at that priority.
    """

    _run_count: int = 0
    _global_errors: List[Exception] = []
    _global_errors_stacktrace: List[str] = []

    def __init__(
        self,
        primary_lane: Optional["_LaneCore"] = None,
    ):
        self._primary_lane = primary_lane
        self._errors: List[Exception] = []
        self._errors_stacktrace: List[str] = []
        self._terminated: TerminateKind = TerminateKind.NO
        self._start_time = perf_counter()
        #: Cumulative time spent inside this lane's own process() calls.
        #: Truthful "work" time, excluding waiting for downstream lanes to pull
        #: (the wall-clock duration would otherwise bunch up at pipeline drain).
        self._work_seconds = 0.0
        #: Seconds spent parked at a breakpoint, subtracted from work time so a
        #: manual pause never inflates the truthful compute total.
        self._paused_seconds = 0.0
        #: Stable run number for THIS instance, used in all N-x logs. Predicted
        #: at construction (the next run index) and confirmed in run(); avoids
        #: the lazily-logged class counter showing the wrong N at drain time.
        self._run_index = self.__class__._run_count + 1
        #: Immediate parent lane (the lane that ran this as a sub-lane), used for
        #: hierarchical visualization. Distinct from primary_lane (the top of the
        #: chain, used for error/termination aggregation). None for a top lane.
        self._tree_parent: Optional["_LaneCore"] = None
        #: Whether the "started" line has been logged this run (logged at the
        #: first process() call, i.e. real execution order, not lazy gen entry).
        self._started_logged = False

        logger.debug(
            "N-{0} {1} initialized.",
            self._run_index,
            self.first_name(),
        )

        self.init()

    def _log_started(self):
        """Logs/emits the 'started' line once, at the first process() call.

        Done here (not at generator entry) so it reflects real execution order
        rather than the lazy, reversed generator-entry order.
        """
        if self._started_logged:
            return

        self._started_logged = True

        logger.debug(
            "N-{0} {1} started.",
            self._run_index,
            self.first_name(),
        )

        events.emit(
            "lane_started",
            run_id=id(self),
            name=self.first_name(),
            parent_id=id(self._tree_parent) if self._tree_parent else None,
        )

    @final
    def breakpoint(self, label: Optional[str] = None):
        """Pauses this lane until a dev tool resumes it. **Dev-only, no-op otherwise.**

        Call inside :meth:`process` to halt the pipeline at this point, like
        ``pdb.set_trace()`` but driven by the dev UI rather than a debugger
        prompt. Blocks the calling thread until
        :meth:`l2l.events.resume`/``resume_all`` is invoked for this lane.

        Does nothing (and costs nothing) unless breakpoints have been armed via
        ``events.enable_breakpoints()`` — so it is inert in ``moo run``. Use
        :meth:`abreakpoint` from an :class:`l2l.AsyncLane`.

        Args:
            label: Optional note surfaced to the dev tool (e.g. a phase name).
        """
        if not events.breakpoints_enabled:
            return

        gate = _SyncGate()
        self._begin_breakpoint(gate, label)
        start = perf_counter()
        gate.wait()
        self._end_breakpoint(start)

    @final
    async def abreakpoint(self, label: Optional[str] = None):
        """Async counterpart of :meth:`breakpoint` (awaits instead of blocking).

        Call as ``await self.abreakpoint()`` inside an async ``process``. Same
        dev-only semantics; releasable from another thread (the dev tool's).
        """
        if not events.breakpoints_enabled:
            return

        gate = _AsyncGate(asyncio.get_running_loop())
        self._begin_breakpoint(gate, label)
        start = perf_counter()
        await gate.wait()
        self._end_breakpoint(start)

    def _begin_breakpoint(self, gate, label: Optional[str]):
        run_id = id(self)
        events._register_gate(run_id, gate)

        logger.debug(
            "N-{0} {1} paused at breakpoint.",
            self._run_index,
            self.first_name(),
        )

        events.emit(
            "lane_breakpoint",
            run_id=run_id,
            name=self.first_name(),
            parent_id=id(self._tree_parent) if self._tree_parent else None,
            label=label,
        )

    def _end_breakpoint(self, start: float):
        run_id = id(self)
        # Don't count the manual pause as compute time.
        self._paused_seconds += perf_counter() - start
        events._clear_gate(run_id)

        logger.debug(
            "N-{0} {1} resumed.",
            self._run_index,
            self.first_name(),
        )

        events.emit(
            "lane_resumed",
            run_id=run_id,
            name=self.first_name(),
        )

    @final
    def terminate(
        self,
        kind: TerminateKind = TerminateKind.SELF,
    ):
        """Terminates the current lane execution.

        Sets the terminated flag, which stops the execution flow of the current
        lane and, depending on ``kind``, neighbor or all lanes. Once terminated,
        further processing in this lane instance is skipped.

        See Also:
            terminated: Property that checks if the lane has been terminated.
            terminate_on_error: Class method that determines termination behavior on errors.
        """
        self._terminated = kind

        events.emit(
            "lane_terminated",
            run_id=id(self),
            name=self.first_name(),
            terminate_kind=kind.value,
        )

        if kind in [
            TerminateKind.NEIGHBOR,
            TerminateKind.ALL,
        ]:
            primary_lane = self.primary_lane

            if primary_lane is not None:
                primary_lane.terminate()

        logger.debug(
            "N-{0} {1} terminated.",
            self._run_index,
            self.first_name(),
        )

    @property
    @final
    def start_time(self):
        """`perf_counter()` timestamp when this lane's processing last started."""
        return self._start_time

    @property
    @final
    def duration(self):
        """Wall-clock seconds since processing started (live until done)."""
        return perf_counter() - self._start_time

    @classmethod
    @final
    def terminate_on_error(cls):
        """Whether an unhandled error in `process()` terminates the lane (True)."""
        return True

    @classmethod
    @final
    def _reset_global_errors(cls):
        _LaneCore._global_errors = []
        _LaneCore._global_errors_stacktrace = []

    @staticmethod
    @final
    def global_errors():
        """Yields every exception captured across all lanes since the last run."""
        yield from _LaneCore._global_errors

    @staticmethod
    @final
    def global_errors_str():
        """Yields the string form of each globally captured exception."""
        return (str(error) for error in _LaneCore._global_errors)

    @staticmethod
    @final
    def global_errors_stacktrace():
        """Yields the formatted traceback of each globally captured exception."""
        yield from _LaneCore._global_errors_stacktrace

    @staticmethod
    @final
    def global_errors_count():
        """Number of exceptions captured across all lanes since the last run."""
        return len(_LaneCore._global_errors)

    @property
    @final
    def terminated(self):
        """The chain's `TerminateKind` (reads the primary lane's flag)."""
        return (self.primary_lane or self)._terminated

    @classmethod
    def get_lanes(cls, self: Optional["_LaneCore"] = None):
        """Retrieves all lanes associated with this lane class and its parents.

        Collects lanes from this class's ``lanes`` dictionary and merges them
        with lanes from all parent lane classes.
        """
        lanes = {**cls.lanes} if self is None else {**self.lanes}

        for base in cls.__mro__[1:]:
            if issubclass(base, _LaneCore):
                lanes.update(base.lanes)

        return lanes

    @classmethod
    def _resolve_lane_reference(cls, lane):
        """Resolves a lane reference (dict/Mock/str/class/None) to a runnable."""
        if isinstance(lane, dict):
            return Mock(lanes=lane)

        if isinstance(lane, Mock):
            return lane

        if lane is None:
            return None

        if isinstance(lane, str):
            return cls.get_lane(lane)

        return lane

    @classmethod
    def _get_lane_ref(cls, value):
        """Resolves a single lane reference for ``goto`` (str/class/None)."""
        if value is None:
            return None

        if isinstance(value, str):
            return cls.get_lane(value)

        return value

    @classmethod
    def get_before_lanes(cls, self: Optional["_LaneCore"] = None):
        """Retrieves all lanes that should execute before the current lane.

        Only lanes with negative priority are 'before' lanes, sorted ascending
        (more negative priorities execute first).
        """
        for _, lane in sorted(
            filter(
                lambda v: v[0] < 0,
                cls.get_lanes(self).items(),
            ),
            key=lambda v: v[0],
        ):
            lane = cls._resolve_lane_reference(lane)

            if lane is not None:
                yield lane

    @classmethod
    def get_after_lanes(cls, self: Optional["_LaneCore"] = None):
        """Retrieves all lanes that should execute after the current lane.

        Only lanes with non-negative priority are 'after' lanes, sorted
        descending (higher priorities execute first).
        """
        for _, lane in sorted(
            filter(
                lambda v: v[0] >= 0,
                cls.get_lanes(self).items(),
            ),
            key=lambda v: v[0],
        ):
            lane = cls._resolve_lane_reference(lane)

            if lane is not None:
                yield lane

    @property
    @final
    def primary_lane(self):
        """Returns the primary lane associated with this lane instance.

        Returns ``None`` for primary lanes; otherwise the primary lane that
        initiated this lane's execution chain.
        """
        return self._primary_lane

    @property
    @final
    def errors_count(self):
        """Number of exceptions captured in this lane's chain."""
        return len((self.primary_lane or self)._errors)

    @property
    @final
    def errors(self):
        """Yields each exception captured in this lane's chain."""
        yield from (self.primary_lane or self)._errors

    @property
    @final
    def errors_str(self):
        """Yields the string form of each exception in this lane's chain."""
        return (str(error) for error in (self.primary_lane or self)._errors)

    @property
    @final
    def errors_stacktrace(self):
        """Yields the formatted traceback of each exception in this lane's chain."""
        yield from (self.primary_lane or self)._errors_stacktrace

    @final
    def _add_error(self, error: Exception, stacktrace: str):
        (self.primary_lane or self)._errors.append(error)
        (self.primary_lane or self)._errors_stacktrace.append(stacktrace)
        _LaneCore._global_errors.append(error)
        _LaneCore._global_errors_stacktrace.append(stacktrace)

    @classmethod
    @final
    def get_run_count(cls):
        """Returns the number of times this lane class has been executed."""
        return cls._run_count

    @classmethod
    def primary(cls) -> bool:
        """Determines if this lane is a primary entry point.

        Primary lanes can be directly executed through ``start``. Non-primary
        lanes only execute as part of a lane chain. False by default.
        """
        return False

    @classmethod
    def passive(cls) -> bool:
        """Determines if this lane should be hidden from listings.

        Passive lanes still execute but don't appear in ``print_available_lanes``.
        False by default.
        """
        return False

    @classmethod
    def max_run_count(cls) -> int:
        """Maximum number of times this lane can run. 0 (default) means unlimited."""
        return 0

    @classmethod
    def name(cls) -> Iterable[str]:
        """Yields one or more names that identify this lane.

        By default, the class name (or filename if ``use_filename``) converted
        from CamelCase to SNAKE_CASE.
        """
        yield re.sub(
            r"(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z0-9])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)",
            "_",
            cls.__module__.split(".")[-1] if cls.use_filename else cls.__name__,
        ).upper()

    @classmethod
    @final
    def first_name(cls) -> str:  # type: ignore
        """Returns the first name from the lane's name generator."""
        for name in cls.name():
            return name

    @classmethod
    def priority_number(cls) -> float:
        """Returns the priority number for this lane (higher runs first). 0 by default."""
        return 0

    @classmethod
    def condition(cls, name: str):
        """Determines if this lane should execute for the given requested name.

        Primary lanes run when ``name`` matches one of their names; non-primary
        lanes always return True.
        """
        if cls.primary():
            return name in cls.name()

        return True

    def init(self):
        """Hook for custom initialization, called at the end of ``__init__``.

        The default implementation does nothing. Note this hook is synchronous
        for both lane types.
        """
        pass

    @staticmethod
    @final
    def _lane_predicate(lane: Type["_LaneCore"]):
        max_run_count = lane.max_run_count()

        if max_run_count <= 0:
            return True

        return lane._run_count < max_run_count

    @classmethod
    @final
    def get_lane(cls, name: str):
        """Retrieves a lane class by its name from this lane family's registry."""
        for lane in cls.all_lanes():
            if lane.__name__ == name:
                return lane

            if name in lane.name():
                return lane

    @classmethod
    @final
    def all_lanes(cls):
        """Returns all descendant lane classes in this lane family (excluding ABC)."""
        return get_all_descendant_classes(
            cls._registry or cls,
            exclude=[ABC],
        )

    @classmethod
    @final
    def available_lanes(cls):
        """Returns available lane classes (under max run count), sorted by priority."""
        return sorted(
            filter(cls._lane_predicate, cls.all_lanes()),
            key=lambda descendant: descendant.priority_number(),
        )

    @classmethod
    @final
    def get_primary_lane(cls, name: str):
        """Returns the first primary lane that matches the specified name."""
        for lane in cls.get_primary_lanes(name):
            return lane

    @classmethod
    @final
    def get_primary_lanes(cls, name: str):
        """Yields instantiated primary lanes that match the condition for ``name``."""
        descendants = cls.available_lanes()

        for descendant in descendants:
            if not descendant.primary():
                continue

            ok = descendant.condition(name)

            if not ok:
                continue

            yield descendant()

    @classmethod
    def from_mock(cls, mock: Mock):
        """Builds a lane instance from a `Mock` (inline, anonymous sub-pipeline)."""
        lane = cls()
        lane.lanes = mock.lanes
        lane.isolated = mock.isolated
        lane.process_mode = mock.process_mode
        lane.multiprocessing = mock.multiprocessing

        return lane

    @classmethod
    @final
    def _print_load_order(
        cls,
        lanes: Sequence["_LaneCore"],
    ):
        if not lanes:
            return

        print(
            f"<{style.yellow('Load Order')}>",
            style.yellow.bold("↓"),
        )

        items = [
            (
                lane.priority_number(),
                lane.first_name(),
            )
            for lane in lanes
        ]

        has_negative = items[-1][0] < 0
        zfill = map(
            lambda item: item[0],
            items,
        )
        zfill = map(lambda number: len(str(abs(number))), zfill)
        zfill = max(zfill)

        if has_negative:
            zfill += 1

        for priority_number, name in items:
            if has_negative:
                priority_number = "%+d" % priority_number
            else:
                priority_number = str(priority_number)

            priority_number = priority_number.zfill(zfill)
            name = style.green.bold(name)

            print(
                f"[{style.yellow(priority_number)}]",
                style.green(name),
            )

        print()

    @staticmethod
    @final
    def _get_printed_name(
        item: Type["_LaneCore"],
        name: Optional[str],
    ):
        text = item.first_name()

        if name is None:
            return text

        if not item.condition(name):
            # Primary, but condition is not met.
            text = style.dim.gray(text)
            text = f"{text} {style.bold('✕')}"

        elif item.passive():
            # Passive lane.
            text = style.blue.bold(text)
            text = f"{text} {style.bold('✓')}"

        else:
            # Primary lane.
            text = style.green.bold(text)
            text = f"{text} {style.bold('✓')}"

        return text

    @classmethod
    @final
    def _draw_lanes(
        cls,
        name: str,
        lanes,
        indent_text: str,
    ):
        lanes0: List[Type["_LaneCore"]] = [lane[0] for lane in lanes]
        lanes0.sort(
            key=lambda lane: lane.priority_number(),
        )

        count = len(lanes0)
        priority_numbers = [lane.priority_number() for lane in lanes0]
        max_priority_len = map(
            lambda number: len(str(abs(number))),
            priority_numbers,
        )
        max_priority_len = max(max_priority_len)
        has_negative = map(
            lambda number: number < 0,
            priority_numbers,
        )
        has_negative = any(has_negative)

        if has_negative:
            max_priority_len += 1

        for lane in lanes0:
            count -= 1

            priority_number = lane.priority_number()

            if has_negative:
                priority_number = "%+d" % priority_number
            else:
                priority_number = str(priority_number)

            priority_number = priority_number.zfill(
                max_priority_len,
            )
            line = "├" if count > 0 else "└"

            print(
                f"{indent_text}{line}",
                f"[{style.yellow(priority_number)}]",
                cls._get_printed_name(
                    lane,
                    name,
                ),
            )

    @classmethod
    @final
    def _draw_categories(
        cls,
        name: str,
        indent_size: int,
        indent_scale: int,
        keyword: Optional[str],
        category: Any,
    ):
        if keyword is None:
            keyword = "*"

        indent_text = " " * indent_size * indent_scale

        print(f"{indent_text}{style.yellow(keyword)}:")

        if isinstance(category, list):
            cls._draw_lanes(
                name=name,
                lanes=category,
                indent_text=indent_text,
            )
            return

        for sub_category in category.items():
            yield indent_size + 1, sub_category

    @classmethod
    @final
    def print_available_lanes(
        cls,
        name: str = None,  # type: ignore
        indent: int = 2,
    ):
        """Prints a hierarchical listing of all available and visible lanes.

        Primary lanes matching ``name`` are highlighted (green ✓); non-matching
        primary lanes are dimmed (✕); passive lanes are blue.
        """
        categorized = [
            (0, pair)
            for pair in categorizer(
                [
                    (
                        lane,
                        lane.first_name(),
                    )
                    for lane in cls.available_lanes()
                    if lane.primary()
                ],
                lambda tuple: tuple[1],
            ).items()
        ]

        while len(categorized) > 0:
            indent_size, (keyword, category) = categorized.pop()

            for sub_category in cls._draw_categories(
                name=name,
                indent_size=indent_size,
                indent_scale=indent,
                keyword=keyword,
                category=category,
            ):
                categorized.append(sub_category)

    @staticmethod
    def load(path: str, recursive: bool = True):
        """Recursively loads modules and packages from the given import path.

        Used to preload lane implementations so they can be discovered. Imports
        register both sync and async lane subclasses.
        """
        yield from load_modules(path, recursive)
