import re
import traceback
from abc import ABC
from inspect import isgenerator
from multiprocessing.pool import ThreadPool
from time import perf_counter
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    final,
)

from fun_things import categorizer, get_all_descendant_classes, load_modules
from simple_chalk import chalk

from .constants import LOGGER

T = TypeVar("T")


class Lane(Generic[T], ABC):
    process_all: bool = False
    multiprocessing: bool = True
    """
    Determines how multiprocessing is handled for this lane.

    - "no": No multiprocessing is used.
    - "ordered": Multiprocessing is used with ordered results.
    - "unordered": Multiprocessing is used with unordered results.
    """

    lanes: Dict[int, Union[Type["Lane"], str, None]] = {}
    """
    A dictionary of lane classes, indexed by their priority number.

    This dictionary is used to store all lane classes that inherit from the `Lane` class.
    The keys are the priority numbers, and the values are the lane class instances or
    class names.
    Example:
        ```python
        class MyLane(Lane):
            # Define lanes to run before and after this lane
            lanes = {
                -10: "OtherLane",  # Run OtherLane before this lane (higher negative priority runs first) - string reference
                -5: AnotherLane,   # Run AnotherLane after OtherLane but before this lane
                0: SomeLane,       # Run SomeLane after this lane
                10: PostLane,      # Run PostLane after SomeLane (higher positive priority runs first)
                20: None,          # Use None to remove a lane at this priority
            }
        ```
    
    The priority numbers determine the execution order:
    - Negative priorities: Lanes that run before this lane (more negative runs first)
    - Positive priorities: Lanes that run after this lane (higher positive runs first)

    """

    __run_count: int = 0
    __global_errors: List[Exception] = []
    __global_errors_stacktrace: List[str] = []

    def __init__(
        self,
        primary_lane: Optional["Lane"] = None,
    ):
        self.__primary_lane = primary_lane
        self.__errors: List[Exception] = []
        self.__errors_stacktrace: List[str] = []
        self.__terminated = False
        self.__start_time = perf_counter()

        LOGGER().debug(
            "N-%d %s initialized.",
            self.__class__.__run_count,
            self.first_name(),
        )

        self.init()

    def terminate(self):
        """Terminates the current lane execution.

        Sets the terminated flag to True, which will stop the execution flow of the current lane
        and potentially the entire lane chain depending on implementation. Once terminated,
        further processing in this lane instance will be skipped.

        The termination is logged via the logger configured for the application.

        See Also:
            terminated: Property that checks if the lane has been terminated.
            terminate_on_error: Class method that determines termination behavior on errors.
        """
        self.__terminated = True

        LOGGER().debug(
            "N-%d %s terminated.",
            self.__class__.__run_count,
            self.first_name(),
        )

    @property
    @final
    def start_time(self):
        return self.__start_time

    @property
    @final
    def duration(self):
        return perf_counter() - self.__start_time

    @classmethod
    @final
    def terminate_on_error(cls):
        return True

    @staticmethod
    @final
    def global_errors():
        yield from Lane.__global_errors

    @staticmethod
    @final
    def global_errors_str():
        return (str(error) for error in Lane.__global_errors)

    @staticmethod
    @final
    def global_errors_stacktrace():
        yield from Lane.__global_errors_stacktrace

    @staticmethod
    @final
    def global_errors_count():
        return len(Lane.__global_errors)

    @property
    @final
    def terminated(self):
        return self.__terminated

    @classmethod
    def get_lanes(cls):
        """Retrieves all lanes associated with this lane class and its parent classes.

        This method collects all lanes from the current class's `lanes` dictionary and
        merges them with lanes from all parent Lane classes. This provides a complete
        view of all lanes in the inheritance hierarchy.

        Returns:
            dict: A dictionary mapping priority numbers to lane classes, combining
                  lanes from this class and all parent Lane classes.

        Example:
            ```python
            # Get all lanes associated with MyLane and its parent classes
            all_lanes = MyLane.get_lanes()
            ```

        See Also:
            get_before_lanes: Method that filters to only lanes with negative priority.
            get_after_lanes: Method that filters to only lanes with non-negative priority.
        """
        lanes = {**cls.lanes}

        for base in cls.__mro__[1:]:
            if issubclass(base, Lane):
                lanes.update(base.lanes)

        return lanes

    @classmethod
    def get_before_lanes(cls):
        """Retrieves all lanes that should execute before the current lane.

        Lanes are ordered based on their priority number, with lower (more negative)
        numbers running first. Only lanes with negative priority values are considered
        'before' lanes.

        The lanes are retrieved from the class's `lanes` dictionary and from all parent classes,
        then filtered and sorted according to priority.

        Yields:
            Lane: Lane class instances with negative priority numbers, sorted in ascending
                  priority order (more negative priorities execute first).

        See Also:
            get_after_lanes: Complementary method for retrieving lanes to run after.
            priority_number: Method that defines the execution priority.
        """
        for _, lane in sorted(
            filter(
                lambda v: v[0] < 0,
                cls.get_lanes().items(),
            ),
            key=lambda v: v[0],
        ):
            lane = cls.__get_lane(lane)

            if lane is not None:
                yield lane

    @classmethod
    def get_after_lanes(cls):
        """Retrieves all lanes that should execute after the current lane.

        Lanes are ordered based on their priority number, with higher (more positive)
        numbers running first. Only lanes with non-negative priority values are considered
        'after' lanes.

        The lanes are retrieved from the class's `lanes` dictionary and from all parent classes,
        then filtered and sorted according to priority.

        Yields:
            Lane: Lane class instances with non-negative priority numbers, sorted in descending
                  priority order (higher priorities execute first).

        See Also:
            get_before_lanes: Complementary method for retrieving lanes to run before.
            priority_number: Method that defines the execution priority.
        """
        for _, lane in sorted(
            filter(
                lambda v: v[0] >= 0,
                cls.get_lanes().items(),
            ),
            key=lambda v: v[0],
        ):
            lane = cls.__get_lane(lane)

            if lane is not None:
                yield lane

    @staticmethod
    def __get_lane(value: Union[Type["Lane"], str, None]):
        if value is None:
            return None

        if isinstance(value, str):
            return Lane.get_lane(value)

        return value

    @property
    @final
    def primary_lane(self):
        """Returns the primary lane associated with this lane instance.

        For primary lanes, this will return None as they don't have a parent primary lane.
        For non-primary lanes (those executed as part of a lane chain), this returns
        a reference to the primary lane that initiated the execution chain.

        Returns:
            Lane or None: The primary lane that initiated this lane's execution,
                         or None if this is itself a primary lane.

        See Also:
            primary: Class method that determines if a lane is a primary entry point.
        """
        return self.__primary_lane

    @property
    @final
    def errors_count(self):
        return len((self.primary_lane or self).__errors)

    @property
    @final
    def errors(self):
        yield from (self.primary_lane or self).__errors

    @property
    @final
    def errors_str(self):
        return (str(error) for error in (self.primary_lane or self).__errors)

    @property
    @final
    def errors_stacktrace(self):
        yield from (self.primary_lane or self).__errors_stacktrace

    @final
    def __add_error(self, error: Exception, stacktrace: str):
        (self.primary_lane or self).__errors.append(error)
        (self.primary_lane or self).__errors_stacktrace.append(stacktrace)
        self.__global_errors.append(error)
        self.__global_errors_stacktrace.append(stacktrace)

    @classmethod
    @final
    def get_run_count(cls):
        """Returns the number of times this lane has been executed.

        This method returns the total number of times the lane has been run across
        all primary and non-primary lanes. It includes the current execution if it's
        part of a lane chain.

        Returns:
            int: The total number of times the lane has been executed.

        Example:
            ```python
            class MyLane(Lane):
                pass

            # Run the lane
            my_lane = MyLane()
            my_lane.run()

            # Check the run count
            print(f"MyLane has been run {MyLane.get_run_count()} times")
            ```

        See Also:
            run_count: Property that returns the number of times this lane has been executed.
        """
        return cls.__run_count

    @classmethod
    def primary(cls) -> bool:
        """
        Determines if this lane is a primary entry point.

        Primary lanes are designated entry points that can be directly executed through
        the `start` or `start_all` class methods. Non-primary lanes can only be executed
        as part of a lane chain (before/after) initiated by a primary lane.

        By default, all lanes are non-primary. Subclasses should override this method
        to return True if they represent an executable entry point.

        Returns:
            bool: False by default. Override to return True if this lane should be
                 directly executable as a primary lane.

        Example:
            ```python
            class MyPrimaryLane(Lane):
                @classmethod
                def primary(cls) -> bool:
                    return True
            ```

        See Also:
            start: Method that launches primary lanes.
            condition: Method that further filters which primary lanes execute.
        """
        return False

    @classmethod
    def hidden(cls) -> bool:
        """
        Determines if this lane should be hidden from listings and documentation.

        Hidden lanes will not appear in the output of `print_available_lanes` and similar
        listing methods, which can help reduce noise when there are many utility lanes
        that shouldn't be directly accessed by users.

        By default, all lanes are visible. Subclasses should override this method to
        return True if they should be hidden from lane listings.

        Returns:
            bool: False by default. Override to return True to hide this lane from listings.

        Example:
            ```python
            class UtilityLane(Lane):
                @classmethod
                def hidden(cls) -> bool:
                    return True
            ```

        See Also:
            print_available_lanes: Method that lists available lanes.
        """
        return False

    @classmethod
    def max_run_count(cls) -> int:
        """Defines the maximum number of times this lane can be executed.

        This limit applies across the entire application lifetime, not just within a single
        primary lane execution. Once a lane has run up to its max count, it will be
        excluded from further execution.

        A value of 0 (the default) means the lane can be run an unlimited number of times.

        Returns:
            int: Maximum number of times this lane can run. 0 means unlimited runs.

        Example:
            ```python
            class OneTimeLane(Lane):
                @classmethod
                def max_run_count(cls) -> int:
                    return 1  # Run this lane only once
            ```

        See Also:
            get_run_count: Method that returns how many times a lane has been run.
            available_lanes: Method that filters out lanes that have reached their max runs.
        """
        return 0

    @classmethod
    def name(cls) -> Iterable[str]:
        """Provides one or more names that identify this lane.

        By default, the name is derived from the class name, converting from CamelCase
        to SNAKE_CASE. Subclasses can override this to provide custom names or aliases.

        The names are used by the `condition` method to determine if a primary lane
        should execute for a given request name.

        This method is designed as a generator to potentially yield multiple names/aliases
        for a single lane.

        Yields:
            str: The lane name(s), by default the class name converted to SNAKE_CASE.

        Example:
            ```python
            class MyCustomLane(Lane):
                @classmethod
                def name(cls) -> Iterable[str]:
                    yield "MY_CUSTOM_TRAIL"
                    yield "CUSTOM"  # An alias
            ```

        See Also:
            first_name: Method that returns the first name from this generator.
            condition: Method that uses these names to determine execution conditions.
        """
        yield re.sub(
            r"(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z0-9])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)",
            "_",
            cls.__name__,
        ).upper()

    @classmethod
    @final
    def first_name(cls) -> str:  # type: ignore
        """Returns the first name from the lane's name generator.

        This is a convenience method that returns the first name yielded by the
        `name` method. It's useful when you need just a single identifier for the lane
        rather than iterating through all possible names/aliases.

        Returns:
            str: The first name yielded by the `name` method.

        Example:
            ```python
            class MyLane(Lane):
                pass

            # Instead of:
            first = next(iter(MyLane.name()))

            # You can use:
            first = MyLane.first_name()
            ```

        See Also:
            name: The method that generates all names/aliases for this lane.
        """
        for name in cls.name():
            return name

    @classmethod
    def priority_number(cls) -> float:
        """Returns the priority number for this lane.

        The priority number is used to determine the order in which primary lanes
        are executed when multiple lanes match the same condition. Lanes with higher
        priority numbers are executed first.

        This method is only relevant for primary lanes (will only be used if primary is True).
        The default implementation returns 0, but subclasses can override this to provide
        custom priority values.

        Returns:
            float: The priority number for this lane. Higher values indicate higher priority.

        Example:
            ```python
            class HighPriorityLane(Lane):
                @classmethod
                def priority_number(cls) -> float:
                    return 100  # This lane will run before lanes with lower priorities
            ```
        """
        return 0

    @classmethod
    def condition(cls, name: str):
        """Determines if this lane should execute for the given requested name.

        For primary lanes, this method checks if the requested name matches any of the
        names provided by the lane's `name` method. For non-primary lanes, the default
        implementation always returns True, as they will execute whenever their parent
        lane runs.

        Subclasses can override this to implement more complex conditional logic for
        determining whether a lane should run based on the requested name or other
        contextual factors.

        Args:
            name: The name of the lane being requested for execution.

        Returns:
            bool: True if the lane should run, False otherwise.
                  - For primary lanes: True if name matches one of this lane's names
                  - For non-primary lanes: Always True by default

        Example:
            ```python
            class ConditionalLane(Lane):
                @classmethod
                def condition(cls, name: str):
                    # Run this lane only for names starting with "TEST_"
                    return name.startswith("TEST_")
            ```

        See Also:
            name: Method that provides the names to check against.
            primary: Method that determines if a lane is primary.
        """
        if cls.primary():
            return name in cls.name()

        return True

    def init(self):
        """Performs custom initialization after the lane instance is created.

        This method is called automatically at the end of the lane's `__init__` method
        and provides a hook for subclasses to perform initialization logic without
        having to override `__init__`.

        The default implementation does nothing. Subclasses should override this method
        to implement custom initialization behavior.

        Example:
            ```python
            class DatabaseLane(Lane):
                def init(self):
                    self.db_connection = database.connect()
                    self.session = self.db_connection.create_session()
            ```

        Note:
            This method should not take any arguments. For custom initialization that
            requires arguments, override the `__init__` method instead.
        """
        pass

    @staticmethod
    @final
    def __lane_predicate(lane: Type["Lane"]):
        max_run_count = lane.max_run_count()

        if max_run_count <= 0:
            return True

        return lane.__run_count < max_run_count

    @staticmethod
    @final
    def get_lane(classname: str):
        """Retrieves a lane instance by its class name.

        Searches through all descendant lane classes to find one matching the specified
        class name. This is useful for getting a specific lane by name for inspection
        or when you need to reference a lane dynamically.

        Args:
            classname: The exact class name of the lane to retrieve (case-sensitive).

        Returns:
            Lane: The first lane with a matching class name, or None if not found.

        Example:
            ```python
            # Get a lane by its class name
            cleanup_lane = Lane.get_lane("CleanupLane")
            if cleanup_lane:
                cleanup_lane.run()
            ```

        Note:
            This method returns an actual class, not an instance. To use the lane,
            you'll need to instantiate it first.

        See Also:
            all_lanes: Method that returns all descendant lane classes.
        """
        for lane in Lane.all_lanes():
            if lane.__name__ == classname:
                return lane

    @staticmethod
    @final
    def all_lanes():
        """Returns all descendant lane classes in the application.

        This method uses the `get_all_descendant_classes` function to find all classes
        that inherit from this Lane class, either directly or indirectly. It excludes
        ABC (Abstract Base Class) to ensure only concrete lane implementations are returned.

        This is the foundational method for discovering lane classes in the application.

        Returns:
            list: A list of all Lane descendant classes (excluding ABC).

        Example:
            ```python
            # Print the names of all available lane classes
            for lane_class in Lane.all_lanes():
                print(lane_class.__name__)
            ```

        See Also:
            available_lanes: Method that filters to only lanes that haven't reached their max run count.
            get_all_descendant_classes: External function that performs the class discovery.
        """
        return get_all_descendant_classes(
            Lane,
            exclude=[ABC],
        )

    @staticmethod
    @final
    def available_lanes():
        """Returns all available lane classes that can be run.

        This method filters the lanes returned by `all_lanes()` to include only those
        that haven't exceeded their maximum run count, and sorts them by priority number
        in descending order (highest priority first).

        Returns:
            list: Available lane classes sorted by priority number in descending order.

        Example:
            ```python
            # Get all lanes that are still available to run, sorted by priority
            for lane_class in Lane.available_lanes():
                print(f"{lane_class.__name__}: {lane_class.priority_number()}")
            ```

        See Also:
            all_lanes: Method that returns all descendant lane classes.
            max_run_count: Method that defines the maximum number of runs for a lane.
            priority_number: Method that defines the sorting order.
        """
        return sorted(
            filter(Lane.__lane_predicate, Lane.all_lanes()),
            key=lambda descendant: descendant.priority_number(),
            reverse=True,
        )

    @staticmethod
    @final
    def get_primary_lane(name: str):
        """Returns the first primary lane that matches the specified name.

        This is a convenience method that calls `get_primary_lanes` and returns
        just the first matching lane, or None if no lanes match.

        Args:
            name: The name to check against lane conditions.

        Returns:
            Lane: The first matching primary lane instance, or None if none found.

        Example:
            ```python
            # Get the first primary lane that matches "PROCESS_DATA"
            lane = Lane.get_primary_lane("PROCESS_DATA")
            if lane:
                result = lane.run(data)
            ```

        See Also:
            get_primary_lanes: Method that returns all matching primary lanes.
            primary: Method that determines if a lane is primary.
            condition: Method that checks if a lane should run for a name.
        """
        for lane in Lane.get_primary_lanes(name):
            return lane

    @staticmethod
    @final
    def get_primary_lanes(name: str):
        """Returns all primary lanes that match the specified name.

        This method:
        1. Gets all available lane classes
        2. Filters to include only primary lanes
        3. Further filters to only those that satisfy the condition for the given name
        4. Instantiates and yields each matching lane

        Args:
            name: The name to check against lane conditions.

        Yields:
            Lane: Instantiated primary lanes that match the condition for the given name.

        Example:
            ```python
            # Run all primary lanes that match "PROCESS_DATA"
            for lane in Lane.get_primary_lanes("PROCESS_DATA"):
                result = lane.run(data)
                process_result(result)
            ```

        See Also:
            get_primary_lane: Method that returns just the first matching lane.
            available_lanes: Method that returns all available lane classes.
            primary: Method that determines if a lane is primary.
            condition: Method that checks if a lane should run for a name.
        """
        descendants = Lane.available_lanes()

        for descendant in descendants:
            if not descendant.primary():
                continue

            ok = descendant.condition(name)

            if not ok:
                continue

            yield descendant()

    def process(self, value: T) -> Any:
        """Processes a value through this lane's core logic.

        This is the main method that subclasses should override to implement their
        specific processing logic. The default implementation simply returns the
        input value unchanged.

        When this lane is run, this method will be called for each value in the input
        stream (if the input is a generator) or once for the input value (if it's not
        a generator).

        Args:
            value: The input value to process.

        Returns:
            Any: The processed value. By default, returns the input value unchanged.

        Example:
            ```python
            class TransformLane(Lane):
                def process(self, value):
                    # Transform the input value in some way
                    return transform_function(value)
            ```

        Note:
            This method can return a generator, in which case the yield values will be
            collected and yielded by the `run` method.

        See Also:
            run: Method that handles the full execution lifecycle including this method.
        """
        return value

    def __process(
        self,
        value,
        processes: Optional[int],
    ):
        self.__start_time = perf_counter()

        LOGGER().debug(
            "N-%d %s started.",
            self.__class__.__run_count,
            self.first_name(),
        )

        try:
            if isgenerator(value):
                if processes is None or not self.multiprocessing:
                    if self.process_all:
                        data: Any = [*value]
                        result = self.process(data)

                        if isgenerator(result):
                            yield from result

                        else:
                            yield result

                    else:
                        for subvalue in value:
                            result = self.process(subvalue)

                            if isgenerator(result):
                                yield from result

                            else:
                                yield result

                else:
                    with ThreadPool(processes=processes) as pool:
                        for result in pool.map(
                            self.process,
                            value,
                        ):
                            if isgenerator(result):
                                yield from result

                            else:
                                yield result

            else:
                result = self.process(value)

                if isgenerator(result):
                    yield from result

                else:
                    yield result

        except Exception as e:
            self.__add_error(
                e,
                traceback.format_exc(),
            )

            if self.terminate_on_error():
                self.terminate()

            traceback.print_exc()

        LOGGER().debug(
            "N-%d %s done in %ss.",
            self.__class__.__run_count,
            self.first_name(),
            f"{self.duration:.2f}",
        )

    @final
    def run(
        self,
        value: Any = None,
        processes: Optional[int] = None,
    ):
        """Executes this lane with the given input value.

        This is the main entry point for running a lane. The method:
        1. Increments the run count for this lane class
        2. Executes all 'before' lanes in priority order
        3. Processes the value through this lane's `process` method
        4. Executes all 'after' lanes in priority order

        Throughout execution, it checks for termination and will stop processing
        if the lane has been terminated.

        Args:
            value: The input value to process. Defaults to None.

        Returns:
            Any: The final processed value after running through this lane and all
                 sub-lanes in the chain.

        Example:
            ```python
            # Create and run a lane with a specific input value
            result = MyLane().run(input_data)

            # Run with no input (will use None as the input value)
            result = MyLane().run()
            ```

        Note:
            If the lane's processing logic (or any sub-lane) returns a generator,
            this method will collect the yielded values.

        See Also:
            process: The method that implements the core processing logic.
            get_before_lanes: Method that retrieves 'before' lanes.
            get_after_lanes: Method that retrieves 'after' lanes.
            terminate: Method that can be called to stop processing.
        """
        self.__class__.__run_count += 1

        for sub_lane in self.get_before_lanes():
            new_value = sub_lane().run(
                value=value,
                processes=processes,
            )

            if self.terminated:
                return value

            value = new_value

        new_value = self.__process(
            value=value,
            processes=processes,
        )

        if self.terminated:
            return value

        value = new_value

        for sub_lane in self.get_after_lanes():
            new_value = sub_lane().run(
                value=value,
                processes=processes,
            )

            if self.terminated:
                return value

            value = new_value

        return value

    @staticmethod
    @final
    def start_all(
        name: str,
        print_lanes=True,
        print_indent=2,
    ):
        """Starts all primary lanes matching the given name and returns results as a list.

        This is a convenience wrapper around the `start` method that collects all results
        into a list rather than yielding them as an iterator.

        Args:
            name: The name to match against lane conditions.
            print_lanes: Whether to print available lanes and the execution order.
                          Defaults to True.
            print_indent: Indentation level for printing lane information.
                          Defaults to 2.

        Returns:
            list: A list containing all results from matching primary lanes.

        Example:
            ```python
            # Run all lanes matching "PROCESS_DATA" and collect results in a list
            results = Lane.start_all("PROCESS_DATA")

            # Run quietly (no lane listings printed)
            results = Lane.start_all("PROCESS_DATA", print_lanes=False)
            ```

        See Also:
            start: The underlying method that actually starts the lanes.
            get_primary_lanes: Method that finds lanes to execute.
        """
        return [
            *Lane.start(
                name,
                print_lanes,
                print_indent,
            )
        ]

    @staticmethod
    @final
    def start(
        name: str,
        print_lanes=True,
        print_indent=2,
        processes: Optional[int] = None,
    ):
        """Starts all primary lanes matching the given name and yields their results.

        This is the main entry point for executing lanes by name. The method:
        1. Clears global errors
        2. Finds all primary lanes matching the name
        3. Optionally prints available lanes and execution order
        4. Runs each matching lane and yields its results

        Args:
            name: The name to match against lane conditions.
            print_lanes: Whether to print available lanes and the execution order.
                          Defaults to True.
            print_indent: Indentation level for printing lane information.
                          Defaults to 2.

        Yields:
            Any: Results from each primary lane that matches the name.

        Example:
            ```python
            # Run all lanes matching "PROCESS_DATA" and process results as they come
            for result in Lane.start("PROCESS_DATA"):
                process_result(result)
            ```

        See Also:
            start_all: Method that collects all results into a list.
            get_primary_lanes: Method that finds lanes to execute.
            print_available_lanes: Method that shows available lanes.
        """
        Lane.__global_errors = []
        Lane.__global_errors_stacktrace = []

        lanes = [*Lane.get_primary_lanes(name)]
        visible_lanes = filter(
            lambda lane: not lane.hidden(),
            lanes,
        )

        if not any(visible_lanes):
            raise ValueError(f"No lanes found for '{name}'!")

        if print_lanes:
            Lane.print_available_lanes(
                name,
                print_indent,
            )

            Lane.__print_load_order(
                lanes,
            )

        if processes is None:
            for lane in lanes:
                result = lane.run(
                    value=None,
                    processes=processes,
                )

                if isgenerator(result):
                    yield from result
                    continue

                yield result

            return

        for lane in lanes:
            result = lane.run(
                value=None,
                processes=processes,
            )

            if isgenerator(result):
                yield from result
                continue

            yield result

    @staticmethod
    @final
    def __print_load_order(
        lanes: List["Lane"],
    ):
        if not any(lanes):
            return

        print(
            f"<{chalk.yellow('Load Order')}>",
            chalk.yellow.bold("↓"),
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
            name = chalk.green.bold(name)

            print(
                f"[{chalk.yellow(priority_number)}]",
                chalk.green(name),
            )

        print()

    @staticmethod
    @final
    def __get_printed_name(
        item: Type["Lane"],
        name: Optional[str],
    ):
        text = item.first_name()

        if name is None:
            return text

        if not item.primary():
            # Not primary.
            text = chalk.dim.gray(text)

        elif not item.condition(name):
            # Primary, but condition is not met.
            text = f"{text} {chalk.bold('✕')}"

        else:
            # Primary lane.
            text = chalk.green.bold(text)
            text = f"{text} {chalk.bold('✓')}"

        return text

    @staticmethod
    @final
    def __draw_lanes(
        name: str,
        lanes,
        indent_text: str,
    ):
        lanes0: List[Type["Lane"]] = [lane[0] for lane in lanes]
        lanes0.sort(
            key=lambda lane: lane.priority_number(),
            reverse=True,
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
                f"[{chalk.yellow(priority_number)}]",
                Lane.__get_printed_name(
                    lane,
                    name,
                ),
            )

    @staticmethod
    @final
    def __draw_categories(
        name: str,
        indent_size: int,
        indent_scale: int,
        keyword: Optional[str],
        category: Any,
    ):
        if keyword is None:
            keyword = "*"

        indent_text = " " * indent_size * indent_scale

        print(f"{indent_text}{chalk.yellow(keyword)}:")

        if isinstance(category, list):
            Lane.__draw_lanes(
                name=name,
                lanes=category,
                indent_text=indent_text,
            )
            return

        for sub_category in category.items():
            yield indent_size + 1, sub_category

    @staticmethod
    @final
    def print_available_lanes(
        name: str = None,  # type: ignore
        indent: int = 2,
    ):
        """Prints a hierarchical listing of all available and visible lanes.

        This method formats and displays all non-hidden lanes, organized by their names
        and showing their priority numbers. It supports highlighting primary lanes that
        match the specified name.

        The output is color-coded:
        - Primary lanes that match the name are highlighted in green with a ✓
        - Primary lanes that don't match the name are marked with a ✕
        - Non-primary lanes are displayed in gray

        Args:
            name: Optional name to highlight primary lanes that match. When None,
                  no special highlighting is applied. Defaults to None.
            indent: Indentation level for formatting the hierarchical display.
                   Defaults to 2.

        Example:
            ```python
            # Print all available lanes with no special highlighting
            Lane.print_available_lanes()

            # Print lanes and highlight those that match "PROCESS_DATA"
            Lane.print_available_lanes("PROCESS_DATA")
            ```

        See Also:
            available_lanes: Method that returns all available lane classes.
            hidden: Method that determines if a lane should be hidden from listings.
        """
        categorized = [
            (0, pair)
            for pair in categorizer(
                [
                    (
                        lane,
                        lane.first_name(),
                    )
                    for lane in filter(
                        lambda lane: not lane.hidden(),
                        Lane.available_lanes(),
                    )
                ],
                lambda tuple: tuple[1],
            ).items()
        ]

        while len(categorized) > 0:
            indent_size, (keyword, category) = categorized.pop()

            for sub_category in Lane.__draw_categories(
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

        This method imports all modules and packages within the specified path,
        including nested subpackages. It's typically used to preload all lane
        implementations so they can be discovered by the lane system.

        The method intelligently handles both regular modules and packages with
        __init__.py files, and it skips special folders (those with names like
        __pycache__).

        Args:
            path: The import path to load from (e.g., "myproject.lanes").
                 This should be a Python import path, not a file path.

        Yields:
            module: Each imported module object.

        Example:
            ```python
            # Load all modules in the 'myproject.lanes' package
            for module in Lane.load("myproject.lanes"):
                print(f"Loaded module: {module.__name__}")
            ```

        Note:
            This method is often called during application startup to ensure all
            lane classes are loaded and registered with the lane system.
        """
        yield from load_modules(path, recursive)
