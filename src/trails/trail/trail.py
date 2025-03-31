import re
import traceback
from abc import ABC
from inspect import isgenerator
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union, final

from fun_things import categorizer, get_all_descendant_classes
from simple_chalk import chalk

T = TypeVar("T")


class Trail(ABC):
    trails: Dict[int, Union[Type["Trail"], str, None]] = {}

    __run_count: int = 0

    def __init__(
        self,
        primary_trail: Optional["Trail"] = None,
    ):
        self.__primary_trail = primary_trail
        self.__errors: List[Exception] = []
        self.__errors_stacktrace: List[str] = []
        self.__terminated = False

    def terminate(self):
        """
        Terminates this trail, preventing further processing.
        """
        self.__terminated = True

    @property
    @final
    def terminated(self):
        """
        Returns whether this trail has been terminated.

        Returns:
            bool: True if terminated, False otherwise.
        """
        return self.__terminated

    @classmethod
    def __get_trails(cls):
        """
        Gets all trails from this class and its parent classes.

        Returns:
            Dict: A dictionary of trails indexed by priority number.
        """
        trails = {**cls.trails}

        for base in cls.__mro__[1:]:
            if isinstance(base, Trail):
                trails.update(base.trails)

        return trails

    @classmethod
    def get_before_trails(cls):
        """
        Gets all trails with negative priority numbers, sorted by priority.
        These trails are executed before the main trail.

        Yields:
            Trail: Trail instances with negative priority numbers.
        """
        for _, trail in sorted(
            filter(
                lambda v: v[0] < 0,
                cls.__get_trails().items(),
            ),
            key=lambda v: v[0],
        ):
            trail = cls.__get_trail(trail)

            if trail != None:
                yield trail

    @classmethod
    def get_after_trails(cls):
        """
        Gets all trails with non-negative priority numbers, sorted by priority.
        These trails are executed after the main trail.

        Yields:
            Trail: Trail instances with non-negative priority numbers.
        """
        for _, trail in sorted(
            filter(
                lambda v: v[0] >= 0,
                cls.__get_trails().items(),
            ),
            key=lambda v: v[0],
        ):
            trail = cls.__get_trail(trail)

            if trail != None:
                yield trail

    @classmethod
    def __get_trail(cls, value: Union[Type["Trail"], str, None]):
        """
        Converts a trail value to a trail instance.

        Args:
            value: A trail class, trail name, or None.

        Returns:
            Trail: The trail instance if value is not None, otherwise None.
        """
        if value == None:
            return None

        if isinstance(value, str):
            return cls.get_trail(value)

        return value

    @property
    @final
    def primary_trail(self):
        """
        Returns the primary trail this trail is associated with.
        """
        return self.__primary_trail

    @property
    @final
    def errors_count(self):
        """
        Returns the count of errors encountered in this trail or its primary trail.
        """
        return len((self.primary_trail or self).__errors)

    @property
    @final
    def errors(self):
        """
        Yields all errors encountered in this trail or its primary trail.
        """
        yield from (self.primary_trail or self).__errors

    @property
    @final
    def errors_str(self):
        """
        Returns string representations of all errors encountered in this trail or its primary trail.
        """
        return (str(error) for error in (self.primary_trail or self).__errors)

    @property
    @final
    def errors_stacktrace(self):
        """
        Yields all error stacktraces encountered in this trail or its primary trail.
        """
        yield from (self.primary_trail or self).__errors_stacktrace

    @final
    def __add_error(self, error: Exception, stacktrace: str):
        """
        Adds an error and its stacktrace to the primary trail's error list.

        Args:
            error: The exception that occurred.
            stacktrace: The stacktrace of the exception.
        """
        (self.primary_trail or self).__errors.append(error)
        (self.primary_trail or self).__errors_stacktrace.append(stacktrace)

    @classmethod
    @final
    def get_run_count(cls):
        """
        Returns the number of times this trail has been run.
        """
        return cls.__run_count

    @classmethod
    def primary(cls) -> bool:
        """
        Indicates whether this trail is a primary trail.
        Default implementation returns True.
        """
        return True

    @classmethod
    def hidden(cls) -> bool:
        """
        Indicates whether this trail should be hidden from trail listings.
        Default implementation returns False.
        """
        return False

    @classmethod
    def max_run_count(cls) -> int:
        """
        Returns the maximum number of times this trail can be run.
        Default implementation returns 0 (unlimited).
        """
        return 0

    @classmethod
    def name(cls) -> Iterable[str]:
        """
        Yields the name(s) of this trail, converted from the class name
        using a snake_case convention.
        """
        yield re.sub(
            # 1;
            # Look for an uppercase after a lowercase.
            # HelloWorld = HELLO_WORLD
            # 2;
            # Look for an uppercase followed by a lowercase,
            # after an uppercase or a number.
            # Example; HELLOWorld = HELLO_WORLD
            # 3;
            # Look for a number after a letter.
            # Example; HelloWorld1 = HELLO_WORLD_1
            r"(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z0-9])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)",
            "_",
            cls.__name__,
        ).upper()

    @classmethod
    @final
    def first_name(cls) -> str:  # type: ignore
        """
        Returns the first name of this trail.
        """
        for name in cls.name():
            return name

    @classmethod
    def priority_number(cls) -> float:
        """
        Returns the priority number of this trail.
        Higher priority trails are processed first.
        Default implementation returns 0.
        """
        return 0

    @classmethod
    def condition(cls, name: str):
        """
        Determines whether this trail should be activated for the given name.
        Default implementation checks if the name is in the trail's names.

        Args:
            name: The name to check.

        Returns:
            True if the trail should be activated, False otherwise.
        """
        if cls.primary():
            return name in cls.name()

        return True

    def init(self):
        """
        Initializes the trail. Called before processing begins.
        Default implementation does nothing.
        """
        pass

    @staticmethod
    @final
    def __trail_predicate(trail: Type["Trail"]):
        """
        Determines whether a trail can be run based on its maximum run count.

        Args:
            trail: The trail to check.

        Returns:
            bool: True if the trail can be run, False otherwise.
        """
        max_run_count = trail.max_run_count()

        if max_run_count <= 0:
            return True

        return trail.__run_count < max_run_count

    @classmethod
    @final
    def get_trail(cls, classname: str):
        """
        Retrieves a trail class by its class name.

        Args:
            classname: The name of the trail class to retrieve.

        Returns:
            The trail class if found, None otherwise.
        """
        for trail in cls.all_trails():
            if trail.__class__.__name__ == classname:
                return trail

    @classmethod
    @final
    def all_trails(cls):
        """
        Yields all descendant trail classes that haven't exceeded their maximum run count.
        """
        return get_all_descendant_classes(
            cls,
            exclude=[ABC],
        )

    @classmethod
    @final
    def available_trails(cls):
        """
        Returns all available trail classes, sorted by priority number in descending order.
        """
        return sorted(
            filter(Trail.__trail_predicate, cls.all_trails()),
            key=lambda descendant: descendant.priority_number(),
            reverse=True,
        )

    @classmethod
    @final
    def get_primary_trail(cls, name: str):
        """
        Returns the first trail with the given `name`
        and the highest priority number.
        """

        for trail in cls.get_primary_trails(name):
            return trail

    @classmethod
    @final
    def get_primary_trails(cls, name: str):
        """
        Returns all trails that has a
        satisfied `condition(name)`,
        starting from the highest priority number.

        The trails are instantiated while generating.
        """
        descendants = Trail.available_trails()

        for descendant in descendants:
            if not descendant.primary():
                continue

            ok = descendant.condition(name)

            if not ok:
                continue

            yield descendant()

    def process(self, value) -> Any:
        """
        Processes the given value. Must be implemented by subclasses.

        Args:
            value: The value to process.

        Returns:
            The processed value.
        """
        return value

    def __process(self, value):
        if isgenerator(value):
            for subvalue in value:
                if self.terminated:
                    return

                result = self.process(subvalue)

                if isgenerator(result):
                    yield from result
                    return

                yield result

            return

        yield self.process(value)

    @final
    def run(self, value: Any = None):
        """
        Runs the trail with the given value, processing it and any sub-trails.

        Args:
            value: The value to process.

        Yields:
            Results from processing the value and any sub-trails.
        """
        try:
            self.__run_count += 1

            for sub_trail in self.get_before_trails():
                value = sub_trail().run(value)

                if self.terminated:
                    return value

            value = self.__process(value)

            if self.terminated:
                return value

            for sub_trail in self.get_after_trails():
                value = sub_trail().run(value)

                if self.terminated:
                    return value

            return value
        except Exception as e:
            self.__add_error(
                e,
                traceback.format_exc(),
            )

            raise e

    @classmethod
    @final
    def start_all(
        cls,
        name: str,
        print_trails=True,
        print_indent=2,
    ):
        """
        Starts all primary trails that match the given name and returns the results as a list.

        Args:
            name: The name to match trails against.
            print_trails: Whether to print available trails.
            print_indent: Indentation level for printing.

        Returns:
            A list of results from all matching trails.
        """
        return [
            *cls.start(
                name,
                print_trails,
                print_indent,
            )
        ]

    @classmethod
    @final
    def start(
        cls,
        name: str,
        print_trails=True,
        print_indent=2,
    ):
        """
        Starts all primary trails that match the given name and yields the results.

        Args:
            name: The name to match trails against.
            print_trails: Whether to print available trails.
            print_indent: Indentation level for printing.

        Yields:
            Results from processing each matching trail.
        """
        trails = [*cls.get_primary_trails(name)]

        if print_trails:
            cls.print_available_trails(
                name,
                print_indent,
            )

            cls.__print_load_order(
                trails,
            )

        for trail in trails:
            result = trail.run(None)

            if isgenerator(result):
                yield from result
                continue

            yield result

            if trail.terminated:
                return

    @staticmethod
    @final
    def __print_load_order(
        trails: List["Trail"],
    ):
        """
        Prints the load order of trails based on their priority numbers.

        Args:
            trails: A list of trails to print.
        """
        if not any(trails):
            return

        print(
            f"<{chalk.yellow('Load Order')}>",
            chalk.yellow.bold("↓"),
        )

        items = [
            (
                trail.priority_number(),
                trail.first_name(),
            )
            for trail in trails
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
        item: Type["Trail"],
        name: Optional[str],
    ):
        """
        Formats a trail name for printing with appropriate styling.

        Args:
            item: The trail class.
            name: The name to check conditions against, or None.

        Returns:
            str: Formatted trail name with appropriate styling.
        """
        text = item.first_name()

        if name == None:
            return text

        if not item.primary():
            # Not primary.
            text = chalk.dim.gray(text)

        elif not item.condition(name):
            # Primary, but condition is not met.
            text = f"{text} {chalk.bold('✕')}"

        else:
            # Primary trail.
            text = chalk.green.bold(text)
            text = f"{text} {chalk.bold('✓')}"

        return text

    @staticmethod
    @final
    def __draw_trails(
        name: str,
        trails,
        indent_text: str,
    ):
        """
        Draws trails in a tree-like format with their priority numbers.

        Args:
            name: The name to check conditions against.
            trails: The trails to draw.
            indent_text: The indentation text to use.
        """
        trails0: List[Type["Trail"]] = [trail[0] for trail in trails]
        trails0.sort(
            key=lambda trail: trail.priority_number(),
            reverse=True,
        )

        count = len(trails0)
        priority_numbers = [trail.priority_number() for trail in trails0]
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

        for trail in trails0:
            count -= 1

            priority_number = trail.priority_number()

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
                Trail.__get_printed_name(
                    trail,
                    name,
                ),
            )

        print()

    @staticmethod
    @final
    def __draw_categories(
        name: str,
        indent_size: int,
        indent_scale: int,
        keyword: str,
        category: Any,
    ):
        """
        Draws categorized trails with appropriate indentation.

        Args:
            name: The name to check conditions against.
            indent_size: The base indentation size.
            indent_scale: The indentation scale factor.
            keyword: The category keyword.
            category: The category to draw.

        Yields:
            Tuple: (indent_size, sub_category) pairs for nested categories.
        """
        if keyword == None:
            keyword = "*"

        indent_text = " " * indent_size * indent_scale

        print(f"{indent_text}{chalk.yellow(keyword)}:")

        if isinstance(category, list):
            Trail.__draw_trails(
                name=name,
                trails=category,
                indent_text=indent_text,
            )
            return

        for sub_category in category.items():
            yield indent_size + 1, sub_category

    @classmethod
    @final
    def print_available_trails(
        cls,
        name: str = None,  # type: ignore
        indent: int = 2,
    ):
        """
        Prints all available trails categorized by their names, with indicators for
        which trails match the given name.

        Args:
            name: The name to check trail conditions against.
            indent: Indentation level for printing.
        """
        categorized = [
            (0, pair)
            for pair in categorizer(
                [
                    (
                        trail,
                        trail.first_name(),
                    )
                    for trail in filter(
                        lambda trail: not trail.hidden(),
                        cls.available_trails(),
                    )
                ],
                lambda tuple: tuple[1],
            ).items()
        ]

        while len(categorized) > 0:
            indent_size, (keyword, category) = categorized.pop()

            for sub_category in Trail.__draw_categories(
                name=name,
                indent_size=indent_size,
                indent_scale=indent,
                keyword=keyword,
                category=category,
            ):
                categorized.append(sub_category)
