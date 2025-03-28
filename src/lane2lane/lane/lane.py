import re
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, final

from fun_things import categorizer, get_all_descendant_classes
from simple_chalk import chalk

from ..constants import TERMINATE

T = TypeVar("T")


class Lane(ABC):
    sub_lanes: Dict[int, Optional[Type["Lane"]]] = {}

    __run_count: int = 0

    def __init__(
        self,
        primary_lane: Optional["Lane"],
    ):
        self.__primary_lane = primary_lane
        self.__kwargs = {}

    def get(self, key: str, default: Any = None):
        if self.__primary_lane != None:
            return self.__primary_lane.get(key, default)

        return self.__kwargs.get(key, default)

    def set(self, key: str, value: T) -> T:
        if self.__primary_lane != None:
            return self.__primary_lane.set(key, value)

        self.__kwargs[key] = value

        return value

    @property
    @final
    def primary_lane(self):
        return self.__primary_lane

    @classmethod
    @final
    def get_run_count(cls):
        return cls.__run_count

    @classmethod
    def primary(cls) -> bool:
        return True

    @classmethod
    def hidden(cls) -> bool:
        return False

    @classmethod
    def max_run_count(cls) -> int:
        return 0

    @classmethod
    def name(cls) -> Iterable[str]:
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
        for name in cls.name():
            return name

    @classmethod
    def priority_number(cls) -> float:
        return 0

    @classmethod
    def condition(cls, name: str):
        return name in cls.name()

    def init(self):
        pass

    @staticmethod
    @final
    def __lane_predicate(lane: Type["Lane"]):
        max_run_count = lane.max_run_count()

        if max_run_count <= 0:
            return True

        return lane.__run_count < max_run_count

    @classmethod
    @final
    def available_lanes(cls):
        return sorted(
            [
                descendant
                for descendant in get_all_descendant_classes(
                    cls,
                    exclude=[ABC],
                )
                if Lane.__lane_predicate(descendant)
            ],
            key=lambda descendant: descendant.priority_number(),
            reverse=True,
        )

    @classmethod
    @final
    def get_primary_lane(cls, name: str):
        """
        Returns the first lane with the given `name`
        and the highest priority number.
        """
        for lane in cls.get_primary_lanes(name):
            return lane

    @classmethod
    @final
    def get_primary_lanes(cls, name: str):
        """
        Returns all lanes that has a
        satisfied `condition(name)`,
        starting from the highest priority number.

        The lanes are instantiated while generating.
        """
        descendants = Lane.available_lanes()

        for descendant in descendants:
            if not descendant.primary():
                continue

            ok = descendant.condition(name)

            if not ok:
                continue

            yield descendant(
                primary_lane=None,
            )

    def get_sub_lanes(self) -> Dict[int, Type["Lane"]]:
        return {}

    def __get_sub_lanes(self):
        result = {
            **self.sub_lanes,
        }

        result.update(self.get_sub_lanes())

        return result

    @abstractmethod
    def process(self, **kwargs) -> Any:
        pass

    def __run(self, **kwargs):
        value = self.process(
            kwargs=kwargs,
        )

        yield value

        if value == TERMINATE:
            return

        sub_lanes = {k: v for k, v in self.__get_sub_lanes().items() if v is not None}

        for sub_lane in (
            sub_lanes[key](
                primary_lane=self.__primary_lane or self,
            )
            for key in sorted(
                sub_lanes.keys(),
                reverse=True,
            )
            if sub_lanes[key].condition(self.first_name())
        ):
            for value in sub_lane.run(
                kwargs=kwargs,
            ):
                yield value

                if value == TERMINATE:
                    return

    @final
    def run(self, **kwargs):
        for value in self.__run(
            **kwargs,
        ):
            if value == TERMINATE:
                return

            yield value

    @classmethod
    @final
    def start_all(
        cls,
        name: str,
        print_lanes=True,
        print_indent=2,
    ):
        return [
            *cls.start(
                name,
                print_lanes,
                print_indent,
            )
        ]

    @classmethod
    @final
    def start(
        cls,
        name: str,
        print_lanes=True,
        print_indent=2,
    ):
        lanes = [*cls.get_primary_lanes(name)]

        if print_lanes:
            cls.print_available_lanes(
                name,
                print_indent,
            )

            cls.__print_load_order(
                lanes,
            )

        __errors: List[Exception] = []
        __errors_str: List[str] = []
        __errors_stacktrace: List[str] = []

        for lane in lanes:
            stop = False

            try:
                for item in lane.run(
                    kwargs=dict(
                        __errors=__errors,
                        __errors_str=__errors_str,
                        __errors_stacktrace=__errors_stacktrace,
                    ),
                ):
                    if item == TERMINATE:
                        stop = True
                        break

                    yield item

            except Exception as e:
                traceback.print_exc()
                __errors.append(e)
                __errors_str.append(str(e))
                __errors_stacktrace.append(traceback.format_exc())

            if stop:
                break

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

        if name == None:
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
        if keyword == None:
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

    @classmethod
    @final
    def print_available_lanes(
        cls,
        name: str = None,  # type: ignore
        indent: int = 2,
    ):
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
                        cls.available_lanes(),
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
