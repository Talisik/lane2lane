from dataclasses import dataclass, field

from .types import LaneDictType, ProcessModeType


@dataclass(frozen=True)
class Mock:
    """A mock object that simulates a lane for runtime lane creation.

    This class provides a way to create mock lanes dynamically at runtime
    without requiring predefined lane implementations. Mock lanes can be
    used to inject custom behavior into the lane processing pipeline.

    Attributes:
        isolated (bool): Whether the mock lane should run in isolation. Defaults to False.
        process_mode (ProcessModeType): The processing mode for the mock lane.
            Can be "one" or "all". Defaults to "one".
        multiprocessing (bool): Whether the mock lane supports multiprocessing. Defaults to True.
        lanes (LaneDictType): A dictionary mapping lane names to their mock implementations.
            Defaults to an empty dictionary.
    """

    isolated: bool = False
    process_mode: ProcessModeType = "one"
    multiprocessing: bool = True
    lanes: LaneDictType = field(default_factory=dict)
