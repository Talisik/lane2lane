from abc import ABC

from .types import LaneDictType


class LaneDirection(ABC):
    """
    Abstract base class for lane direction definitions.

    Lane directions define how data flows between lanes in the processing pipeline.
    They provide a way to control the execution flow and data routing between lanes.

    Attributes:
        lanes: A dictionary of lanes indexed by their priority number.
    """

    lanes: LaneDictType

    def __init__(self, lanes: LaneDictType):
        """
        Initialize a lane direction with specified lanes.

        Args:
            lanes: A dictionary of lanes indexed by their priority number.
        """
        self.lanes = lanes


class Fork(LaneDirection):
    """
    A lane direction that forks the data flow to multiple lanes.

    Fork allows the same data to be processed by multiple lanes.
    """

    def __init__(
        self,
        lanes: LaneDictType,
        isolated: bool = False,
    ):
        """
        Initialize a fork with specified lanes.

        Args:
            lanes: A dictionary of lanes indexed by their priority number.
            isolated: If True, the lanes will not return their values for the next lane.
        """
        super().__init__(
            lanes=lanes,
        )

        self.isolated = isolated
