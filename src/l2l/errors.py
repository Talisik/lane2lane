from .action import Action


class LaneNotFoundError(Exception):
    """Raised when a requested lane cannot be found.

    This error is raised when attempting to access a lane that doesn't exist
    in the system, either by name or by class reference.

    Attributes:
        lane_name: The name of the lane that was not found
    """

    def __init__(self, lane_name):
        self.lane_name = lane_name
        super().__init__(f"Lane '{lane_name}' not found!")


class UnknownActionError(Exception):
    """Raised when an unknown action is encountered.

    This error is raised when an action is encountered that is not recognized
    by the system.
    """

    def __init__(self, action: Action):
        self.action = action

        super().__init__(f"Unknown action: {action.name}")
