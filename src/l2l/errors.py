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
