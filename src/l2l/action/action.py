from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    name: str
    kwargs: dict = field(default_factory=dict)
