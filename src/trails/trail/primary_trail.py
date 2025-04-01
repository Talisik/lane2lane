from .trail import Trail


class PrimaryTrail(Trail):
    @classmethod
    def primary(cls) -> bool:
        return True
