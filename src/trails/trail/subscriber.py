from .trail import Trail


class Subscriber(Trail):
    trails = {
        1: "A",
    }

    def process(self, value):
        print(value)
