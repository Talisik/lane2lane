import fun_things.logger
from fun_things import lazy


@lazy.fn
def LOGGER():
    return fun_things.logger.new("l2l")
