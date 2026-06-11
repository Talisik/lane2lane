import sys
import traceback as _traceback
from typing import Callable, Dict, List

_LEVELS: Dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}

#: A sink receives ``(level, message)`` for every emitted record.
Sink = Callable[[str, str], None]


class Logger:
    """Minimal, dependency-free logger for lane2lane.

    Replaces the previous ``loguru`` dependency. Can be toggled on/off and
    leveled. Messages use ``str.format`` positional style, matching the existing
    ``logger.debug("N-{0} {1}", a, b)`` call sites.

    Access the shared instance via ``from l2l import logger`` and configure it::

        logger.disable()              # silence completely
        logger.enable()               # turn back on
        logger.set_level("INFO")      # hide DEBUG
        logger.set_stream(sys.stdout) # redirect output
    """

    def __init__(self):
        self.enabled: bool = True
        self.level: str = "DEBUG"
        self._stream = sys.stderr
        self._sinks: List[Sink] = []

    def enable(self):
        """Turns logging on."""
        self.enabled = True

    def disable(self):
        """Turns logging off."""
        self.enabled = False

    def set_level(self, level: str):
        """Sets the minimum level to emit (``DEBUG``/``INFO``/``WARNING``/``ERROR``)."""
        level = level.upper()

        if level not in _LEVELS:
            raise ValueError(f"Unknown log level: {level!r}")

        self.level = level

    def set_stream(self, stream):
        """Redirects output to the given file-like stream."""
        self._stream = stream

    def add_sink(self, sink: Sink) -> Sink:
        """Registers a sink that receives ``(level, message)`` per record.

        Sinks fire regardless of ``_stream`` and let tools (e.g. a TUI log
        pane) consume records. Level gating still applies. Returns the sink.
        """
        if sink not in self._sinks:
            self._sinks.append(sink)

        return sink

    def remove_sink(self, sink: Sink):
        """Removes a previously registered sink."""
        if sink in self._sinks:
            self._sinks.remove(sink)

    def _enabled_for(self, level: str) -> bool:
        return self.enabled and _LEVELS[level] >= _LEVELS[self.level]

    def _emit(self, level: str, message: str):
        # DEBUG is the common lifecycle log — keep it unlabeled; tag the rest.
        prefix = "" if level == "DEBUG" else f"[{level}] "
        print(f"{prefix}{message}", file=self._stream)

        for sink in list(self._sinks):
            try:
                sink(level, message)
            except Exception:
                pass

    def _log(self, level: str, message: str, *args, **kwargs):
        if not self._enabled_for(level):
            return

        if args or kwargs:
            try:
                message = message.format(*args, **kwargs)
            except Exception:
                pass

        self._emit(level, message)

    def debug(self, message: str, *args, **kwargs):
        """Logs at DEBUG. ``message`` is ``str.format``-ed with ``*args``/``**kwargs``."""
        self._log("DEBUG", message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Logs at INFO. ``message`` is ``str.format``-ed with ``*args``/``**kwargs``."""
        self._log("INFO", message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Logs at WARNING. ``message`` is ``str.format``-ed with ``*args``/``**kwargs``."""
        self._log("WARNING", message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Logs at ERROR. ``message`` is ``str.format``-ed with ``*args``/``**kwargs``."""
        self._log("ERROR", message, *args, **kwargs)

    def exception(self, error: BaseException, *args, **kwargs):
        """Logs an exception at ERROR level with its traceback."""
        if not self._enabled_for("ERROR"):
            return

        self._log("ERROR", str(error), *args, **kwargs)
        _traceback.print_exc(file=self._stream)


#: Shared logger instance used across lane2lane.
logger = Logger()
