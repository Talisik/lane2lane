"""Lane lifecycle event hub (UI-agnostic).

Lets external tools observe lane execution in real time without l2l depending
on any UI library. Subscribe a callback and receive ``(kind, payload)`` tuples
as lanes run::

    from l2l import events

    def on_event(kind, payload):
        if kind == "lane_started":
            print("active:", payload["name"])

    events.subscribe(on_event)

Event kinds and payload keys:

- ``lane_started``    — ``run_id``, ``name``, ``parent_id``
- ``lane_done``       — ``run_id``, ``name``, ``duration``, ``work``, ``terminated``
  (``duration`` is wall-clock since start — which bunches up at pipeline drain
  for lazy/streaming lanes; ``work`` is the cumulative time spent inside this
  lane's own ``process()`` calls, i.e. its truthful compute time.)
- ``lane_terminated`` — ``run_id``, ``name``, ``terminate_kind``

``run_id`` is the lane instance's identity; ``parent_id`` is the primary lane's
``run_id`` (or ``None`` for a primary lane), so consumers can nest sub-lanes
under their primary.
"""

from typing import Any, Callable, Dict, List

EventCallback = Callable[[str, Dict[str, Any]], None]


class _Events:
    def __init__(self):
        self._subscribers: List[EventCallback] = []

    def subscribe(self, callback: EventCallback) -> EventCallback:
        """Registers a callback. Returns it (handy as a decorator)."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

        return callback

    def unsubscribe(self, callback: EventCallback):
        """Removes a previously registered callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def clear(self):
        """Removes all callbacks."""
        self._subscribers.clear()

    def emit(self, kind: str, **payload):
        """Emits an event to all subscribers.

        Subscriber exceptions are swallowed so observers can never break lane
        execution.
        """
        if not self._subscribers:
            return

        for callback in list(self._subscribers):
            try:
                callback(kind, payload)
            except Exception:
                pass


#: Shared lane lifecycle event hub.
events = _Events()
