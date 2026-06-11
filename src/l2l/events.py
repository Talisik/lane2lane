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

- ``lane_started``    — ``run_id``, ``name``, ``parent_id`` (generator entered)
- ``lane_active``     — ``run_id``, ``name`` (a ``process()`` call is starting)
- ``lane_idle``       — ``run_id``, ``name``, ``work`` (that call returned)
- ``lane_done``       — ``run_id``, ``name``, ``duration``, ``work``, ``terminated``
  (generator drained)

``lane_started``/``lane_done`` track the lazily-chained generator lifecycle, so
they bunch up (all start ≈ together, all finish at pipeline drain).
``lane_active``/``lane_idle`` wrap the actual ``process()`` calls, which run
sequentially — use these to show which lane is computing *now*. ``duration`` is
wall-clock since start; ``work`` is cumulative ``process()`` time (truthful).
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
