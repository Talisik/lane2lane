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
- ``lane_active``     — ``run_id``, ``name``, ``parent_id`` (a ``process()`` call is starting)
- ``lane_idle``       — ``run_id``, ``name``, ``work``, ``value`` (that call
  returned; ``value`` is the lane's output, a generator passed as-is — observers
  must not iterate it)
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

Breakpoints
-----------

A dev-only pause mechanism (see :meth:`l2l._lane_core._LaneCore.breakpoint`).
Disabled by default — :meth:`_Events.breakpoints_enabled` is False, so
``breakpoint()`` is a no-op (and costs nothing) in ``moo run``. A dev tool
enables it, observes ``lane_breakpoint`` / ``lane_resumed`` events, and calls
:meth:`_Events.resume` / :meth:`_Events.resume_all` to let the paused lane
proceed.

- ``lane_breakpoint`` — ``run_id``, ``name``, ``parent_id``, ``label`` (lane paused)
- ``lane_resumed``    — ``run_id``, ``name`` (lane released and continuing)
"""

import asyncio
import threading
from typing import Any, Callable, Dict, List, Union

EventCallback = Callable[[str, Dict[str, Any]], None]


class _SyncGate:
    """A one-shot pause for a synchronous lane (blocks the calling thread)."""

    def __init__(self):
        self._event = threading.Event()

    def wait(self):
        self._event.wait()

    def release(self):
        self._event.set()


class _AsyncGate:
    """A one-shot pause for an async lane (releasable from another thread).

    ``asyncio.Event`` is not thread-safe, so ``release`` (called from the dev
    tool's thread) marshals the ``set`` back onto the lane's event loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._event = asyncio.Event()

    async def wait(self):
        await self._event.wait()

    def release(self):
        self._loop.call_soon_threadsafe(self._event.set)


_Gate = Union[_SyncGate, _AsyncGate]


class _Events:
    def __init__(self):
        self._subscribers: List[EventCallback] = []
        self._breakpoints_enabled = False
        #: run_id -> gate for lanes currently paused at a breakpoint.
        self._gates: Dict[int, _Gate] = {}

    @property
    def has_subscribers(self) -> bool:
        """True if anyone is observing — lets hot paths skip instrumentation."""
        return bool(self._subscribers)

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

    # ---- breakpoints -----------------------------------------------------

    @property
    def breakpoints_enabled(self) -> bool:
        """Whether ``breakpoint()`` actually pauses (dev tools turn this on)."""
        return self._breakpoints_enabled

    def enable_breakpoints(self):
        """Arms breakpoints so ``lane.breakpoint()`` calls start pausing."""
        self._breakpoints_enabled = True

    def disable_breakpoints(self):
        """Disarms breakpoints and releases anything currently paused."""
        self._breakpoints_enabled = False
        self.resume_all()

    def _register_gate(self, run_id: int, gate: _Gate):
        """Records a paused lane's gate (used internally by ``breakpoint``)."""
        self._gates[run_id] = gate

    def _clear_gate(self, run_id: int):
        """Forgets a gate once its lane has resumed."""
        self._gates.pop(run_id, None)

    def resume(self, run_id: int):
        """Releases the lane paused at the breakpoint with this ``run_id``."""
        gate = self._gates.get(run_id)
        if gate is not None:
            gate.release()

    def resume_all(self):
        """Releases every lane currently paused at a breakpoint."""
        for gate in list(self._gates.values()):
            gate.release()


#: Shared lane lifecycle event hub.
events = _Events()
