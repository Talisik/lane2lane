# Lane2Lane

Lane2Lane is a Python library for creating flexible, chainable, and prioritized processing pipelines. It allows you to define sequential processing steps (lanes) that can be executed in a specific order with dependency relationships.

For detailed documentation, check out our [Wiki](https://github.com/Talisik/lane2lane/wiki).

## Installation

```bash
pip install lane2lane
```

## Requirements

-   Python 3.8+
-   fun-things

## Quick Start

```python
from l2l import Lane

# Define a simple processing lane
class ProcessingLane(Lane):
    def process(self, value):
        processed_value = f"{value} - processed"
        yield processed_value

# Define a primary lane (entry point) that uses the processing lane
class Main(Lane):
    lanes = {
        -10: ProcessingLane,  # Run ProcessingLane before this lane
    }

    @classmethod
    def primary(cls) -> bool:
        return True  # Entry point — runnable via Lane.start("MAIN")

    def process(self, value):
        result = f"{value} - main"
        yield result

# Run the pipeline
results = Lane.start("MAIN")

# Process the results
for result in results:
    print(result)
```

## Concepts

### Lanes

A Lane is a processing unit that can transform or act on data. Lanes can be:

-   **Primary Lanes**: Entry points that can be directly executed
-   **Regular Lanes**: Processing stages that run as part of a lane chain

### Lane Ordering

Lanes are executed in a specific order defined by:

-   **Priority**: Integer values that determine execution order
-   **Before/After Relationships**: Negative priorities run before, positive priorities run after

## Basic Usage

### Creating a Lane

```python
from l2l import Lane

class MyLane(Lane):
    # Process data and optionally yield results
    def process(self, value):
        processed_value = do_something(value)
        yield processed_value
```

### Creating a Primary Lane

Primary lanes are entry points for execution. Override the `primary` class
method to make a lane runnable via `Lane.start(...)`:

```python
from l2l import Lane

class MyPrimaryLane(Lane):
    @classmethod
    def primary(cls) -> bool:
        return True  # This makes it a primary lane

    def process(self, value):
        # Process the input value
        result = transform_data(value)
        yield result
```

### Defining Lane Order

Lanes can specify other lanes to run before and after them:

```python
class MainLane(Lane):
    @classmethod
    def primary(cls) -> bool:
        return True

    # Define lanes to run before and after this lane
    lanes = {
        -10: "PreprocessLane",   # Run PreprocessLane before this lane (higher negative priority runs first)
        -5: ValidationLane,      # Run ValidationLane after PreprocessLane but before this lane
        0: PostProcessLane,      # Run PostProcessLane after this lane
        10: CleanupLane,         # Run CleanupLane after PostProcessLane
        20: None,                # Use None to remove a lane at this priority
    }

    def process(self, value):
        # Process after PreprocessLane and ValidationLane
        # but before PostProcessLane and CleanupLane
        return transform_data(value)
```

The priority numbers determine the execution order:

-   Negative priorities: Lanes that run before this lane (more negative runs first)
-   Positive priorities: Lanes that run after this lane (higher positive runs first)

### Running Lanes

```python
# Start a specific primary lane
result = Lane.start("MAIN_LANE")

# Start all primary lanes that match a name
results = [*Lane.start("MAIN")]
```

## Data Source Example

A lane can generate its own data instead of processing input from previous
lanes — just `yield` the payloads from `process()`:

```python
from l2l import Lane

class DataSourceLane(Lane):
    @classmethod
    def primary(cls) -> bool:
        return True

    def process(self, value):
        # Fetch data from some source and emit each item downstream
        for item in fetch_data_from_source():
            yield item
```

## Async Lanes

`AsyncLane` is the asynchronous counterpart of `Lane`. Everything works the
same — `lanes`, priorities, `primary()`, `start()` — but `process`/`run`/`start`
are coroutines and you `await` inside them.

```python
import asyncio
from l2l import AsyncLane


class FetchLane(AsyncLane):
    async def process(self, value):
        data = await fetch(value)
        yield data


class Main(AsyncLane):
    lanes = {1: FetchLane}

    @classmethod
    def primary(cls) -> bool:
        return True

    async def process(self, value):
        await asyncio.sleep(0)
        yield "start"


# start() is an async generator — iterate it with `async for`
async def main():
    async for result in AsyncLane.start("MAIN"):
        print(result)

asyncio.run(main())
```

Notes:

-   `process` may be an `async def` returning a value / sync generator / async
    generator, **or** an `async def` with `yield` (an async generator).
-   Inputs may be plain values, sync generators, or async generators.
-   `AsyncLane` keeps its **own registry**, separate from `Lane`, so sync and
    async lanes never mix inside one chain. Reference async lanes from async
    `lanes` dicts only.
-   Items are processed sequentially with `await` (no implicit concurrency).

## Observing Execution (Events)

`l2l.events` is a UI-agnostic hub that emits lane lifecycle events — useful for
dashboards, progress bars, or metrics. Subscribers never affect lane execution
(their exceptions are swallowed), and when there are **no** subscribers the
emit is a cheap no-op.

```python
from l2l import events

@events.subscribe
def on_event(kind, payload):
    if kind == "lane_active":
        print("running:", payload["name"])
```

Event kinds and payloads:

| kind              | payload                                                       |
| ----------------- | ------------------------------------------------------------- |
| `lane_started`    | `run_id`, `name`, `parent_id` (first `process` call)          |
| `lane_active`     | `run_id`, `name`, `parent_id` (a `process` call)              |
| `lane_idle`       | `run_id`, `name`, `work`, `value`                             |
| `lane_done`       | `run_id`, `name`, `duration`, `work`, `terminated`, `errors`  |
| `lane_terminated` | `run_id`, `name`, `terminate_kind`                            |
| `lane_breakpoint` | `run_id`, `name`, `parent_id`, `label`                        |
| `lane_resumed`    | `run_id`, `name`                                              |

-   `run_id` is a stable, unique id per lane instance (a monotonic counter — not
    `id()`, which a finished instance can have reused); `parent_id` is its
    immediate parent's `run_id` (or `None`), so you can nest sub-lanes under
    their parent.
-   `duration` is wall-clock since start (bunches up at pipeline drain for lazy
    chains); `work` is the truthful cumulative time spent inside the lane's own
    `process()` calls.
-   `lane_idle.value` is the **input** to that `process()` call (a concrete
    item/batch — never the output, which may be a generator that must not be
    iterated). `lane_done.errors` is `True` if the lane raised during the run.

## Breakpoints

A dev-only pause, like `pdb.set_trace()` but driven by an observer instead of a
prompt. Call `self.breakpoint()` inside `process()` to halt the pipeline at that
point until something resumes it:

```python
class Enrich(Lane):
    def process(self, value):
        data = fetch(value)
        self.breakpoint("after fetch")   # pauses here…
        return transform(data)
```

Breakpoints are **disarmed by default** — `breakpoint()` does nothing (and costs
nothing) unless a tool arms them. So they are inert in production runs and only
pause under a dev tool (e.g. Carabao's `moo dev` UI, which arms them and binds a
"continue" key):

```python
from l2l import events

events.enable_breakpoints()       # arm (a dev tool does this)
# … observe lane_breakpoint events, then release the paused lane:
events.resume(run_id)             # one lane
events.resume_all()               # every paused lane
events.disable_breakpoints()      # disarm + release everything
```

Use `await self.abreakpoint()` from an `AsyncLane` (it awaits instead of
blocking, and is releasable from another thread). Time spent parked at a
breakpoint is excluded from the lane's `work` total. Pauses log at the dedicated
`PAUSE` level (between `INFO` and `WARNING`) so they're easy to spot.

## Logging

`l2l.logger` is a tiny, dependency-free logger (no loguru). Toggle and level it,
or attach a sink to consume records (e.g. a TUI log pane).

```python
from l2l import logger

logger.disable()              # silence
logger.enable()
logger.set_level("INFO")      # TRACE / DEBUG / INFO / WARNING / ERROR
logger.set_stream(sys.stdout) # default: stderr

# stream records elsewhere (level/message)
logger.add_sink(lambda level, message: my_pane.append(level, message))
```

Lane lifecycle (`initialized`/`started`/`done`/`paused`/`resumed`) logs at
`TRACE`; the `TRACE`/`DEBUG` tags are omitted in console output, other levels
are tagged.

## Terminal Styling

`l2l.style` is a small chainable ANSI styler (no simple-chalk):

```python
from l2l import style

print(style.green.bold("ok"))
print(style.dim.gray("muted"))
style.disable()   # emit plain text (e.g. non-terminal output)
```

## Inline Lanes (Mock)

Instead of defining a class, drop a `dict` (or `Mock`) straight into a `lanes`
map to declare an inline, anonymous sub-pipeline. Use `Mock` when you need to
set `isolated` / `process_mode` on that inline group:

```python
from l2l import Lane, Mock

class Main(Lane):
    lanes = {
        1: {0: StepA, 1: StepB},          # plain dict → inline group (defaults)
        2: Mock(isolated=True, lanes={     # Mock → inline group with config
            0: SideEffectA,
            1: SideEffectB,
        }),
    }
```

## Advanced Features

### Conditional Execution

Lanes can have conditions for execution:

```python
class ConditionalLane(Lane):
    @classmethod
    def condition(cls, name: str):
        # Only run this lane if the name contains "SPECIAL"
        return "SPECIAL" in name
```

### Custom Naming

Provide custom names or aliases for lanes:

```python
class CustomNamedLane(Lane):
    @classmethod
    def name(cls) -> Iterable[str]:
        yield "CUSTOM_PROCESS"
        yield "PROCESSOR"  # An alias
```

### Maximum Run Count

Limit how many times a lane can run:

```python
class OneTimeLane(Lane):
    @classmethod
    def max_run_count(cls) -> int:
        return 1  # Run this lane only once
```

### Process All Values

Control whether all items should be processed before passing to the next lane:

```python
class BatchProcessingLane(Lane):
    process_all = True  # Process all items before passing to the next lane

    def process(self, value):
        # When process_all is True, all items will be processed by this lane
        # before any are passed to subsequent lanes
        yield processed_value
```

When `process_all` is False (default), each item is processed through the entire lane chain before the next item starts processing.

### Terminating Lane Execution

You can manually terminate a lane's execution:

```python
class TerminatingLane(Lane):
    def process(self, value):
        if some_condition:
            self.terminate()  # Stop processing this lane
            return

        yield processed_value
```

### Multiprocessing Support

Lane2Lane supports multiprocessing for parallel data processing:

```python
class ParallelProcessingLane(Lane):
    multiprocessing = True  # Enable multiprocessing for this lane

    def process(self, value):
        # Process data in parallel
        # Each yielded item will be processed by subsequent lanes
        yield processed_item
```

### Error Handling

Lanes provide built-in error handling capabilities:

```python
class ErrorHandlingLane(Lane):
    @classmethod
    def terminate_on_error(cls):
        return True  # Stop processing on error (default behavior)

    def process(self, value):
        try:
            # Process data
            yield processed_data
        except Exception as e:
            # Access errors with self.errors
            # Global errors available via Lane.global_errors()
            pass
```

## Complete Example

Here's a complete example showing a data processing pipeline:

```python
from l2l import Lane

# Data source that fetches records
class DataSourceLane(Lane):
    def process(self, value):
        data = [
            {"id": 1, "name": "Alice", "score": 85},
            {"id": 2, "name": "Bob", "score": 92},
            {"id": 3, "name": "Charlie", "score": 78},
        ]
        for item in data:
            yield item

# Validation lane
class ValidationLane(Lane):
    def process(self, value):
        if "id" not in value or "name" not in value:
            raise ValueError(f"Invalid data format: {value}")
        yield value

# Processing lane
class ScoreProcessingLane(Lane):
    def process(self, value):
        # Add grade based on score
        if "score" in value:
            if value["score"] >= 90:
                value["grade"] = "A"
            elif value["score"] >= 80:
                value["grade"] = "B"
            elif value["score"] >= 70:
                value["grade"] = "C"
            else:
                value["grade"] = "D"
        yield value

# Output formatting lane
class FormattingLane(Lane):
    def process(self, value):
        yield f"Student {value['name']} (ID: {value['id']}) - Score: {value['score']}, Grade: {value.get('grade', 'N/A')}"

# Main primary lane that orchestrates the pipeline
class StudentProcessingLane(Lane):
    @classmethod
    def primary(cls) -> bool:
        return True

    lanes = {
        -30: DataSourceLane,       # First fetch the data
        -20: ValidationLane,       # Then validate it
        -10: ScoreProcessingLane,  # Then process scores
        0: FormattingLane,         # Finally format for output
    }

    # Note: No need to implement process() if you're just passing values through
    # The Lane class already handles this behavior by default

# Run the pipeline
results = Lane.start("STUDENT_PROCESSING")
for result in results:
    print(result)
```

Output:

```
Student Alice (ID: 1) - Score: 85, Grade: B
Student Bob (ID: 2) - Score: 92, Grade: A
Student Charlie (ID: 3) - Score: 78, Grade: C
```

## License

[MIT License](LICENSE)
