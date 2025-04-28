# Lane2Lane

Lane2Lane is a Python library for creating flexible, chainable, and prioritized processing pipelines. It allows you to define sequential processing steps (lanes) that can be executed in a specific order with dependency relationships.

For detailed documentation, check out our [Wiki](https://github.com/Talisik/lane2lane/wiki).

## Installation

```bash
pip install lane2lane
```

```bash
pip install git+https://github.com/Talisik/lane2lane.git
```

## Requirements

-   Python 3.8+
-   fun-things
-   simple-chalk

## Quick Start

```python
from l2l import Lane, PrimaryLane

# Define a simple processing lane
class ProcessingLane(Lane):
    def process(self, value):
        processed_value = f"{value} - processed"
        yield processed_value

# Define a primary lane that uses the processing lane
class Main(PrimaryLane):
    lanes = {
        -10: ProcessingLane,  # Run ProcessingLane before this lane
    }

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

Primary lanes are entry points for execution:

```python
from l2l import PrimaryLane

class MyPrimaryLane(PrimaryLane):
    def process(self, value):
        # Process the input value
        result = transform_data(value)
        yield result
```

You can also override the 'primary' class method in a Lane class:

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
class MainLane(PrimaryLane):
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
results = Lane.start_all("MAIN")
```

## Subscriber Example

Subscriber is a pre-defined lane class that provides a standard way to generate data. Rather than processing input from previous lanes, Subscriber lanes generate their own payloads:

```python
from l2l import Subscriber

class DataSourceLane(Subscriber):
    def get_payloads(self, value):
        # Fetch data from some source
        data = fetch_data_from_source()
        for item in data:
            yield item
```

Instead of implementing `process()`, you only need to implement `get_payloads()` to define where your data comes from. The Subscriber class handles the rest automatically.

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
from l2l import Lane, PrimaryLane, Subscriber

# Data source that fetches records
class DataSourceLane(Subscriber):
    def get_payloads(self, value):
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
class StudentProcessingLane(PrimaryLane):
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
