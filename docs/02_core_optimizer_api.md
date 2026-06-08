# 02. Core Optimizer API

This page explains the reusable optimizer.

The optimizer needs three things:

```text
1. choice_sizes
2. X_init
3. y_init
```

Then it repeatedly asks for one more thing:

```text
objective score for the proposed vector
```

## The Vector

FLIWBO works on integer vectors.

Example:

```text
x = [2, 0, 4]
```

Each number is a choice. If `choice_sizes = [4, 3, 5]`, then:

```text
x[0] can be 0, 1, 2, or 3
x[1] can be 0, 1, or 2
x[2] can be 0, 1, 2, 3, or 4
```

The optimizer does not know what the choices mean. Your integration decides that.

## The Objective Function

The objective function receives one vector and returns one number:

```python
def objective(x_vector) -> float:
    ...
```

Higher scores are better.

For example:

```python
def objective(x_vector) -> float:
    return number_of_tasks_solved - cost_penalty
```

## Simple Run

Use this when the objective is cheap or when you are testing locally:

```python
import numpy as np

from fliwbo_core import DiscreteSearchSpace, FLIWBOConfig, FLIWBOOptimizer
from fliwbo_core import PROptimizerConfig


search_space = DiscreteSearchSpace([4, 3, 5])


def objective(x_vector) -> float:
    target = np.array([2, 1, 4])
    return -float(np.sum((x_vector - target) ** 2))


X_init = np.array([
    [0, 0, 0],
    [3, 2, 4],
])
y_init = np.array([objective(x) for x in X_init])

config = FLIWBOConfig(
    n_iters=10,
    pr_config=PROptimizerConfig(
        num_restarts=5,
        num_steps=10,
        num_samples=16,
    ),
)

optimizer = FLIWBOOptimizer(search_space, config=config)
result = optimizer.run(objective, X_init, y_init)
```

The result contains all observed vectors and scores:

```python
print(result.x_observed)
print(result.y_observed)
print(result.best_x)
print(result.best_y)
```

## Crash-Resistant Run

Use this when each evaluation is expensive.

```python
run = optimizer.start(X_init, y_init, run_dir="runs/my_run")

proposal = run.ask()
score = evaluate_external_system(proposal.x_vector)
run.tell(proposal, score)
```

`ask()` writes the proposal to disk before returning it.

`tell()` writes the completed score to disk before updating memory.

If the process crashes after `ask()`:

```python
run = FLIWBOOptimizer.resume("runs/my_run")
proposal = run.ask()
```

That returns the same pending proposal. It does not silently invent a new vector.

## Durable Run Files

A run directory contains:

```text
runs/my_run/
|-- manifest.json
|-- observations.csv
|-- proposals.csv
`-- events.jsonl
```

Use these files like this:

```text
manifest.json     = what config and search space started this run
observations.csv  = initial and completed x/y observations
proposals.csv     = proposed vectors, status, acquisition, warp metadata
events.jsonl      = append-only history of important events
```

The final `OptimizationResult` is convenient for code. The run directory is the
recovery source of truth.

## Important Classes

```text
DiscreteSearchSpace       finite product of integer choices
FLIWBOConfig              optimizer settings
PROptimizerConfig         acquisition optimizer settings
FLIWBOOptimizer           main user-facing optimizer
OptimizationRun           active ask/tell run
OptimizationProposal      vector proposed by ask()
BOIterationRecord         completed proposal plus score
OptimizationResult        convenient final/in-memory result
```

## Main Knobs

```text
n_iters              number of BO iterations
noise_std            assumed objective noise
lengthscale          base GP kernel lengthscale
epsilon_warp         resolution of the finite warp library
use_warp_prior       whether to favor near-identity warps
warp_search_sweeps   coordinate-wise warp search sweeps
warp_search_n_jobs   parallel workers for warp scoring
pr_config            settings for acquisition search over integer vectors
```

For a first integration, change as little as possible. Start with small
`n_iters`, small `PROptimizerConfig`, and a deterministic dry-run objective.
