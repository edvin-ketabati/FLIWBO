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
The optimizer can relearn the "distance" between two choices, but not the ordering.
So, make sure the ordering reflects the real and intended structure.

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
    warp_prior_weight=0.005,
    warp_prior_tau=0.75,
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

That returns the same pending proposal.

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
use_warp_prior       whether to favor unity warps
warp_prior_weight    strength of the prior toward alpha=beta=1
warp_prior_tau       width of the prior around alpha=beta=1
warp_search_sweeps   coordinate-wise warp search sweeps
warp_search_n_jobs   parallel workers for warp scoring
pr_config            settings for acquisition search over integer vectors
```

For a first integration, change as little as possible. Start with small
`n_iters`, small `PROptimizerConfig`, and a deterministic dry-run objective.

## The Warp Prior

The warp is a Beta-CDF transform applied to each normalized input coordinate.

Each coordinate gets two warp parameters:

```text
alpha
beta
```

When:

```text
alpha = 1
beta = 1
```

the Beta-CDF warp is the identity transform. In plain words, this is the
unity warp: no input stretching, no input squeezing, and no change to the
distances the base GP sees. The finite warp library always includes this exact
unity warp.

The warp prior is a soft preference for that unity warp.

```python
config = FLIWBOConfig(
    use_warp_prior=True,
    warp_prior_weight=0.005,
    warp_prior_tau=0.75,
)
```

The optimizer scores each candidate warp like this:

```text
warp score = GP log marginal likelihood + warp_prior_weight * log_prior
```

The `log_prior` is highest at `alpha=beta=1`. It becomes more negative as
`alpha` and `beta` move away from 1.

Use the prior as a brake on aggressive warps:

```text
higher warp_prior_weight = stronger pull toward unity
lower warp_prior_weight  = more freedom to choose strong warps
warp_prior_weight = 0    = no practical prior penalty
```

Tune `warp_prior_weight` and `warp_prior_tau` as a pair. In the current
implementation, candidate ranking depends on the effective prior strength:

```text
warp_prior_weight / warp_prior_tau^2
```

Lower effective strength gives the warp search more freedom. Higher effective
strength pulls harder toward unity. Too much regularization can keep the model
too close to the no-warp case.

You can disable the prior with either:

```python
FLIWBOConfig(use_warp_prior=False)
```

or:

```python
FLIWBOConfig(warp_prior_weight=0.0)
```
