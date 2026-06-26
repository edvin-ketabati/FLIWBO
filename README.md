# FLIWBO

Finite-library input-warped Bayesian optimization for typed design spaces.

FLIWBO is a small optimizer package plus one worked example. It proposes a
vector, your integration evaluates that vector, and FLIWBO uses the returned
score to decide what to try next.

The vector can contain:

```text
Discrete(k)          one integer choice in 0..k-1
Continuous(a, b)     one float inside [a, b]
```
## Project Status / Disclaimer

This repository is shared as the code from my master's thesis project. I was a
student when I wrote it, and I am not a professional programmer. The repository
is mainly an attempt to make the work more available to people/organizations who are
interested in it.

I do not promise active maintenance, support, bug fixes, or help integrating the
code into other projects. The code may be inefficient, incomplete, or contain
errors. Please treat it as research/thesis code that is made public in good
faith, rather than as a polished or production-ready software package.

## What Is In This Repo

```text
.
|-- src/fliwbo_core/        # the reusable optimizer package
|-- examples/quixbugs/      # one full integration example
|-- examples/notebooks/     # visual demo of hyperparameters
|-- docs/                   # step-by-step docs
|-- pyproject.toml          # package metadata
`-- requirements.txt        # pinned development environment snapshot
```

The core package does not know about QuixBugs, LLMs, tools, agents, or API keys.
It only knows about typed vectors and scalar scores.

The QuixBugs folder shows one way to connect discrete vectors to a real
multi-agent runtime and evaluation loop.

## Read The Docs In This Order

1. [Repository Tour](docs/01_repository_tour.md)
   Start here. It explains what each folder is for.

2. [Core Optimizer API](docs/02_core_optimizer_api.md)
   Explains the optimizer, the input vector, the objective score, and the
   crash-resistant ask/tell loop.

3. [Integration Guide](docs/03_integration_guide.md)
   Explains how to plug in your own tools, endpoints, runtime, and scoring
   function.

4. [QuixBugs Walkthrough](docs/04_quixbugs_walkthrough.md)
   Explains the example integration in this repo, from vector to MAS spec to
   score.

5. [Hyperparameter Effects](docs/05_hyperparameter_effects.md)
   Explains the optimizer settings and links to the visual notebook demo.

## Quick Install

From the repository root:

```bash
pip install -e .
```

For the QuixBugs example dependencies:

```bash
pip install -e ".[quixbugs]"
```

For the visual hyperparameter notebook:

```bash
pip install -e ".[notebooks]"
python -m notebook examples/notebooks/01_hyperparameter_visualization.ipynb
```

## Core Example

```python
import numpy as np

from fliwbo_core import Discrete, FLIWBOConfig, FLIWBOOptimizer, PROptimizerConfig
from fliwbo_core import SearchSpace


search_space = SearchSpace([
    Discrete(4),
    Discrete(3),
    Discrete(5),
])


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
result = optimizer.run(objective, X_init, y_init, run_dir="runs/tiny_example")

print(result.best_x)
print(result.best_y)
```

## Compute Backend

By default, `FLIWBOConfig(backend="auto", device="auto")` uses the Torch
backend. It runs on CUDA when PyTorch can see a CUDA device, otherwise it runs
on CPU. The Torch backend keeps the same optimizer mechanism: the same finite
Beta-CDF warp library, the same coordinate-wise warp search, the same fixed
Matern-5/2 GP with white noise, and the same PR acquisition search.

To force a specific mode:

```python
config = FLIWBOConfig(
    backend="torch",   # "auto", "torch", or "sklearn"
    device="cuda",     # "auto", "cpu", "cuda", or "cuda:N"
)
```

Use `backend="sklearn"` if you want the original sklearn GP path for debugging
or comparison. Use `backend="torch", device="cpu"` to exercise the accelerated
code path without requiring a GPU.

Mixed spaces use the same optimizer:

```python
from fliwbo_core import Continuous, Discrete, SearchSpace


search_space = SearchSpace([
    Discrete(4),
    Continuous(-1.0, 1.0),
    Discrete(5),
])
```

In proposals, discrete coordinates are plain Python `int` values and continuous
coordinates are `float` values. In NumPy arrays for mixed runs, all coordinates
share a float dtype, but discrete entries remain integer-valued.

## Crash-Resistant Mode

For expensive evaluations, use ask/tell:

```python
run = optimizer.start(X_init, y_init, run_dir="runs/partner_run_001")

proposal = run.ask()
score = evaluate_external_system(proposal.x_vector)
run.tell(proposal, score)
```

If the process crashes after `ask()`, resume it:

```python
run = FLIWBOOptimizer.resume("runs/partner_run_001")
proposal = run.ask()  # returns the pending proposal
```

The durable run folder contains:

```text
manifest.json
observations.csv
proposals.csv
events.jsonl
```

The final return value is convenient. The run folder is the recovery source of
truth.
