# FLIWBO

Finite-library input-warped Bayesian optimization for discrete design spaces.

FLIWBO is a small optimizer package plus one worked example. The optimizer asks:

```text
Which integer vector should we try next?
```

Your system answers:

```text
Here is the score for that vector.
```

Then FLIWBO learns from the score and proposes the next vector.

## What Is In This Repo

```text
.
|-- src/fliwbo_core/        # the reusable optimizer package
|-- examples/quixbugs/      # one full integration example
|-- docs/                   # step-by-step handoff docs
|-- pyproject.toml          # package metadata
`-- requirements.txt        # pinned development environment snapshot
```

The core package does not know about QuixBugs, LLMs, tools, agents, Docker, or
API keys. It only knows about integer vectors and scalar scores.

The QuixBugs folder shows one way to connect those integer vectors to a real
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

## Quick Install

From the repository root:

```bash
pip install -e .
```

For the QuixBugs example dependencies:

```bash
pip install -e ".[quixbugs]"
```

## Tiny Core Example

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
result = optimizer.run(objective, X_init, y_init, run_dir="runs/tiny_example")

print(result.best_x)
print(result.best_y)
```

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
