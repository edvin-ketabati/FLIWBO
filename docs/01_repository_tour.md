# 01. Repository Tour

This repo has two main pieces:

```text
src/fliwbo_core/
examples/quixbugs/
```

Think of them like this:

```text
fliwbo_core       = the optimizer brain
examples/quixbugs = one body built around that brain
```

The optimizer brain only chooses integer vectors. It does not know what those
integers mean.

The QuixBugs example explains one way to turn those integers into a multi-agent
system, run it, test it, and send a score back to the optimizer.

## Top-Level Files

```text
README.md          # start here
docs/              # longer handoff docs
src/               # installable Python package
examples/          # example integrations
pyproject.toml     # package install metadata
requirements.txt   # pinned development dependency snapshot
.gitignore         # generated files that should not be committed
```

## Core Package

```text
src/fliwbo_core/
|-- optimizer.py        # public API: run, start, ask, tell, resume
|-- BO_loop.py          # compatibility wrapper for older scripts
|-- BO_utils.py         # beta warps and BO schedules
|-- warp_optimizer.py   # chooses the best input warp from a finite library
|-- PR_optimizer.py     # searches the discrete space for the best acquisition value
|-- discrete_space.py   # maps integer vectors into the model input domain
|-- decoder.py          # helper for decoding ordered resource maps
`-- BO_config.py        # default knobs and constants
```

Most external users should start with:

```python
from fliwbo_core import DiscreteSearchSpace, FLIWBOConfig, FLIWBOOptimizer
```

## QuixBugs Example

```text
examples/quixbugs/
|-- resource_statement.py   # LLMs, tools, prompts, agent limits
|-- encoding_map.json       # ordered choices used by the vector encoding
|-- search_space.py         # computes choice sizes for the QuixBugs vector
|-- features_to_spec.py     # turns a vector into mas_spec.json
|-- mas_spec.json           # current generated MAS spec
|-- objective.py            # objective(x_vector) adapter for QuixBugs
|-- run_BO.py               # runs FLIWBO on the QuixBugs MAS design space
|-- mas_builder.py          # builds runnable agents from mas_spec.json
|-- run_agents.py           # runs the MAS over QuixBugs files
|-- run_evaluation.py       # tests the edited files and counts resolved bugs
`-- initial_x_points.csv    # seed observations for BO
```

Generated example folders are ignored by git:

```text
examples/quixbugs/evaluation_repos/QuixBugs/
examples/quixbugs/workdirs/
examples/quixbugs/outputs/
examples/quixbugs/Results/
examples/quixbugs/BO metadata/
examples/quixbugs/noise_estimates/
```

## The Big Picture

The whole workflow is:

```text
1. Define a finite design space.
2. Give FLIWBO a few starting vectors and scores.
3. FLIWBO proposes the next vector.
4. Your code evaluates that vector.
5. Your code returns one scalar score.
6. FLIWBO learns and proposes again.
```

That is the whole contract. Everything else is plumbing.
