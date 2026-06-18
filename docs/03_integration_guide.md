# 03. Integration Guide

This page explains how to connect FLIWBO to your own system.

The optimizer does not build your system. It only proposes vectors.

Your integration does the rest.

## What You Must Provide

You need to provide:

```text
1. A bounded typed design space.
2. A way to decode vectors.
3. A runtime that can build and run the decoded design.
4. A scoring function that returns one number.
```

In code, the optimizer sees only this:

```python
def objective(x_vector) -> float:
    score = run_my_system(x_vector)
    return score
```

## Step 1: Define The Search Space

First decide what each coordinate means.

Example:

```text
x[0] = model choice
x[1] = temperature
x[2] = prompt choice
x[3] = next agent choice
```

Then declare the search space:

```python
from fliwbo_core import Continuous, Discrete, SearchSpace


search_space = SearchSpace([
    Discrete(3),           # model choice: 0, 1, or 2
    Continuous(0.0, 1.0),  # temperature
    Discrete(20),          # prompt choice: 0..19
    Discrete(6),           # next agent choice: 0..5
])
```

Discrete coordinates are stored in proposals as integers. In NumPy arrays for
mixed runs, they remain integer-valued. Continuous coordinates are floats inside
the bounds you declared.

## Step 2: Decode The Vector

Write a function that turns a vector into a useful spec.

Example:

```python
def vector_to_spec(x_vector):
    return {
        "model": MODELS[x_vector[0]],
        "temperature": x_vector[1],
        "prompt": PROMPTS[x_vector[2]],
        "next_agent": x_vector[3],
    }
```

The optimizer should not know about this mapping. Keep it in your integration
layer.

## Step 3: Build The Runtime

Use the decoded spec to build the thing you want to evaluate.

That thing might be:

```text
a multi-agent system
a prompt chain
a tool configuration
a routing graph
a model ensemble
an engineering design
```

For a MAS, this step usually means:

```text
1. choose models
2. attach tools
3. attach prompts
4. connect agents
5. run a task
```

## Step 4: Score The Result

FLIWBO needs one scalar objective value.

Example:

```python
score = solved_tasks - token_weight * tokens - time_weight * seconds
```

Higher must mean better.

If your natural metric is a loss, negate it:

```python
score = -loss
```

## Step 5: Use Ask/Tell For Expensive Runs

If your evaluation takes a long time, you might want each iteration written to disk in case of a crash or error.
In that case, use ask/tell.

```python
run = optimizer.start(X_init, y_init, run_dir="runs/customer_system_001")

proposal = run.ask()
spec = vector_to_spec(proposal.x_vector)
score = run_runtime_and_score(spec)
run.tell(proposal, score)
```

This makes every proposed vector durable before evaluation starts.

## Step 6: Plug In Endpoints And Tools

Endpoint setup belongs in your integration, not in `fliwbo_core`.

Examples of things your objective adapter may do:

```text
read API keys from the environment
start a local runtime
call LLM endpoints
bind tools
run tests
count tokens
measure latency
write integration logs
return one score
```

The optimizer only receives:

```python
score = objective(x_vector)
```

or:

```python
proposal = run.ask()
score = objective(proposal.x_vector)
run.tell(proposal, score)
```
