# 03. Integration Guide

This page explains how to connect FLIWBO to your own system.

The optimizer does not build your system. It only proposes vectors.

Your integration does the rest.

## What You Must Provide

You need to provide:

```text
1. A finite design space.
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

## Step 1: Define Choices

First decide what each integer means.

Example:

```text
x[0] = model choice
x[1] = toolset choice
x[2] = prompt choice
x[3] = next agent choice
```

Then decide how many choices each coordinate has:

```python
choice_sizes = [3, 8, 20, 6]
```

This means:

```text
x[0] is in 0..2
x[1] is in 0..7
x[2] is in 0..19
x[3] is in 0..5
```

## Step 2: Decode The Vector

Write a function that turns a vector into a useful spec.

Example:

```python
def vector_to_spec(x_vector):
    return {
        "model": MODELS[x_vector[0]],
        "tools": TOOLSETS[x_vector[1]],
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

Good places for endpoint code:

```text
your_project/runtime.py
your_project/build_system.py
your_project/objective.py
```

Things your objective adapter may do:

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

The optimizer should still only receive:

```python
score = objective(x_vector)
```

or:

```python
proposal = run.ask()
score = objective(proposal.x_vector)
run.tell(proposal, score)
```

## Step 7: Start With A Dry Run

Before using real endpoints, make a fake objective:

```python
def dry_objective(x_vector) -> float:
    return float(sum(x_vector))
```

Then verify:

```text
the optimizer runs
the run directory is written
ask/tell works
resume works
your vector bounds are correct
```

Only after that should you connect expensive runtimes.

## Integration Checklist

```text
[ ] I know what each vector coordinate means.
[ ] I know each coordinate's number of choices.
[ ] I have seed vectors and seed scores.
[ ] I can decode one vector into one runnable design.
[ ] I can evaluate one design and return one number.
[ ] I can run ask/tell with a durable run_dir.
[ ] I can resume a pending proposal after a simulated crash.
```
