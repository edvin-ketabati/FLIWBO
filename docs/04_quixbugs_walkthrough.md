# 04. QuixBugs Walkthrough

This page explains the example integration.

The example is not the core product. It is a worked pattern:

```text
vector -> MAS spec -> runtime -> tests -> score -> optimizer
```

## What The Example Optimizes

The QuixBugs example searches over multi-agent system designs.

Each vector encodes up to five agents.

Each agent uses four values:

```text
[llm_choice, toolset_choice, prompt_choice, next_agent]
```

For five agents, the vector has twenty values:

```text
5 agents * 4 values = 20 integers
```

## Step 1: Declare The Design Space

File:

```text
examples/quixbugs/resource_statement.py
```

This file declares:

```text
LLMS                  available model names
TOOLS                 available tool names
SYSTEM_PROMPTS        available agent prompts
MAX_NUMBER_OF_AGENTS  how many agents a vector can describe
recursion limits      safety limits for runtime execution
transition limits     safety limits for agent handoffs
```

This is where you change the MAS design options.

## Step 2: Build The Encoding Map

File:

```text
examples/quixbugs/run_encoding.py
```

Output:

```text
examples/quixbugs/encoding_map.json
```

The encoding map decides which integer points to which LLM, toolset, and prompt.
It tries to put similar choices near each other so the optimizer has a smoother
space to explore.

After editing `resource_statement.py`, rebuild it:

```bash
python -m examples.quixbugs.run_encoding
```

## Step 3: Compute Choice Sizes

File:

```text
examples/quixbugs/search_space.py
```

This gives the optimizer the list of valid sizes for the 20-vector.

Example:

```python
from examples.quixbugs.search_space import get_default_choice_sizes

choice_sizes = get_default_choice_sizes()
```

## Step 4: Convert Vector To MAS Spec

File:

```text
examples/quixbugs/features_to_spec.py
```

Input:

```text
x_vector
```

Output:

```text
examples/quixbugs/mas_spec.json
```

The spec contains agents like:

```text
id
active
model
system_prompt
tools
next1
```

Only agents reachable from the inferred start agent are marked active.

## Step 5: Prepare A Fresh Workspace

File:

```text
examples/quixbugs/prep_workspace.py
```

It copies QuixBugs Python program files from:

```text
examples/quixbugs/evaluation_repos/QuixBugs/python_programs/
```

into:

```text
examples/quixbugs/workdirs/<system-name>/
```

Each evaluation gets a clean editable workspace.

## Step 6: Build And Run The MAS

Files:

```text
examples/quixbugs/mas_builder.py
examples/quixbugs/run_agents.py
examples/quixbugs/mcp_tools.py
```

`mas_builder.py` reads `mas_spec.json` and creates runnable agents.

`run_agents.py` loops over target QuixBugs files.

`mcp_tools.py` exposes file tools such as:

```text
read_file
edit_file
write_file
search_current_file
run_tests
```

Agent traces are written to:

```text
examples/quixbugs/outputs/<system-name>/agent_result.json
```

## Step 7: Evaluate The Edited Files

File:

```text
examples/quixbugs/run_evaluation.py
```

It runs canonical QuixBugs tests against the edited workspace and writes:

```text
examples/quixbugs/Results/<system-name>_quixbugs_<timestamp>.json
```

The main value used by the objective is:

```text
resolved_files
```

## Step 8: Return One Score

File:

```text
examples/quixbugs/objective.py
```

The objective does:

```text
1. features_to_spec(x_vector)
2. prep_workspace(system_name)
3. run_agent_loop(system_name)
4. run_evaluation(system_name)
5. return resolved_files - token_penalty - time_penalty
```

The optimizer sees only:

```python
score = objective(x_vector)
```

## Step 9: Run BO On The Example

File:

```text
examples/quixbugs/run_BO.py
```

Command:

```bash
python -m examples.quixbugs.run_BO
```

This reads:

```text
examples/quixbugs/initial_x_points.csv
```

and writes durable BO journals under:

```text
examples/quixbugs/BO metadata/
```

## Generated Paths

These folders are generated and ignored by git:

```text
examples/quixbugs/evaluation_repos/
examples/quixbugs/workdirs/
examples/quixbugs/outputs/
examples/quixbugs/Results/
examples/quixbugs/BO metadata/
examples/quixbugs/noise_estimates/
```

## What To Copy For A New Example

For a new benchmark or partner integration, copy the pattern, not necessarily the
files.

You usually need equivalents of:

```text
resource_statement.py
search_space.py
features_to_spec.py
objective.py
runtime builder
runtime runner
evaluator
seed observations
```

The optimizer package stays the same.
