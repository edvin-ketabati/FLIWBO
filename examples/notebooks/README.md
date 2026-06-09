# Visual Notebook Demos

This folder contains dry-run notebooks for understanding FLIWBO behavior.

## Install

From the repository root:

```bash
pip install -e ".[notebooks]"
```

## Run

```bash
python -m notebook examples/notebooks/01_hyperparameter_visualization.ipynb
```

## Current Notebook

```text
01_hyperparameter_visualization.ipynb
```

This notebook uses a small 1D toy objective with deliberately non-stationary
behavior. It shows how the fitted GP, selected Beta-CDF input warp, and finite
warp library change under different values of these settings:

```text
lengthscale
noise_std
warp regularization
epsilon_warp
```

The notebook does not use QuixBugs, LLM endpoints, API keys, or external
services.
