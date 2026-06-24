# 05. Hyperparameter Effects

This page explains what the main optimizer knobs do.

Use it after you understand the basic API in
[02. Core Optimizer API](02_core_optimizer_api.md).

There is also a visual notebook:

```text
examples/notebooks/01_hyperparameter_visualization.ipynb
```

The notebook uses a tiny 1D toy function with deliberately non-stationary
behavior. It shows how the GP fit, the selected warp, and the finite warp
library change as the optimizer knobs move.

## Quick Mental Model

FLIWBO does three things at each BO iteration:

```text
1. Normalize the observed typed vectors into [0, 1]^D.
2. Choose one input warp from a finite warp library.
3. Fit a GP on the warped inputs and optimize the acquisition function.
```

The hyperparameters decide how flexible this model is, how much it trusts the
data, and how hard it searches.

## Core Model Knobs

```text
lengthscale
```

The base GP kernel lengthscale controls how far influence travels in the warped
input space. Small values allow fast local changes. Large values produce a
smoother model that treats distant designs as more related.

```text
noise_std
```

This is the assumed objective noise. Small values make the GP trust observed
scores more tightly. Large values tell the GP that repeated evaluations of the
same design might vary, so the fitted curve becomes less pinned to each point.

```text
beta_scaling
```

This divides the built-in `beta_t` exploration schedule used by the UCB
acquisition. Larger values reduce the exploration bonus; smaller values increase
it. The default is `5.0`. This setting only applies to the default schedule. If
you pass a custom `beta_fn` to `FLIWBOOptimizer`, that custom function controls
the beta value.

## Warp Knobs

```text
epsilon_warp
```

This controls the resolution of the finite one-dimensional warp library. Smaller
values create a finer library with more candidate warps and more compute cost.
Larger values create a coarser, faster library.

```text
use_warp_prior
```

This turns the unity warp prior on or off. The unity warp is:

```text
alpha = 1
beta = 1
```

For a Beta-CDF warp, that means the identity transform: no stretching and no
squeezing of the input coordinate. The finite warp library always includes this
exact unity warp.

```text
warp_prior_weight
warp_prior_tau
```

These two settings define the strength of the regularization toward the unity
warp. Treat them as a pair.

The practical purpose is to keep the warp search cautious while data is still
sparse. With no regularization, the optimizer can overcommit to an extreme warp
that happens to explain a few early points. With too much regularization, the
optimizer stays close to unity and gives up the benefit of learning useful input
geometry.

In the current implementation, the prior contribution is:

```text
weighted prior = -0.5 * warp_prior_weight * distance_from_unity^2 / warp_prior_tau^2
```

So the selected warp is affected by the ratio:

```text
effective prior strength = warp_prior_weight / warp_prior_tau^2
```

That means both knobs can make the prior stronger or weaker. In most
integrations, tune the pair as one regularization setting:

```text
lower effective strength  = more freedom to choose aggressive warps
higher effective strength = stronger pull toward unity
too high                  = almost no useful warping
```

## Warp Search Knobs

```text
warp_search_sweeps
```

The warp search is coordinate-wise. A sweep means: for each input coordinate,
try the one-dimensional warp options while holding the other coordinates fixed.
More sweeps can help in higher-dimensional spaces where coordinates interact,
but they cost more GP fits.

```text
warp_search_n_jobs
```

This controls parallel warp scoring. Use `1` for simple debugging. Use `-1` to
let joblib use available workers.

## Acquisition Search Knobs

The `pr_config` object controls the probabilistic reparameterization optimizer
used to search over typed vectors. Discrete coordinates are optimized through
factorized categorical distributions; continuous coordinates are optimized
directly in the warped GP-input domain.

```text
num_restarts
```

How many independent acquisition searches to try.

```text
num_steps
```

How many optimization steps each restart gets.

```text
num_samples
```

How many relaxed discrete samples are used per step.

```text
learning_rate
```

How quickly the acquisition optimizer updates its relaxed categorical
parameters.

```text
tau_init, tau_decay, tau_min
```

These control the temperature schedule for the relaxed categorical samples.
Higher temperature explores more softly. Lower temperature pushes choices closer
to hard integer selections.

## Useful Starting Recipe

For a first dry run, keep the model conservative and cheap:

```python
from fliwbo_core import FLIWBOConfig, PROptimizerConfig


config = FLIWBOConfig(
    n_iters=5,
    lengthscale=0.35,
    noise_std=2.21,
    beta_scaling=5.0,
    use_warp_prior=True,
    warp_prior_weight=0.005,
    warp_prior_tau=0.75,
    epsilon_warp=3.0,
    warp_search_sweeps=1,
    warp_search_n_jobs=1,
    pr_config=PROptimizerConfig(
        num_restarts=3,
        num_steps=5,
        num_samples=8,
    ),
)
```

After the dry run works, increase the acquisition search budget first. Then
adjust the warp knobs if the selected warps look too aggressive or too timid.

## Common Interactions

`lengthscale` and the warp interact. The warp changes the distances the GP sees.
The lengthscale decides how much those distances matter.

`warp_prior_weight` and `warp_prior_tau` should be tuned together. In this
implementation, their combined effect on candidate ranking is governed by
`warp_prior_weight / warp_prior_tau^2`.

`epsilon_warp` and `warp_search_sweeps` both affect runtime. A smaller
`epsilon_warp` creates a denser finite library of Beta-CDF curves. More sweeps
can improve the coordinate-wise search in higher dimensions. Both increase the
number of GP fits that may be scored.

## Visual Notebook

Open the notebook from the repository root:

```bash
pip install -e ".[notebooks]"
python -m notebook examples/notebooks/01_hyperparameter_visualization.ipynb
```

The notebook is intentionally small. It does not call LLM endpoints, QuixBugs,
or external services.
