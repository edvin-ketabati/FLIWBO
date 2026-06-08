"""Default configuration constants for the FLIWBO package.

Most users should pass FLIWBOConfig instead of editing this file directly. These
values exist as shared defaults and to keep older scripts working.
"""

# Lengthscale for base kernel k_0:
LENGTHSCALE = 0.35

# Boundary margin for the BO model input domain.
# Discrete choices are mapped from their raw integer indices into [X_DOMAIN_TAU, 1 - X_DOMAIN_TAU]^D.
X_DOMAIN_TAU = 0.02

# Observation noise
OBS_NOISE = 2.21

# Raw objective normalization for the 40-file QuixBugs benchmark.
# The GP models (y_raw - Y_CENTER) / Y_SCALE; logs still store raw objective values.
Y_CENTER = 20.0
Y_SCALE = 20.0

# Maximum number of BO iterations
N_ITERS = 158

# Beta scaling
BETA_SCALING = 10

# Whether to use the unity warp prior
USE_WARP_PRIOR = True

# Epsilon for warp library construction (resolution of the grid)
EPSILON_WARP = 3.0

# Coordinate-wise warp search settings
WARP_SEARCH_SWEEPS = 1
WARP_SEARCH_N_JOBS = -1

# Limit how many QuixBugs bugs/files each objective evaluation processes.
# Set to None to run the full benchmark.
OBJECTIVE_EVALUATION_LIMIT = None

# Probabilistic reparameterization acquisition optimizer settings
PR_NUM_RESTARTS = 20
PR_NUM_STEPS = 30
PR_NUM_SAMPLES = 64
PR_LEARNING_RATE = 0.1
PR_TAU_INIT = 1.0
PR_TAU_DECAY = 0.98
PR_TAU_MIN = 0.01

# Objective function weights
# Objective = resolved_instances - TOKEN_WEIGHT * total_tokens - TIME_WEIGHT * elapsed_time
TOKEN_WEIGHT = 1e-5  # Weight for token consumption (per token)
TIME_WEIGHT = 0.00  # Weight for execution time (per second)
