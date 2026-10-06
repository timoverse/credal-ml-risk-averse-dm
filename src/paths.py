"""Filesystem paths used throughout the project."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment variable that relocates the dataset root, for machines where the datasets do not
# live under the home dir (the cluster stages them on /data instead).
DATA_PATH_ENV_VAR = "DATA_PATH"

# Datasets live under the user's home dir (not the project tree) unless $DATA_PATH says otherwise.
DATA_PATH = Path(os.environ.get(DATA_PATH_ENV_VAR, "~/datasets")).expanduser()

# Result/plot/logits dirs live next to the project root.
RESULTS_PATH = _PROJECT_ROOT / "results"
PLOTS_PATH = _PROJECT_ROOT / "plots"
LOGITS_PATH = _PROJECT_ROOT / "logits"
CACHE_PATH = _PROJECT_ROOT / "cache"

# Cluster-side checkpoint scratch dir. Exists on the training nodes and nowhere else.
CLUSTER_CHECKPOINTS_PATH = Path("/home/scratch/likelihood-ensembles/checkpoints")

# Project-local fallback, used when the cluster scratch dir is absent (i.e. on a laptop).
LOCAL_CHECKPOINTS_PATH = _PROJECT_ROOT / "checkpoints"

# Environment variable that overrides the choice between the two.
CHECKPOINTS_PATH_ENV_VAR = "CHECKPOINTS_PATH"


def _resolve_checkpoints_path() -> Path:
    """Pick the checkpoint directory for this machine.

    Resolution order, most explicit first:

    1. ``$CHECKPOINTS_PATH``, so a run can be pointed anywhere without editing code. The directory
       need not exist yet; train.py creates it on first write.
    2. :data:`CLUSTER_CHECKPOINTS_PATH`, if it exists. This keeps the cluster behaviour byte for
       byte what it was before the local path was introduced.
    3. :data:`LOCAL_CHECKPOINTS_PATH`, the ``checkpoints/`` directory in the project tree.

    Existence, not a hostname or an env sniff, is what distinguishes the cluster from a laptop:
    the scratch dir is mounted there and cannot be mistaken for anything local.

    Returns:
        The directory training writes ``.pt`` checkpoints into and local artifact loads read from.
    """
    override = os.environ.get(CHECKPOINTS_PATH_ENV_VAR)
    if override:
        return Path(override).expanduser()
    if CLUSTER_CHECKPOINTS_PATH.exists():
        return CLUSTER_CHECKPOINTS_PATH
    return LOCAL_CHECKPOINTS_PATH


CHECKPOINTS_PATH = _resolve_checkpoints_path()
