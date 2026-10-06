"""Offline data collection for the credal hazardous road (controlled per-cell sampling).

The agents plan from a fixed dataset of observations -- no online resampling, so the epistemic
uncertainty over the hazard never resolves itself away. `collect_observations` cases the whole building
(every accessible cell observed many times) and glimpses the hazard a controlled number of times,
so a sweep over that number shows how each agent's behaviour depends on how well the danger is
observed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from driving.count_model import CountModel
from driving.gridworld import HAZARD, WALL

if TYPE_CHECKING:
    import numpy as np

    from driving.gridworld import RoadMap


def collect_observations(
    road: RoadMap,
    *,
    n_case: int,
    k_hazard: int,
    alpha: float,
    rng: np.random.Generator,
) -> CountModel:
    """Case the building thoroughly, glimpse the hazard `k_hazard` times (the swept quantity).

    Every accessible (non-hazard) cell is observed `n_case` times; with p_true=0 there those
    observations are deterministically safe, so the counts are set directly. Each hazard cell is
    observed `k_hazard` times with its true Bernoulli crash probability, drawn as one binomial
    count per cell. Sweeping `k_hazard` shows how the point-estimate agents need ever more hazard
    observations to react, while the credal agent is cautious from the first glimpse.

    Args:
        road: The road providing geometry and true crash probabilities.
        n_case: Observations of each non-hazard cell.
        k_hazard: Observations of each hazard cell (the swept quantity).
        alpha: Relative-likelihood threshold for the credal intervals.
        rng: Numpy random generator.

    Returns:
        A CountModel populated with the cased observations.
    """
    model = CountModel(road, alpha=alpha)
    hazard = road.grid == HAZARD
    free = (road.grid != WALL) & ~hazard
    model.n_safe[free] = n_case
    n_crash = rng.binomial(k_hazard, road.p_true[hazard])
    model.n_crash[hazard] = n_crash
    model.n_safe[hazard] = k_hazard - n_crash
    return model
