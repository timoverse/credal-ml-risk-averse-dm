"""Method-specific baseline machinery.

Exposes the inference-time predictor and representer classes for each baseline.
Importing this package transitively imports each submodule so their class-dispatch
registrations (representer.register, LogitDistributionPredictor.register) fire at
import time. This means downstream consumers can write 'from methods import X'
without worrying about which submodule X lives in.
"""

from __future__ import annotations

from methods.adacvar import Exp3Sampler, IndexedDataset, kdpp_marginals
from methods.credal_rl_multinomial import CredalRLMultinomialPredictor

__all__ = [
    "CredalRLMultinomialPredictor",
    "Exp3Sampler",
    "IndexedDataset",
    "kdpp_marginals",
]
