"""Pick a late, high-sensitivity layer for the sign-polish proof of concept.

Late blocks are where the measured (local Gauss-Newton) sensitivity under-values damage, which is why the pipeline
keeps a depth prior (``docs/learnings.md`` section 2). The picker scores every layer of the last third of blocks by
its depth-adjusted predicted loss at its allocated rank and drops the occasional inflated-Fisher outlier.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from ..utils.utils import depth_multipliers, predicted_loss


class LayerScore(BaseModel):
    """Score of one ``"<block>.<name>"`` layer.

    ``level`` is the depth-adjusted curve level ``exp(a) m_l`` (predicted loss at rank 1); ``score`` is the
    depth-adjusted predicted loss at the allocated ``rank``.
    """

    key: str
    block: int
    name: str
    rank: int
    level: float
    score: float


def rank_late_layers(curves: dict[str, tuple[float, float]], ranks: dict[str, int], n_blocks: int, depth_ramp: float,
                     late_fraction: float = 1 / 3, outlier_factor: float = 20.0) -> list[LayerScore]:
    """Rank the layers of the last ``late_fraction`` of blocks by depth-adjusted predicted loss.

    Parameters
    ----------
    curves : dict
        ``"<block>.<name>" -> (a, beta)`` fitted sensitivity curves (the rank-probe artifact's ``"curves"``).
    ranks : dict
        ``"<block>.<name>" -> rank`` allocated by the pipeline. Layers without a rank or a curve are skipped.
    n_blocks : int
        Number of decoder blocks.
    depth_ramp : float
        The run's ``rank_depth_ramp`` (log-ratio last/first block multiplier).
    late_fraction : float
        Fraction of blocks (from the end) considered.
    outlier_factor : float
        Layers whose level exceeds this multiple of the window's median level are dropped (inflated Fisher).

    Returns
    -------
    list of LayerScore
        Sorted by ``score`` descending.
    """
    first_late = n_blocks - int(n_blocks * late_fraction)
    keys = [k for k in curves if k in ranks and int(k.split(".", 1)[0]) >= first_late]
    if not keys:
        return []
    mult = depth_multipliers({k: (0, 0) for k in keys}, n_blocks, depth_ramp)
    scores: list[LayerScore] = []
    for k in keys:
        a, beta = curves[k]
        blk, name = k.split(".", 1)
        level = math.exp(a) * mult[k]
        scores.append(LayerScore(key=k, block=int(blk), name=name, rank=int(ranks[k]), level=level,
                                 score=predicted_loss((a + math.log(mult[k]), beta), int(ranks[k]))))
    median = float(np.median([s.level for s in scores]))
    kept = [s for s in scores if s.level <= outlier_factor * median]
    return sorted(kept, key=lambda s: s.score, reverse=True)
