#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bradley–Terry ranking from pairwise preferences.

Given pairwise outcomes among models, estimate latent strengths via MLE with a logistic link.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import math
import numpy as np
from scipy.optimize import minimize


def fit_bradley_terry(
    model_ids: List[str],
    comparisons: List[Tuple[int, int, int]],
    l2_reg: float = 1e-3,
) -> Dict[str, float]:
    """
    Fit Bradley–Terry strengths.

    Args:
        model_ids: names of models (length M)
        comparisons: list of (i, j, outcome) where i, j are indices into model_ids and
                     outcome ∈ {1, -1} meaning i beats j if 1 else j beats i.
        l2_reg: L2 regularization strength to stabilize identifiability.

    Returns:
        dict model_id -> strength (mean-centered)
    """
    M = len(model_ids)
    if M == 0:
        return {}
    if not comparisons:
        return {mid: 0.0 for mid in model_ids}

    # Build counts
    wins = np.zeros((M, M), dtype=float)
    for i, j, outcome in comparisons:
        if outcome == 1:
            wins[i, j] += 1.0
        elif outcome == -1:
            wins[j, i] += 1.0

    # Initial strengths
    x0 = np.zeros(M, dtype=float)

    def neg_log_likelihood(x: np.ndarray) -> float:
        # Penalize large magnitudes slightly for identifiability and stability
        reg = 0.5 * l2_reg * float(np.dot(x, x))
        total = reg
        # For each pair
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                w_ij = wins[i, j]
                w_ji = wins[j, i]
                if w_ij == 0.0 and w_ji == 0.0:
                    continue
                # Probability i beats j under strengths x
                # p = exp(x_i) / (exp(x_i) + exp(x_j))
                xi = x[i]
                xj = x[j]
                # Use stable log-sum-exp
                m = max(xi, xj)
                denom = math.log(math.exp(xi - m) + math.exp(xj - m)) + m
                log_p_ij = xi - denom
                log_p_ji = xj - denom
                total -= w_ij * log_p_ij
                total -= w_ji * log_p_ji
        return total

    res = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
    x = res.x if res.success else x0
    # Mean-center
    x = x - np.mean(x)
    return {model_ids[i]: float(x[i]) for i in range(M)}



