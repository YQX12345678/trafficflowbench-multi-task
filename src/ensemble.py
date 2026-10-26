from __future__ import annotations
import numpy as np
import pandas as pd

def rank_ensemble(all_preds, weights=None):
    """Rank ensemble: rank each model independently, then weighted average."""
    n_models = len(all_preds)
    if weights is None:
        weights = np.ones(n_models) / n_models
    n_studies, n_targets = all_preds[0].shape
    rank_sum = np.zeros((n_studies, n_targets))
    for i, pred in enumerate(all_preds):
        for j in range(n_targets):
            rank_sum[:, j] += weights[i] * pd.Series(pred[:, j]).rank(pct=True).values
    final = np.zeros_like(rank_sum)
    for j in range(n_targets):
        final[:, j] = pd.Series(rank_sum[:, j]).rank(pct=True).values
    return final
