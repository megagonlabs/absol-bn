"""
Runtime fixes for causal-learn with NumPy 2.x.

GES uses local_score_BIC_from_cov, where yX @ XX_inv @ yX.T can be a (1, 1) array;
float(...) then raises TypeError. See causallearn/score/LocalScoreFunction.py.
"""

from __future__ import annotations

import numpy as np

_PATCHED = False


def apply_causallearn_numpy2_bic_fix() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import causallearn.score.LocalScoreFunction as lsf

    orig = lsf.local_score_BIC_from_cov

    def patched(
        Data,
        i: int,
        PAi: list,
        parameters=None,
    ):
        cov, n = Data
        if parameters is None:
            lambda_value = 0.5
        else:
            lambda_value = parameters["lambda_value"]
        sigma = cov[i, i]
        if len(PAi) > 0:
            yX = cov[np.ix_([i], PAi)]
            XX = cov[np.ix_(PAi, PAi)]
            try:
                XX_inv = np.linalg.inv(XX)
            except np.linalg.LinAlgError:
                XX_inv = np.linalg.pinv(XX)
            resid = cov[i, i] - yX @ XX_inv @ yX.T
            sigma = float(np.asarray(resid).ravel()[0])
        else:
            sigma = float(np.asarray(sigma).ravel()[0])
        if sigma <= 0:
            sigma = np.finfo(float).eps
        likelihood = -0.5 * n * (1 + np.log(sigma))
        penalty = lambda_value * (len(PAi) + 1) * np.log(n)
        return likelihood - penalty

    lsf.local_score_BIC_from_cov = patched
    _PATCHED = True


_DET_PATCHED = False


def apply_causallearn_numpy2_bic_deterministic_fix() -> None:
    """Patch DGES's deterministic BIC local score (same (1,1) residual array issue as BIC)."""
    global _DET_PATCHED
    if _DET_PATCHED:
        return
    import causallearn.search.ScoreBased.DGES as dges_mod

    def patched_det(Data, i: int, PAi: list, parameters=None):
        cov, n = Data
        if parameters is None:
            parameters = {}
        lambda_value = parameters.get("lambda_value", 0.5)
        det_epsilon = parameters.get("det_epsilon", 0.01)
        sigma = cov[i, i]
        if len(PAi) > 0:
            yX = cov[np.ix_([i], PAi)]
            XX = cov[np.ix_(PAi, PAi)]
            try:
                XX_inv = np.linalg.inv(XX)
            except np.linalg.LinAlgError:
                XX_inv = np.linalg.pinv(XX)
            resid = cov[i, i] - yX @ XX_inv @ yX.T
            sigma = float(np.asarray(resid).ravel()[0])
        else:
            sigma = float(np.asarray(sigma).ravel()[0])
        if sigma <= 0:
            sigma = 0.0
        likelihood = -0.5 * n * (1 + np.log(sigma + det_epsilon))
        penalty = lambda_value * (len(PAi) + 1) * np.log(n)
        return likelihood - penalty

    dges_mod.local_score_BIC_from_cov_deterministic = patched_det
    _DET_PATCHED = True
