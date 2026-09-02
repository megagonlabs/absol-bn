"""
Map user-facing names to causal-learn score strings for GES / DGES.

- GES: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/GES.html
- DGES: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/DGES.html

Note: installed causal-learn wires only a subset of scores into ``dges``; BDeu is listed in docs
but ``dges(..., score_func='local_score_BDeu')`` raises in current releases — use ``ges`` for BDeu.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

# alias -> (causal-learn score_func string, use discrete integer encoding for matrix prep)
SCORE_ALIASES: Dict[str, Tuple[str, bool]] = {
    "bdeu": ("local_score_BDeu", True),
    "bic": ("local_score_BIC", False),
    # Deterministic-aware BIC (Li et al., NeurIPS 2024; DGES default; also available in GES)
    "bic_det": ("local_score_BIC_from_cov_deterministic", False),
    "cv_general": ("local_score_CV_general", False),
    "marginal_general": ("local_score_marginal_general", False),
    "cv_multi": ("local_score_CV_multi", False),
    "marginal_multi": ("local_score_marginal_multi", False),
}

STRUCTURE_ALGORITHMS: Tuple[str, ...] = ("ges", "dges")

# Score functions implemented in DGES._setup_score_func (causal-learn 0.1.4.x)
_DGES_SCORE_FUNCS: Set[str] = {
    "local_score_BIC",
    "local_score_BIC_from_cov",  # not exposed as user alias; bic uses BIC path
    "local_score_BIC_from_cov_deterministic",
    "local_score_CV_general",
    "local_score_marginal_general",
    "local_score_CV_multi",
    "local_score_marginal_multi",
}

DEFAULT_SCORE = "bdeu"
DEFAULT_STRUCTURE_ALGORITHM = "ges"


def list_structure_algorithms() -> List[str]:
    return list(STRUCTURE_ALGORITHMS)


def list_score_methods() -> List[str]:
    return sorted(SCORE_ALIASES.keys())


def resolve_score_alias(name: str) -> Tuple[str, bool]:
    key = (name or DEFAULT_SCORE).strip().lower()
    if key not in SCORE_ALIASES:
        raise ValueError(
            f"Unknown score_method {name!r}. Choose one of: {', '.join(list_score_methods())}"
        )
    return SCORE_ALIASES[key]


def resolve_score_method(name: str) -> Tuple[str, bool]:
    """Same as :func:`resolve_score_alias` (name kept for older call sites)."""
    return resolve_score_alias(name)


def validate_score_for_structure_algorithm(
    structure_algorithm: str,
    score_func: str,
) -> None:
    alg = (structure_algorithm or DEFAULT_STRUCTURE_ALGORITHM).strip().lower()
    if alg not in STRUCTURE_ALGORITHMS:
        raise ValueError(
            f"Unknown structure_algorithm {structure_algorithm!r}. "
            f"Use one of: {', '.join(STRUCTURE_ALGORITHMS)}"
        )
    if alg == "dges" and score_func not in _DGES_SCORE_FUNCS:
        raise ValueError(
            f"score_func {score_func!r} is not supported by DGES in this causal-learn build. "
            f"Use structure_algorithm=ges for discrete BDeu, or pick a DGES score such as "
            f"bic, bic_det, cv_general, marginal_general, cv_multi, marginal_multi. "
            f"See https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/DGES.html"
        )


def default_ges_parameters(score_func: str) -> Dict[str, Any]:
    if score_func == "local_score_BDeu":
        return {}
    if score_func == "local_score_CV_general":
        return {"kfold": 10, "lambda": 0.01}
    if score_func == "local_score_CV_multi":
        return {"kfold": 10, "lambda": 0.01, "dlabel": {}}
    if score_func in ("local_score_marginal_multi",):
        return {"dlabel": {}}
    if score_func == "local_score_BIC_from_cov_deterministic":
        return {"lambda_value": 0.5, "det_epsilon": 0.01}
    return {}


def merge_parameters(
    score_func: str,
    user: Optional[Dict[str, Any]],
    lambda_value: Optional[float],
) -> Dict[str, Any]:
    p = default_ges_parameters(score_func)
    if user:
        p.update(user)
    if lambda_value is not None and score_func.startswith("local_score_BIC"):
        p["lambda_value"] = lambda_value
    elif score_func.startswith("local_score_BIC") and "lambda_value" not in p:
        p["lambda_value"] = 0.5
    if score_func == "local_score_BIC_from_cov_deterministic" and "det_epsilon" not in p:
        p["det_epsilon"] = 0.01
    return p
