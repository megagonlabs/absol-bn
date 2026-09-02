"""
Call causal-learn GES or DGES where pgmpy would run HillClimbSearch; output edges for pgmpy.

- GES: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/GES.html
- DGES: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/DGES.html

CLI: ``--structure_algorithm ges|dges`` selects the outer search; ``--score_method`` picks the
local score (alias → causal-learn ``score_func`` string). DGES rejects combinations it cannot run
(e.g. BDeu in current causal-learn builds).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from causallearn.graph.Endpoint import Endpoint
from causallearn.search.ScoreBased.DGES import dges
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.PDAG2DAG import pdag2dag

from core.compat import (
    apply_causallearn_numpy2_bic_deterministic_fix,
    apply_causallearn_numpy2_bic_fix,
)
from core.scoring import (
    DEFAULT_STRUCTURE_ALGORITHM,
    merge_parameters,
    resolve_score_alias,
    validate_score_for_structure_algorithm,
)

logger = logging.getLogger(__name__)


def _edges_from_dag(G_dag) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for edge in G_dag.get_graph_edges():
        if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            parent = edge.get_node1().get_name()
            child = edge.get_node2().get_name()
            edges.append((parent, child))
    return edges


def encode_discrete_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, int]]:
    X_list = []
    r_i_map: Dict[int, int] = {}
    for j, col in enumerate(df.columns):
        codes, _ = pd.factorize(df[col].astype(str), sort=True)
        X_list.append(codes.astype(np.float64))
        r_i_map[j] = int(codes.max()) + 1 if len(codes) else 1
    X = np.column_stack(X_list)
    return X, r_i_map


def prepare_ges_matrix(
    df: pd.DataFrame,
    discrete: bool,
) -> Tuple[np.ndarray, Optional[Dict[int, int]]]:
    if discrete:
        return encode_discrete_dataframe(df)
    cols: List[np.ndarray] = []
    for col in df.columns:
        s = df[col]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().all():
            cols.append(num.astype(np.float64).to_numpy())
        else:
            c, _ = pd.factorize(s.astype(str), sort=True)
            cols.append(c.astype(np.float64))
    X = np.column_stack(cols)
    return X, None


def learn_dag_edges_with_ges(
    df: pd.DataFrame,
    score_method: str,
    max_parents: int,
    score_params: Optional[Dict[str, Any]] = None,
    lambda_value: Optional[float] = None,
    structure_algorithm: str = DEFAULT_STRUCTURE_ALGORITHM,
) -> List[Tuple[str, str]]:
    """
    Run GES or DGES and return directed edges (parent, child) for pgmpy.

    ``score_method`` is a short alias (see ``core.scoring.SCORE_ALIASES``).
    Optional keys in ``score_params`` for DGES only (ignored by GES): ``det_threshold``,
    ``skip_exact_search``, ``exact_search_method``. ``det_epsilon`` is merged into score parameters
    for ``bic_det`` / deterministic BIC.
    """
    alg = (structure_algorithm or DEFAULT_STRUCTURE_ALGORITHM).strip().lower()
    score_func, needs_discrete = resolve_score_alias(score_method)
    validate_score_for_structure_algorithm(alg, score_func)

    if score_func == "local_score_BIC":
        apply_causallearn_numpy2_bic_fix()
    if score_func == "local_score_BIC_from_cov_deterministic":
        apply_causallearn_numpy2_bic_deterministic_fix()

    raw_extra: Dict[str, Any] = dict(score_params) if score_params else {}
    det_threshold = float(raw_extra.pop("det_threshold", 1e-5))
    skip_exact_search = bool(raw_extra.pop("skip_exact_search", True))
    exact_search_method = str(raw_extra.pop("exact_search_method", "astar"))

    parameters = merge_parameters(score_func, raw_extra if raw_extra else None, lambda_value)

    node_names = list(df.columns)
    X, _unused = prepare_ges_matrix(df, needs_discrete)

    if score_func in ("local_score_CV_multi", "local_score_marginal_multi"):
        parameters = dict(parameters)
        if "dlabel" not in parameters or not parameters["dlabel"]:
            parameters["dlabel"] = {i: i for i in range(X.shape[1])}

    if alg == "ges":
        ges_parameters = None if score_func == "local_score_BDeu" else (parameters if parameters else None)
        record = ges(
            X,
            score_func=score_func,
            maxP=max_parents,
            parameters=ges_parameters,
            node_names=node_names,
        )
    else:
        det_eps = float(parameters.get("det_epsilon", 0.01)) if parameters else 0.01
        record = dges(
            X,
            score_func=score_func,
            maxP=max_parents,
            parameters=parameters if parameters else {},
            node_names=node_names,
            det_threshold=det_threshold,
            det_epsilon=det_eps,
            skip_exact_search=skip_exact_search,
            exact_search_method=exact_search_method,
        )

    G_dag = pdag2dag(record["G"])
    edges = _edges_from_dag(G_dag)
    return edges
