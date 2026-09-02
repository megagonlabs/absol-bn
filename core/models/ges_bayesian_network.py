"""Full-dataset GES/DGES Bayesian-network structure learner."""

import logging
import multiprocessing
import traceback

import networkx as nx
import pandas as pd

from ..ges_structure import learn_dag_edges_with_ges
from .bayesian_network import BayesianNetwork
from .config import GESConfig


GES_TRAIN_TIMEOUT_SECONDS = 24 * 60 * 60


def _ges_worker(queue, train_df, kwargs):
    try:
        queue.put(("ok", learn_dag_edges_with_ges(train_df, **kwargs)))
    except BaseException as error:
        queue.put(("err", repr(error), traceback.format_exc()))


def _run_ges_with_timeout(train_df, kwargs, timeout_seconds):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_ges_worker, args=(queue, train_df, kwargs))
    process.start()
    try:
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            raise TimeoutError(
                f"GES exceeded {timeout_seconds // 3600}h timeout"
            )
        try:
            result = queue.get(timeout=5)
        except Exception as error:
            raise RuntimeError(
                f"GES worker exited (code={process.exitcode}) without returning a result"
            ) from error
        if result[0] == "ok":
            return result[1]
        raise RuntimeError(f"GES worker raised in subprocess:\n{result[2]}")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


class GESBayesianNetwork(BayesianNetwork):
    """Learn a DAG with one full-dataset GES or DGES search."""

    def __init__(self, config: GESConfig) -> None:
        super().__init__(config)

    def _fit_structure(self, df: pd.DataFrame) -> nx.DiGraph:
        config = self.config
        train_df = df.astype(str)
        settings = config.ges
        logging.info(
            "Running %s on the full dataset with a %dh timeout",
            settings.structure_algorithm.upper(),
            GES_TRAIN_TIMEOUT_SECONDS // 3600,
        )
        edges = _run_ges_with_timeout(
            train_df,
            {
                "score_method": settings.score_method,
                "max_parents": config.max_parents_per_node,
                "score_params": settings.score_params,
                "lambda_value": settings.lambda_value,
                "structure_algorithm": settings.structure_algorithm,
            },
            GES_TRAIN_TIMEOUT_SECONDS,
        )
        graph = nx.DiGraph(edges)
        graph.add_nodes_from(train_df.columns)
        logging.info("%s found %d edges", settings.structure_algorithm.upper(), len(edges))
        return graph
