"""PC-algorithm Bayesian-network structure learner."""

import logging

import networkx as nx
import pandas as pd
from pgmpy.estimators import PC

from .bayesian_network import BayesianNetwork
from .config import PCConfig


class PCBayesianNetwork(BayesianNetwork):
    """Learn a DAG with pgmpy's constraint-based PC estimator."""

    def __init__(self, config: PCConfig) -> None:
        super().__init__(config)

    def _fit_structure(self, df: pd.DataFrame) -> nx.DiGraph:
        train_df = df.astype(str)
        logging.info("Running PC on the full dataset")
        learned = PC(train_df).estimate(
            return_type="dag",
            n_jobs=self.config.max_workers,
            show_progress=True,
        )
        graph = nx.DiGraph(learned.edges())
        graph.add_nodes_from(train_df.columns)
        logging.info("PC found %d edges", graph.number_of_edges())
        return graph
