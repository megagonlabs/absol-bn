"""Shared lifecycle for every Bayesian-network structure learner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Self, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .config import BaseModelConfig
from .inference import ParametersNotFittedError, ProbabilityDistribution


class BayesianNetwork(ABC):
    """Common configuration, fitting, inference, and persistence lifecycle."""

    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config
        self.ignore_columns = config.ignore_columns
        self.random_seed = (
            int(config.random_seed)
            if config.random_seed is not None
            else int(np.random.default_rng().integers(100000))
        )
        self.verbose = config.verbose
        self.graph: Optional[nx.DiGraph] = None
        self._inference = None

    @abstractmethod
    def _fit_structure(self, df: pd.DataFrame) -> nx.DiGraph:
        """Learn and return a graph containing every non-ignored column."""

    def fit(self, df: pd.DataFrame) -> Self:
        train_df = df.drop(
            columns=[column for column in self.ignore_columns if column in df.columns]
        )
        self.graph = self._fit_structure(train_df)
        self.graph.add_nodes_from(train_df.columns)
        self._inference = None
        if self.config.fit_parameters:
            self._fit_parameters(train_df)
        return self

    def _make_inference_backend(self):
        """Build the backend this model fits CPTs into. Override to change it."""
        from .inference import GlobalInferenceBackend

        return GlobalInferenceBackend()

    def _fit_parameters(self, df: pd.DataFrame) -> None:
        if self.graph is None:
            raise RuntimeError("Structure must be fitted before parameters")
        backend = self._make_inference_backend()
        backend.fit(self.graph, df, max_workers=self.config.max_workers)
        self._inference = backend

    @property
    def parameters_fitted(self) -> bool:
        return self._inference is not None

    def _require_inference(self):
        if self._inference is None:
            raise ParametersNotFittedError(
                "Parameters were not fitted. Set fit_parameters=True in the model config."
            )
        return self._inference

    def predict_target(
        self, context_nodes: Mapping[str, Any], target_node: str
    ) -> Optional[ProbabilityDistribution]:
        return self._require_inference().predict_target(context_nodes, target_node)

    def predict_target_batch(
        self,
        queries: Sequence[Tuple[Mapping[str, Any], str]],
        *,
        max_workers: int = 1,
        batch_size: int = 10000,
    ) -> List[Optional[ProbabilityDistribution]]:
        if max_workers <= 0 or batch_size <= 0:
            raise ValueError("max_workers and batch_size must be positive")
        self._require_inference()
        results: List[Optional[ProbabilityDistribution]] = []
        for start in range(0, len(queries), batch_size):
            chunk = queries[start:start + batch_size]
            if max_workers == 1:
                results.extend(self.predict_target(context, target) for context, target in chunk)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results.extend(executor.map(lambda query: self.predict_target(*query), chunk))
        return results

    def predict_df(
        self, df: pd.DataFrame, *, max_workers: int = 1
    ) -> Dict[Any, Dict[str, Dict[str, Any]]]:
        return self._require_inference().predict_df(df, max_workers=max_workers)

    @property
    def supports_propagation(self) -> bool:
        """Whether fitted inference can route evidence between Markov blankets."""
        return hasattr(self._inference, "predict_target_with_propagation")

    def _require_propagating_inference(self):
        backend = self._require_inference()
        if not hasattr(backend, "predict_target_with_propagation"):
            raise TypeError(
                f"{type(backend).__name__} cannot propagate evidence. Belief "
                "propagation requires the Markov-blanket inference backend."
            )
        return backend

    def predict_target_with_propagation(
        self,
        context_nodes: Mapping[str, Any],
        target_node: str,
        max_hops: Optional[int] = 3,
        num_samples: int = 50,
    ) -> Optional[ProbabilityDistribution]:
        return self._require_propagating_inference().predict_target_with_propagation(
            context_nodes, target_node, max_hops=max_hops, num_samples=num_samples
        )

    def predict_target_batch_with_propagation(
        self,
        queries: Sequence[Tuple[Mapping[str, Any], str]],
        max_hops: Optional[int] = 3,
        num_samples: int = 50,
        max_workers: int = 1,
        batch_size: int = 10000,
    ) -> List[Optional[ProbabilityDistribution]]:
        return self._require_propagating_inference().predict_target_batch_with_propagation(
            queries,
            max_hops=max_hops,
            num_samples=num_samples,
            max_workers=max_workers,
            batch_size=batch_size,
        )

    def get_all_nodes(self) -> List[str]:
        if self.graph is None:
            raise RuntimeError("The model must be fitted before its nodes are accessed.")
        return list(self.graph.nodes())

    def save(self, filepath: str | Path) -> None:
        payload = {
            "class": type(self).__name__,
            "model": self,
        }
        with open(filepath, "wb") as stream:
            pickle.dump(payload, stream)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        with open(filepath, "rb") as stream:
            payload = pickle.load(stream)
        if payload.get("class") != cls.__name__:
            raise TypeError(
                f"Saved model is {payload.get('class')}, not {cls.__name__}"
            )
        return payload["model"]
