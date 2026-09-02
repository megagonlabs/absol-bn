"""Parameter fitting and inference backends shared by structure learners."""

from __future__ import annotations

import gc
import logging
import multiprocessing
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


ProbabilityDistribution = Dict[Any, float]
PredictionQuery = Tuple[Mapping[str, Any], str]


_worker_backend = None

def _init_propagation_worker(backend):
    global _worker_backend
    _worker_backend = backend

def _predict_target_with_propagation_worker(args):
    context_nodes, target_node, max_hops, num_samples = args
    return _worker_backend.predict_target_with_propagation(
        context_nodes, target_node, max_hops=max_hops, num_samples=num_samples
    )


class ParametersNotFittedError(RuntimeError):
    """Raised when a predictive query is made after structure-only fitting."""


def _distribution(result: Any, target_node: str) -> ProbabilityDistribution:
    states = list(result.state_names[target_node])
    return {state: float(probability) for state, probability in zip(states, result.values)}


class GlobalInferenceBackend:
    """Fit one pgmpy network over the complete learned DAG."""

    def __init__(self) -> None:
        self.fitted_model: Optional[DiscreteBayesianNetwork] = None
        self.inference: Optional[VariableElimination] = None

    def fit(self, graph: nx.DiGraph, df: pd.DataFrame, **_: Any) -> None:
        train_df = df[list(graph.nodes())].astype(str)
        self.fitted_model = DiscreteBayesianNetwork(list(graph.edges()))
        self.fitted_model.add_nodes_from(graph.nodes())
        self.fitted_model.fit(train_df, estimator=MaximumLikelihoodEstimator)
        self.fitted_model.check_model()
        self.inference = VariableElimination(self.fitted_model)

    def predict_target(
        self, context_nodes: Mapping[str, Any], target_node: str
    ) -> Optional[ProbabilityDistribution]:
        assert self.fitted_model is not None and self.inference is not None
        if target_node not in self.fitted_model.nodes():
            return None
        evidence = {
            key: str(value)
            for key, value in context_nodes.items()
            if key in self.fitted_model.nodes() and key != target_node
        }
        return _distribution(
            self.inference.query(variables=[target_node], evidence=evidence), target_node
        )

    def predict_df(self, df: pd.DataFrame, **_: Any) -> Dict[Any, Dict[str, Dict[str, Any]]]:
        assert self.fitted_model is not None and self.inference is not None
        results: Dict[Any, Dict[str, Dict[str, Any]]] = {}
        nodes = list(self.fitted_model.nodes())
        for index, row in df.iterrows():
            context = row.to_dict()
            node_results: Dict[str, Dict[str, Any]] = {}
            for node in nodes:
                try:
                    prior = _distribution(
                        self.inference.query(variables=[node], evidence={}), node
                    )
                    evidence = {
                        name: str(context[name])
                        for name in nodes
                        if name in context and name != node and pd.notna(context[name])
                    }
                    joint = _distribution(
                        self.inference.query(variables=[node], evidence=evidence), node
                    )
                    node_results[node] = {
                        "prior": prior,
                        "joint": joint,
                        **({"label": context[node]} if node in context else {}),
                    }
                except Exception as error:
                    logging.warning("Inference failed for node %s: %s", node, error)
                    node_results[node] = {
                        "prior": None, "joint": None, "label": context.get(node)
                    }
            results[index] = node_results
        return results


class MarkovBlanketInferenceBackend:
    """Fit one local pgmpy network per node's Markov blanket."""

    def __init__(self, max_nodes: Optional[int] = None, verbose: bool = False) -> None:
        self.max_nodes = max_nodes
        self.verbose = verbose
        self.graph: Optional[nx.DiGraph] = None
        self.node_markov_blankets: Dict[str, Tuple[nx.DiGraph, VariableElimination]] = {}
        self._descendants_cache: Optional[Dict[str, set]] = None

    def _get_markov_blanket(self, node: str) -> nx.DiGraph:
        assert self.graph is not None
        if self.max_nodes is None:
            parents = set(self.graph.predecessors(node))
            children = set(self.graph.successors(node))
            spouses = {parent for child in children for parent in self.graph.predecessors(child)}
            nodes = {node} | parents | children | (spouses - {node})
        else:
            queue = [node]
            nodes = set()
            while queue and len(nodes) < self.max_nodes:
                current = queue.pop(0)
                if current in nodes:
                    continue
                nodes.add(current)
                queue.extend(self.graph.predecessors(current))
                queue.extend(self.graph.successors(current))
        return self.graph.subgraph(nodes).copy()

    def fit(
        self, graph: nx.DiGraph, df: pd.DataFrame, *, max_workers: int = 1, **_: Any
    ) -> None:
        self.graph = graph.copy()

        def fit_node(node: str) -> Tuple[str, Tuple[nx.DiGraph, VariableElimination]]:
            blanket = self._get_markov_blanket(node)
            model = DiscreteBayesianNetwork(list(blanket.edges()))
            model.add_nodes_from(blanket.nodes())
            model.fit(
                df[list(blanket.nodes())].astype(str),
                estimator=MaximumLikelihoodEstimator,
            )
            model.check_model()
            if self.verbose:
                logging.info("Nodes in Markov blanket for %s: %s", node, list(blanket.nodes()))
            return node, (blanket, VariableElimination(model))

        nodes = list(graph.nodes())
        if max_workers == 1:
            fitted = map(fit_node, nodes)
        else:
            executor = ThreadPoolExecutor(max_workers=max_workers)
            fitted = executor.map(fit_node, nodes)
        try:
            self.node_markov_blankets = dict(fitted)
        finally:
            if max_workers != 1:
                executor.shutdown()

    def predict_target(
        self, context_nodes: Mapping[str, Any], target_node: str
    ) -> Optional[ProbabilityDistribution]:
        if target_node not in self.node_markov_blankets:
            return None
        blanket, inference = self.node_markov_blankets[target_node]
        evidence = {
            key: str(value)
            for key, value in context_nodes.items()
            if key in blanket.nodes() and key != target_node
        }
        return _distribution(
            inference.query(variables=[target_node], evidence=evidence), target_node
        )

    def predict_df(self, df: pd.DataFrame, **_: Any) -> Dict[Any, Dict[str, Dict[str, Any]]]:
        results: Dict[Any, Dict[str, Dict[str, Any]]] = {}
        for index, row in df.iterrows():
            context = row.to_dict()
            node_results: Dict[str, Dict[str, Any]] = {}
            for node, (blanket, inference) in self.node_markov_blankets.items():
                try:
                    prior = _distribution(inference.query(variables=[node], evidence={}), node)
                    evidence = {
                        name: str(context[name])
                        for name in blanket.nodes()
                        if name in context and name != node and pd.notna(context[name])
                    }
                    joint = _distribution(
                        inference.query(variables=[node], evidence=evidence), node
                    )
                    node_results[node] = {
                        "prior": prior,
                        "joint": joint,
                        **({"label": context[node]} if node in context else {}),
                    }
                except Exception as error:
                    logging.warning("Inference failed for node %s: %s", node, error)
                    node_results[node] = {
                        "prior": None, "joint": None, "label": context.get(node)
                    }
            results[index] = node_results
        return results

    def predict_target_batch_with_propagation(
        self,
        queries: List[Tuple[Dict[str, int], str]],
        max_hops: Optional[int] = 3,
        num_samples: int = 50,
        max_workers: int = 1,
        batch_size: int = 10000,
    ) -> List[Union[float, Dict[Any, float]]]:
        """
        Run propagation-aware prediction for a batch of (context_nodes, target_node) queries in parallel.

        Processes queries in chunks of batch_size to limit peak memory usage.

        Args:
            queries: List of (context_nodes, target_node) tuples.
            max_hops: Maximum path length for propagation.
            num_samples: Number of Monte Carlo samples.
            max_workers: Number of parallel processes.
            batch_size: Number of queries to process per chunk.
        Returns:
            List of results in the same order as queries.
        """
        all_results = []
        for i in range(0, len(queries), batch_size):
            chunk = queries[i:i + batch_size]
            args = [
                (context_nodes, target_node, max_hops, num_samples)
                for context_nodes, target_node in chunk
            ]
            if max_workers == 1:
                all_results.extend(
                    self.predict_target_with_propagation(
                        a[0], a[1], max_hops=a[2], num_samples=a[3]
                    )
                    for a in args
                )
            else:
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(
                    processes=max_workers,
                    initializer=_init_propagation_worker,
                    initargs=(self,),
                ) as pool:
                    all_results.extend(pool.map(_predict_target_with_propagation_worker, args))
            del args, chunk
            gc.collect()
        return all_results

    def predict_target_with_propagation(
        self,
        context_nodes: Dict[str, int],
        target_node: str,
        max_hops: Optional[int] = 3,
        num_samples: int = 50,
    ) -> Union[float, Dict[Any, float]]:
        """
        Predict target_node given context evidence.

        Uses d-separation-aware path finding to route distant context nodes
        through overlapping Markov blankets to the target.

        Falls back to plain blanket-local prediction when no distant evidence exists
        or no valid propagation paths are found.

        Args:
            context_nodes: Dict mapping node names to discrete values.
            target_node: Name of the node to predict.
            max_hops: Maximum path length for propagation. None allows unlimited.
            num_samples: Number of Monte Carlo samples for integrating soft evidence.
        Returns:
            Dict mapping each possible value to its probability.
        """
        if target_node not in self.node_markov_blankets:
            return None

        markov_blanket, inference_method = self.node_markov_blankets[target_node]
        mb_nodes = set(markov_blanket.nodes)

        direct_evidence = {
            k: str(v) for k, v in context_nodes.items()
            if k in mb_nodes and k != target_node
        }
        distant_nodes = {
            k: v for k, v in context_nodes.items()
            if k not in mb_nodes and k != target_node and k in self.node_markov_blankets
        }

        if not distant_nodes:
            return self.predict_target(context_nodes, target_node)

        target_mb_set = mb_nodes - {target_node}
        observed_set = set(context_nodes.keys())

        # Precompute descendants of each node for collider activation checks
        descendants_cache = self._get_descendants_cache()

        # Collect all paths grouped by entry node
        paths_by_entry: Dict[str, List[List[str]]] = defaultdict(list)
        for dist_node in distant_nodes:
            path = self._find_dsep_aware_path(
                dist_node, target_mb_set, observed_set, descendants_cache, max_hops
            )
            if path is not None:
                entry_node = path[-1]
                paths_by_entry[entry_node].append(path)

        if not paths_by_entry:
            return self.predict_target(context_nodes, target_node)

        # Monte Carlo integration
        # query_cache: (node, frozenset(evidence.items())) -> (states, probs)
        # Caches VE results so repeated (node, evidence) combos skip re-computation.
        query_cache: Dict[Tuple[str, frozenset], Tuple[list, list]] = {}
        accumulated_probs = None
        states = None

        for _ in range(num_samples):
            sample_evidence = dict(direct_evidence)

            for entry_node, paths in paths_by_entry.items():
                if entry_node in direct_evidence:
                    continue

                # Sample all paths for this entry node and average
                entry_samples = []
                for path in paths:
                    sampled_val = self._sample_along_path(path, context_nodes, query_cache)
                    if sampled_val is not None:
                        entry_samples.append(sampled_val)

                if entry_samples:
                    # Pick uniformly among the samples from different paths
                    sample_evidence[entry_node] = np.random.choice(entry_samples)

            cache_key = (target_node, frozenset(sample_evidence.items()))
            if cache_key in query_cache:
                cached_states, cached_probs = query_cache[cache_key]
                probs = np.array(cached_probs, dtype=float)
                if states is None:
                    states = cached_states
            else:
                pred = inference_method.query(variables=[target_node], evidence=sample_evidence)
                probs = np.array(pred.values, dtype=float)
                if states is None:
                    states = list(pred.state_names[target_node])
                query_cache[cache_key] = (states, list(pred.values))

            if accumulated_probs is None:
                accumulated_probs = probs.copy()
            else:
                accumulated_probs += probs

        accumulated_probs /= num_samples
        prob_dict = {state: float(prob) for state, prob in zip(states, accumulated_probs)}

        return prob_dict

    def _get_descendants_cache(self) -> Dict[str, set]:
        """
        Compute and cache the set of descendants for every node in the global DAG.
        Used to check collider activation (a collider is active if it or any
        descendant is observed).

        Returns:
            Dict mapping each node to its set of descendants.
        """
        if self._descendants_cache is None:
            self._descendants_cache = {
                node: nx.descendants(self.graph, node) for node in self.graph.nodes
            }
        return self._descendants_cache

    def _is_d_connected_triple(
        self,
        prev_node: str,
        mid_node: str,
        next_node: str,
        observed_set: set,
        descendants_cache: Dict[str, set]
    ) -> bool:
        """
        Check whether information can flow through mid_node between prev_node
        and next_node, given the observed set, using d-separation rules.

        The three cases for a triple (prev -> mid -> next in the path, where
        arrows indicate the DAG direction, not path direction):

        Chain:    prev -> mid -> next  OR  prev <- mid <- next
                Active iff mid is NOT observed.

        Fork:     prev <- mid -> next
                Active iff mid is NOT observed.

        Collider: prev -> mid <- next
                Active iff mid or any descendant of mid IS observed.

        Args:
            prev_node: Previous node on the path.
            mid_node: Middle node being traversed.
            next_node: Next node on the path.
            observed_set: Set of all observed node names.
            descendants_cache: Precomputed descendants for collider checks.

        Returns:
            True if information can flow through this triple.
        """
        prev_is_parent = self.graph.has_edge(prev_node, mid_node)
        next_is_parent = self.graph.has_edge(next_node, mid_node)

        is_collider = prev_is_parent and next_is_parent

        if is_collider:
            # Collider: active iff mid_node or any descendant is observed
            if mid_node in observed_set:
                return True
            return bool(descendants_cache.get(mid_node, set()) & observed_set)
        else:
            # Chain or fork: active iff mid_node is NOT observed
            return mid_node not in observed_set

    def _find_dsep_aware_path(
        self,
        source: str,
        target_mb_nodes: set,
        observed_set: set,
        descendants_cache: Dict[str, set],
        max_hops: Optional[int]
    ) -> Optional[List[str]]:
        """
        BFS to find the shortest d-separation-valid path from source to any
        node in target_mb_nodes.

        At each expansion, checks that information can flow through the triple
        (prev_node, current_node, candidate_neighbor) before adding the
        neighbor to the queue.

        Args:
            source: Distant evidence node.
            target_mb_nodes: Nodes in the target's Markov blanket.
            observed_set: All currently observed nodes.
            descendants_cache: Precomputed descendant sets.
            max_hops: Maximum path length, or None for unlimited.

        Returns:
            Path from source to an entry node in target's MB, or None.
        """

        # State: (current_node, previous_node) to track triples
        # visited tracks (node, prev_node) pairs since the same node can be
        # reached via different predecessors with different d-sep validity
        queue = deque([(source, [source])])
        visited = {(source, None)}

        while queue:
            node, path = queue.popleft()

            if node in target_mb_nodes and node != source:
                return path

            if max_hops is not None and len(path) - 1 >= max_hops:
                continue

            prev_node = path[-2] if len(path) >= 2 else None

            # Expand along actual DAG edges (parents + children) so that
            # d-separation triple checks are valid — they require consecutive
            # nodes on the path to be directly connected in the global DAG.
            if node not in self.graph:
                continue

            neighbors = list(self.graph.predecessors(node)) + list(self.graph.successors(node))

            for neighbor in neighbors:
                if (neighbor, node) in visited:
                    continue

                # For the first hop (no prev_node), we only need the neighbor
                # to be in the source's blanket, which is guaranteed
                if prev_node is not None:
                    if not self._is_d_connected_triple(
                        prev_node, node, neighbor, observed_set, descendants_cache
                    ):
                        continue

                visited.add((neighbor, node))
                queue.append((neighbor, path + [neighbor]))

        return None

    def _sample_along_path(
        self,
        path: List[str],
        context_nodes: Dict[str, int],
        query_cache: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Propagate evidence along a blanket path by sampling at each intermediate node.

        At each hop, queries the intermediate node's local model using all available
        evidence within its Markov blanket (both hard context and previously sampled values),
        then samples from the resulting posterior.

        Args:
            path: List of node names from distant node to entry node in target's MB.
            context_nodes: Dict of all observed node values (hard evidence).
            query_cache: Optional dict mapping (node, frozenset(evidence.items())) to
                (states, probs) to avoid redundant VariableElimination calls.

        Returns:
            Sampled state (as string) for the entry node (path[-1]), or None on failure.
        """
        if query_cache is None:
            query_cache = {}

        # Track sampled values for intermediate nodes
        sampled = {path[0]: str(context_nodes[path[0]])}

        for i in range(1, len(path)):
            node = path[i]

            # If this node is directly observed, use hard evidence
            if node in context_nodes:
                sampled[node] = str(context_nodes[node])
                continue

            if node not in self.node_markov_blankets:
                return None

            node_mb, node_inference = self.node_markov_blankets[node]

            # Gather all available evidence within this node's MB
            evidence = {}
            for mb_node in node_mb.nodes:
                if mb_node == node:
                    continue
                if mb_node in context_nodes:
                    evidence[mb_node] = str(context_nodes[mb_node])
                elif mb_node in sampled:
                    evidence[mb_node] = sampled[mb_node]

            if not evidence:
                return None

            try:
                cache_key = (node, frozenset(evidence.items()))
                if cache_key in query_cache:
                    node_states, node_probs = query_cache[cache_key]
                else:
                    pred = node_inference.query(variables=[node], evidence=evidence)
                    node_states = list(pred.state_names[node])
                    node_probs = list(pred.values)
                    query_cache[cache_key] = (node_states, node_probs)
                sampled[node] = np.random.choice(node_states, p=node_probs)
            except Exception:
                return None

        return sampled.get(path[-1])
