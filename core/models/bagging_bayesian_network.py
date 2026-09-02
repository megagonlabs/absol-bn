import pandas as pd
import networkx as nx
import gc
import multiprocessing
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from ..ges_structure import learn_dag_edges_with_ges
from ..utils import has_path
from .bayesian_network import BayesianNetwork
import pickle
import logging
logging.getLogger('pgmpy').setLevel(logging.WARNING)
from typing import Dict, List, Optional, Tuple
import copy

from .config import BaggingConfig

# Per-bootstrap in-degree cap for the sub-network search. This is independent of
# ``max_parents_per_node``, which is applied later when merging candidate edges.
SUBGRAPH_MAX_INDEGREE = 5

class BaggingBayesianNetwork(BayesianNetwork):
    def _make_inference_backend(self):
        from .inference import MarkovBlanketInferenceBackend

        return MarkovBlanketInferenceBackend(
            max_nodes=self.config.max_nodes_per_markov_blanket, verbose=self.verbose
        )

    def __init__(self, config: BaggingConfig) -> None:
        super().__init__(config)
        self.num_nodes_per_markov_blanket = config.max_nodes_per_markov_blanket
        self.node_markov_blankets = {}
        # Optional structure-learning priors populated by subclasses.
        # column_groupings: dict mapping col -> set of cols likely causally related.
        # column_upweight_factor: weight applied to a column related to any already-picked column.
        self.column_groupings: Dict[str, set] = {}
        self.column_upweight_factor: float = 1.0
        # GES / causal-learn structure-search settings.
        self._cl_score_method = config.ges.score_method
        self._cl_score_params = config.ges.score_params
        self._cl_lambda_value = config.ges.lambda_value
        self._cl_structure_algorithm = config.ges.structure_algorithm
        self._cl_sample_batch_size = config.sample_batch_size

    def _fit_structure(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Learn a global Bayesian network structure.

        Draws ``num_samples`` bootstrap sub-datasets of shape
        (``num_rows_per_sample``, ``num_columns_per_sample``) and learns a small
        Bayesian network on each. The sub-networks are merged into one global
        structure by the vote/correlation/cycle heuristics in
        ``_build_network_structure``. Every parameter is read from ``self.config``
        (a ``BaggingConfig``); see that dataclass for the full key list.

        Args:
            df: Training data, already stripped of ``ignore_columns``.

        Returns:
            The learned DAG, including any node that received no edges.
        """
        config = self.config
        self.node_markov_blankets = {}
        logging.info('Starting training.')
        if config.candidate_edges_path:
            directed_edges = self.load_candidate_edges(config.candidate_edges_path)
        else:
            directed_edges = self._get_candidate_edges(
                df, config.num_samples, config.num_rows_per_sample,
                config.num_columns_per_sample, config.max_workers
            )
        if config.save_candidate_edges_path:
            self.save_candidate_edges(config.save_candidate_edges_path, directed_edges)
        logging.info('Built all sub-networks.')
        reduced_edges = self._build_network_structure(
            directed_edges, config.max_parents_per_node, df
        )
        del directed_edges
        logging.info('Combined all edges into one network.')

        graph = nx.DiGraph(reduced_edges)
        del reduced_edges  # Free the processed edges list

        isolated_nodes = set(df.columns) - set(graph.nodes)
        graph.add_nodes_from(isolated_nodes)
        if isolated_nodes:
            logging.warning(f"{len(isolated_nodes)} isolated node(s) not found in learned graph (will use prior only): {sorted(isolated_nodes)}")
        return graph

    def _fit_parameters(self, df: pd.DataFrame) -> None:
        super()._fit_parameters(df)
        self.node_markov_blankets = self._inference.node_markov_blankets

    def _get_candidate_edges_single_sample(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[str, str]]:
        """
        Learn a single sub-network on sampled data with GES and extract its edges.

        This is the extension point for swapping in a different structure learner:
        override it to return directed edges from any algorithm, and the rest of the
        pipeline (merge, Markov blankets, CPT fitting, inference) is unchanged.

        Args:
            df: Sampled DataFrame to estimate structure.

        Returns:
            List of directed edges (parent, child).
        """
        if self.ignore_columns:
            df = df.drop(columns=[c for c in self.ignore_columns if c in df.columns])
        return learn_dag_edges_with_ges(
            df.astype(str),
            score_method=self._cl_score_method,
            max_parents=SUBGRAPH_MAX_INDEGREE,
            score_params=self._cl_score_params,
            lambda_value=self._cl_lambda_value,
            structure_algorithm=self._cl_structure_algorithm,
        )

    def _get_candidate_edges(
        self,
        df: pd.DataFrame,
        num_samples: int,
        num_rows_per_sample: int,
        num_cols_per_sample: int,
        max_workers: int,
        sample_batch_size: Optional[int] = None,
        forced_cols_per_sample: Optional[List[List[str]]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Generate bootstrap samples and gather candidate edges from each.

        Args:
            df: Full DataFrame.
            num_samples: Number of bootstrap samples.
            num_rows_per_sample: Rows per sample.
            num_cols_per_sample: Columns per sample.
            max_workers: Parallel processes.
            sample_batch_size: Number of samples to process in each batch. Defaults to
                ``self._cl_sample_batch_size`` (10), deliberately small to bound the
                causal-learn memory leak — see ``__init__``. Raising it will grow RSS
                without bound on long runs.
            forced_cols_per_sample: Optional list of length num_samples giving columns
                that must be included in each corresponding bootstrap sample. Remaining
                slots are filled by the existing column-selection logic (random or
                column_grouping).

        Returns:
            Aggregated list of directed edges from all samples.
        """
        if sample_batch_size is None:
            sample_batch_size = self._cl_sample_batch_size
        results = []

        # Initialize batches to ensure exactly num_samples total
        batches = []
        remaining_samples = num_samples
        while remaining_samples > 0:
            batch_size = min(sample_batch_size, remaining_samples)
            batches.append(batch_size)
            remaining_samples -= batch_size

        # Track co-occurrence counts for parent_frequency_cutoff
        co_occurrence_counts = defaultdict(lambda: defaultdict(int))

        # Process each batch
        sample_offset = 0
        num_rows_per_sample = min(df.shape[0], num_rows_per_sample)
        num_cols_per_sample = min(df.shape[1], num_cols_per_sample)

        use_column_grouping = bool(self.column_groupings) and self.column_upweight_factor != 1.0
        all_cols = [c for c in df.columns if c not in self.ignore_columns]

        if forced_cols_per_sample is not None and len(forced_cols_per_sample) != num_samples:
            raise ValueError(
                f"forced_cols_per_sample length {len(forced_cols_per_sample)} "
                f"does not match num_samples {num_samples}"
            )

        sample_idx = 0
        for batch_size in batches:
            args = []
            for i in range(batch_size):
                sample_seed = self.random_seed + sample_offset + i
                row_sampled = df.sample(num_rows_per_sample, random_state=sample_seed)
                forced = forced_cols_per_sample[sample_idx] if forced_cols_per_sample else None
                sample_idx += 1

                if forced:
                    forced_in_df = [c for c in forced if c in all_cols][:num_cols_per_sample]
                    if use_column_grouping:
                        chosen_cols = self._sample_columns_with_grouping(
                            all_cols, num_cols_per_sample, sample_seed,
                            seed_picked=forced_in_df,
                        )
                    else:
                        remaining_to_pick = num_cols_per_sample - len(forced_in_df)
                        if remaining_to_pick <= 0:
                            chosen_cols = list(forced_in_df)
                        else:
                            available = [c for c in all_cols if c not in forced_in_df]
                            rng = np.random.default_rng(sample_seed)
                            n_pick = min(remaining_to_pick, len(available))
                            fill_idx = rng.choice(len(available), size=n_pick, replace=False)
                            fill = [available[j] for j in fill_idx]
                            chosen_cols = list(forced_in_df) + fill
                    sampled = row_sampled[chosen_cols]
                elif use_column_grouping:
                    chosen_cols = self._sample_columns_with_grouping(
                        all_cols, num_cols_per_sample, sample_seed
                    )
                    sampled = row_sampled[chosen_cols]
                else:
                    sampled = row_sampled.sample(num_cols_per_sample, axis=1, random_state=sample_seed)
                args.append(sampled)

                # Track which columns co-occur in this sample
                cols = list(sampled.columns)
                for ci, col_a in enumerate(cols):
                    for col_b in cols[ci + 1:]:
                        co_occurrence_counts[col_a][col_b] += 1
                        co_occurrence_counts[col_b][col_a] += 1

            sample_offset += batch_size

            with multiprocessing.Pool(processes=max_workers) as pool:
                batch_results = list(tqdm(
                    pool.map(self._get_candidate_edges_single_sample, args),
                    total=len(args),
                    desc=f"Processing batch {len(results)//sample_batch_size + 1}/{len(batches)}"
                ))
                results.extend(batch_results)

            # Clean up memory after each batch
            del args
            gc.collect()

        # Convert to regular dicts so self is picklable for multiprocessing
        self._co_occurrence_counts = {k: dict(v) for k, v in co_occurrence_counts.items()}

        directed_edges = []
        for edge_list in results:
            directed_edges.extend(edge_list)
        return directed_edges

    def _sample_columns_with_grouping(
        self,
        all_cols: List[str],
        num_cols: int,
        seed: int,
        seed_picked: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Sample columns for a single bootstrap, biasing later picks toward columns
        the LLM flagged as causally related to any already-picked column.

        Weight is capped at column_upweight_factor (no compounding): a candidate
        column gets that weight if it is related to at least one already-picked
        column, else weight 1.

        Args:
            seed_picked: Optional list of columns to start with already picked.
                These count toward num_cols and seed the related_to_picked set.
        """
        rng = np.random.default_rng(seed)
        num_cols = min(num_cols, len(all_cols))

        seed_picked = list(seed_picked or [])
        seed_picked = [c for c in seed_picked if c in all_cols][:num_cols]
        remaining = [c for c in all_cols if c not in seed_picked]
        picked: List[str] = list(seed_picked)
        related_to_picked: set = set()
        for c in picked:
            for neighbor in self.column_groupings.get(c, set()):
                if neighbor in remaining:
                    related_to_picked.add(neighbor)

        while len(picked) < num_cols and remaining:
            weights = np.array([
                self.column_upweight_factor if c in related_to_picked else 1.0
                for c in remaining
            ])
            probs = weights / weights.sum()
            idx = rng.choice(len(remaining), p=probs)
            chosen = remaining.pop(idx)
            picked.append(chosen)
            for neighbor in self.column_groupings.get(chosen, set()):
                if neighbor in remaining:
                    related_to_picked.add(neighbor)
        return picked

    def _apply_parent_frequency_cutoff(self, parent_counts):
        """
        Remove parent candidates that appear as parents less than parent_frequency_cutoff
        fraction of the time both nodes are present in a sample.

        Args:
            parent_counts: Nested dict mapping child -> parent -> frequency count.

        Returns:
            Modified parent_counts with low-frequency parents zeroed out.
        """
        if not self.config.parent_frequency_cutoff:
            return parent_counts
        parent_counts = copy.deepcopy(parent_counts)
        for child in list(parent_counts):
            for parent in list(parent_counts[child]):
                co_count = self._co_occurrence_counts.get(child, {}).get(parent, 0)
                if co_count == 0:
                    parent_counts[child][parent] = 0
                elif parent_counts[child][parent] / co_count < self.config.parent_frequency_cutoff:
                    parent_counts[child][parent] = 0
        return parent_counts

    def _get_edge_directions(self, parent_counts):
        """
        Resolve bidirectional edges by keeping only the direction with higher frequency.

        Args:
            parent_counts: Nested dict mapping child -> parent -> frequency count.

        Returns:
            Modified parent_counts with bidirectional conflicts resolved.
        """
        parent_counts = copy.deepcopy(parent_counts)
        for child in list(parent_counts):
            for parent in list(parent_counts[child]):
                if parent_counts[child][parent] > parent_counts[parent][child]:
                    parent_counts[parent][child] = 0
                else:
                    parent_counts[child][parent] = 0
        return parent_counts

    def _get_parent_ordering(self, parent_counts, df):
        """
        Sort candidate parents for each node by correlation strength with the child.

        Args:
            parent_counts: Nested dict mapping child -> parent -> frequency count.
            df: Full DataFrame for computing correlations.

        Returns:
            Dict mapping each child node to its ordered list of candidate parents,
            sorted by decreasing correlation (or frequency as fallback).
        """
        try:
            corr_df = df.corr()
            sorted_parents: Dict[str, List[str]] = defaultdict(list)
            for child in parent_counts:
                pairs = [
                    (p, corr_df[child][p])
                    for p, count in parent_counts[child].items() if count != 0
                ]
                sorted_pairs = sorted(pairs, key=lambda x: (-x[1], x[0]))
                sorted_parents[child] = [p for p, _ in sorted_pairs]

            return sorted_parents
        except:
            # Fallback, if df.corr fails
            # Use number of times the node appeared as a parent instead
            sorted_parents = defaultdict(list)
            for child in parent_counts:
                pairs = [
                    (p, count)
                    for p, count in parent_counts[child].items() if count != 0
                ]
                sorted_pairs = sorted(pairs, key=lambda x: (-x[1], x[0]))
                sorted_parents[child] = [p for p, _ in sorted_pairs]

            return sorted_parents

    def _get_node_addition_ordering(self, df, parent_counts):
        """
        Determine the order in which nodes should be added to the DAG during construction.
        Sorts nodes by how often they appear as a parent across bootstrap samples — nodes
        that appear most frequently as parents are added first, as a proxy for being
        causally upstream.

        Args:
            df: Full DataFrame (used to enumerate valid columns).
            parent_counts: Nested dict mapping child -> parent -> frequency count.

        Returns:
            List of node names in the order they should be processed during DAG construction.
        """
        # Sum up how many times each node appeared as a parent across all children
        parent_appearance: Dict[str, int] = defaultdict(int)
        for child, parents in parent_counts.items():
            for parent, count in parents.items():
                parent_appearance[parent] += count

        node_counts = sorted(
            [(c, parent_appearance[c]) for c in df.columns if c not in self.ignore_columns],
            key=lambda x: (-x[1], x[0])
        )
        return [node for node, _ in node_counts]

    def _build_network_structure(
        self,
        edges: List[Tuple[str, str]],
        max_parents: int,
        df: pd.DataFrame
    ) -> List[Tuple[str, str]]:
        """
        Prune and combine candidate edges into a single DAG using heuristics:
        frequency, correlation, and cycle prevention.

        Args:
            edges: List of directed edges from bootstrap samples.
            max_parents: Maximum parents allowed per node.
            df: Full DataFrame for correlation calculations.

        Returns:
            Processed list of directed edges for final DAG.
        """
        # Step 1: Initialize DAG
        dag_parents: Dict[str, List[str]] = defaultdict(list)
        dag_children: Dict[str, List[str]] = defaultdict(list)

        # Step 2: Count parent frequencies for each node
        # parent_counts = {child_node: {parent_candidate: num_parent_appearances}}
        parent_counts = defaultdict(lambda: defaultdict(int))
        nodes: set = set()
        for parent, child in edges:
            parent_counts[child][parent] += 1
            nodes.add(parent)
            nodes.add(child)

        # Step 3: Apply parent frequency cutoff
        parent_counts = self._apply_parent_frequency_cutoff(parent_counts)

        # Step 4: Remove bidirectional edges
        parent_counts = self._get_edge_directions(parent_counts)

        # Step 5: Sort candidate parents for each node by correlation in df
        sorted_parents = self._get_parent_ordering(parent_counts, df)

        # Step 6: Sort nodes in order to add to graph
        node_ordering = self._get_node_addition_ordering(df, parent_counts)

        # Step 7: Greedily add parent->child edges in order, as long as they don't create cycles
        for child in node_ordering:
            for parent in sorted_parents.get(child, []):
                if len(dag_parents[child]) >= max_parents:
                    break
                if not has_path(child, parent, dag_children):
                    dag_parents[child].append(parent)
                    dag_children[parent].append(child)
        processed_edges: List[Tuple[str, str]] = []
        for child, parents in dag_parents.items():
            for parent in parents:
                processed_edges.append((parent, child))
        return processed_edges

    def save_candidate_edges(self, filepath: str, directed_edges: List[Tuple[str, str]]) -> None:
        """
        Save candidate edges and co-occurrence counts to a pickle file for reuse.

        Args:
            filepath: Path to save the candidate edges data.
            directed_edges: List of directed edges from bootstrap samples.
        """
        data = {
            'directed_edges': directed_edges,
            'co_occurrence_counts': self._co_occurrence_counts,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Candidate edges saved to: {filepath}")

    def load_candidate_edges(self, filepath: str) -> List[Tuple[str, str]]:
        """
        Load candidate edges and co-occurrence counts from a pickle file.

        Args:
            filepath: Path to the saved candidate edges data.

        Returns:
            List of directed edges from bootstrap samples.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self._co_occurrence_counts = data['co_occurrence_counts']
        logging.info(f"Candidate edges loaded from: {filepath}")
        return data['directed_edges']
