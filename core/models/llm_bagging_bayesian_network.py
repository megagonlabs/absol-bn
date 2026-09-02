import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .bagging_bayesian_network import BaggingBayesianNetwork
from .. import prompts
from ..llm_session import LLMSession
from ..utils import has_path, find_directed_paths
from .config import LLMBaggingConfig
import hashlib
import json as _json
import networkx as nx
import numpy as np
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple
import re
import ast
import logging
import traceback

class LLMBaggingBayesianNetwork(BaggingBayesianNetwork):
    # Cap for path enumeration during cycle arbitration. If more than this many
    # directed paths from child to parent already exist, skip arbitration entirely
    # (the prompt would be huge and the LLM unlikely to pick a valid hitting set).
    _cycle_arbitration_max_paths: int = 10

    def __init__(self, config: LLMBaggingConfig) -> None:
        """Initialize LLM augmentations around the shared Bagging lifecycle."""
        super().__init__(config)
        self._llm = LLMSession(
            config.llm, cache_path=config.cache_path, verbose=config.verbose
        )
        self.use_llm_parent_ordering = config.parent_ordering
        self.parent_ordering_show_counts = config.parent_ordering_show_counts
        self.use_llm_structure_refinement = config.structure_refinement
        self.structure_refinement_show_support = config.structure_refinement_show_support
        self.num_refinement_iterations = config.num_refinement_iterations
        self.use_llm_column_grouping = config.column_grouping
        self._requested_column_upweight_factor = config.column_upweight_factor
        self.use_llm_cycle_arbitration = config.cycle_arbitration
        self.use_llm_adaptive_bagging = config.adaptive_bagging
        self.num_adaptive_passes = config.num_adaptive_passes
        self.num_targeted_samples_per_node = config.num_targeted_samples_per_node
        self.uncertainty_entropy_threshold = config.uncertainty_entropy_threshold
        self.use_llm_confounder_redirection = config.confounder_redirection
        self.use_llm_orphan_repair = config.orphan_repair
        self.min_co_occurrence_for_uncertainty = config.min_co_occurrence_for_uncertainty
        self.llm_max_workers = config.llm_max_workers
        # Maps frozenset({parent, child}) -> max Bernoulli entropy (bits) of the
        # parent-frequency at the time adaptive bagging first targeted the pair.
        # Surfaced to parent-ordering as provenance metadata; downstream augs
        # can use it to defer to bootstrap evidence on AB-flagged candidates.
        self._ab_targeted_pairs: Dict[FrozenSet[str], float] = {}
        # Snapshots of co-occurrence and parent counts taken before the
        # adaptive-bagging loop. Used by parent-ordering to display a
        # pre-AB / post-AB trajectory for AB-targeted candidates so the LLM
        # can see how targeted resampling shifted the evidence. Both are
        # None when AB is disabled or before any candidate-edge run.
        self._pre_ab_co_occurrence_counts: Optional[Dict[str, Dict[str, int]]] = None
        self._pre_ab_parent_counts: Optional[Dict[str, Dict[str, int]]] = None

    @property
    def llm_model_name(self) -> str:
        return self._llm.model_name

    @property
    def training_usage_details(self) -> Dict[str, Any]:
        return self._llm.usage_details

    def _send_message(
        self,
        message: str,
        operation_type: str = "general",
        system_prompt: Optional[str] = None,
    ) -> str:
        return self._llm.send(message, operation_type, system_prompt=system_prompt)

    def _log_checkpoint(self, stage: str, edges: Iterable[Tuple[str, str]]) -> None:
        """Emit a structured checkpoint to training.log so we can diff edge
        sets across pipeline stages without dumping pickles. Stages emitted by
        the LLM-augmented pipeline (in order): bootstrap_baseline,
        after_ab_pass_<i>, after_cutoff, after_direction, after_greedy,
        after_refinement, after_confounder_redirection, after_orphan_repair.

        Grep targets:
          [CHECKPOINT stage=X]        — single summary line per stage
          [CHECKPOINT_EDGES stage=X]  — full sorted JSON edge list per stage
        """
        edge_list = sorted({(str(p), str(c)) for p, c in edges})
        edges_json = _json.dumps(edge_list, separators=(",", ":"))
        digest = hashlib.sha256(edges_json.encode("utf-8")).hexdigest()[:10]
        logging.info(
            f"[CHECKPOINT stage={stage}] |E|={len(edge_list)} hash={digest}"
        )
        logging.info(f"[CHECKPOINT_EDGES stage={stage}] {edges_json}")

    def _format_variable_descriptions(self, variables: List[str]) -> str:
        """Format variable descriptions for use in prompts."""
        return self._llm.describe(variables)

    def _compute_column_groupings_llm(self, all_nodes: List[str]) -> Dict[str, set]:
        """
        For each node, ask the LLM which other nodes are likely directly causally
        linked (as parent or child). Merge into a symmetric adjacency dict.
        """
        node_set = set(all_nodes)
        variable_descriptions = self._format_variable_descriptions(list(all_nodes))

        def query(target: str) -> Tuple[str, List[str]]:
            prompt = prompts.COLUMN_GROUPING_PROMPT.format(
                target_node=target,
                variable_descriptions=variable_descriptions,
            )
            try:
                response = self._send_message(prompt, "column_grouping")
                list_pattern = r'\[([^\[\]]*)\]'
                match = re.search(list_pattern, response)
                if not match:
                    logging.info(f"_compute_column_groupings_llm: no list in response for {target}, skipping.")
                    return target, []
                related = ast.literal_eval(match.group(0))
                return target, [
                    str(other) for other in related
                    if str(other) in node_set and str(other) != target
                ]
            except Exception as e:
                logging.info(f"_compute_column_groupings_llm error for {target}: {e}")
                traceback.print_exc()
                return target, []

        groupings: Dict[str, set] = defaultdict(set)
        with ThreadPoolExecutor(max_workers=self.llm_max_workers) as executor:
            for target, related in executor.map(query, all_nodes):
                for other in related:
                    groupings[target].add(other)
                    groupings[other].add(target)
        total_edges = sum(len(v) for v in groupings.values()) // 2
        logging.info(f"LLM column groupings: {total_edges} related pair(s) across {len(groupings)} nodes.")
        return dict(groupings)

    def _compute_node_uncertainty(
        self,
        parent_counts: Dict[str, Dict[str, int]],
        co_occurrence_counts: Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, List[Tuple[str, float, float]]]]:
        """
        Identify nodes whose incoming edges have ambiguous bootstrap frequencies.

        Returns a list of (child, [(parent, frequency, entropy_bits), ...]) for
        each child with at least one incoming edge whose Bernoulli entropy (in
        bits) exceeds self.uncertainty_entropy_threshold and whose (parent, child)
        pair co-occurred in at least self.min_co_occurrence_for_uncertainty
        bootstrap samples.

        The list is sorted by descending maximum edge entropy. Within each node,
        edges are sorted by descending entropy.
        """
        threshold = self.uncertainty_entropy_threshold
        min_co = self.min_co_occurrence_for_uncertainty
        uncertain: List[Tuple[str, List[Tuple[str, float, float]]]] = []
        for child, parents in parent_counts.items():
            edge_info: List[Tuple[str, float, float]] = []
            for parent, count in parents.items():
                if count <= 0:
                    continue
                cooc = co_occurrence_counts.get(child, {}).get(parent, 0)
                if cooc < min_co:
                    continue
                f = count / cooc
                if f <= 0 or f >= 1:
                    h = 0.0
                else:
                    h = -f * np.log2(f) - (1 - f) * np.log2(1 - f)
                if h > threshold:
                    edge_info.append((parent, f, h))
            if edge_info:
                edge_info.sort(key=lambda x: -x[2])
                uncertain.append((child, edge_info))
        uncertain.sort(key=lambda item: -item[1][0][2])
        return uncertain

    def _get_adaptive_column_suggestions_llm(
        self,
        uncertain_nodes: List[Tuple[str, List[Tuple[str, float, float]]]],
        all_cols: List[str],
    ) -> Dict[str, List[str]]:
        """
        For each uncertain node, ask the LLM which columns should be co-sampled
        to disambiguate its uncertain incoming edges.
        """
        all_cols_set = set(all_cols)

        def query(item: Tuple[str, List[Tuple[str, float, float]]]) -> Tuple[str, Optional[List[str]]]:
            node, uncertain_edges = item
            edges_str = "\n".join(
                f"  - {p}: frequency={f:.2f} (entropy={h:.2f} bits)"
                for p, f, h in uncertain_edges[:20]
            )
            excluded = {node} | {p for p, _, _ in uncertain_edges}
            available = [c for c in all_cols if c not in excluded]
            relevant = [node] + [p for p, _, _ in uncertain_edges] + available
            descs = self._format_variable_descriptions(relevant)
            prompt = prompts.ADAPTIVE_BAGGING_PROMPT.format(
                target_node=node,
                uncertain_edges=edges_str,
                variable_descriptions=descs,
            )
            try:
                response = self._send_message(prompt, "adaptive_bagging")
                list_match = re.search(r'\[([^\[\]]*)\]', response, re.DOTALL)
                if not list_match:
                    logging.info(f"Adaptive bagging: no list in response for {node}, skipping.")
                    return node, None
                cols = ast.literal_eval(list_match.group(0))
                if not isinstance(cols, list):
                    return node, None
                cols = [str(c) for c in cols if str(c) in all_cols_set and str(c) != node]
                return node, cols
            except Exception as e:
                logging.info(f"Adaptive bagging error for {node}: {e}")
                traceback.print_exc()
                return node, None

        suggestions: Dict[str, List[str]] = {}
        with ThreadPoolExecutor(max_workers=self.llm_max_workers) as executor:
            for node, cols in executor.map(query, uncertain_nodes):
                if cols is not None:
                    suggestions[node] = cols
        return suggestions

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
        Run standard bagging, then optionally run adaptive-bagging passes that
        target nodes with high-uncertainty parent sets. Edges and co-occurrence
        counts accumulate across all passes; the base class's
        _co_occurrence_counts state is set to the accumulated total before
        returning so downstream merge logic sees combined evidence.
        """
        initial_edges = super()._get_candidate_edges(
            df, num_samples, num_rows_per_sample, num_cols_per_sample,
            max_workers, sample_batch_size,
            forced_cols_per_sample=forced_cols_per_sample,
        )
        self._log_checkpoint("bootstrap_baseline", initial_edges)
        if not self.use_llm_adaptive_bagging:
            return initial_edges

        # Snapshot pre-AB state for trajectory display in parent-ordering.
        self._pre_ab_co_occurrence_counts = {
            k: dict(v) for k, v in self._co_occurrence_counts.items()
        }
        self._pre_ab_parent_counts = defaultdict(lambda: defaultdict(int))
        for parent, child in initial_edges:
            self._pre_ab_parent_counts[child][parent] += 1
        self._pre_ab_parent_counts = {
            k: dict(v) for k, v in self._pre_ab_parent_counts.items()
        }

        accumulated_edges = list(initial_edges)
        accumulated_cooc: Dict[str, Dict[str, int]] = {
            k: dict(v) for k, v in self._co_occurrence_counts.items()
        }
        all_cols = [c for c in df.columns if c not in self.ignore_columns]

        for pass_idx in range(self.num_adaptive_passes):
            parent_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for parent, child in accumulated_edges:
                parent_counts[child][parent] += 1

            uncertain = self._compute_node_uncertainty(parent_counts, accumulated_cooc)
            if not uncertain:
                logging.info(
                    f"Adaptive bagging pass {pass_idx + 1}/{self.num_adaptive_passes}: "
                    "no uncertain nodes remaining, stopping early."
                )
                break

            for child, edge_info in uncertain:
                for parent, _, entropy in edge_info:
                    pair = frozenset({parent, child})
                    prev = self._ab_targeted_pairs.get(pair, 0.0)
                    if entropy > prev:
                        self._ab_targeted_pairs[pair] = entropy

            top_preview = ", ".join(
                f"{node} (max H={edges[0][2]:.2f}, {len(edges)} uncertain edge(s))"
                for node, edges in uncertain[:10]
            )
            logging.info(
                f"Adaptive bagging pass {pass_idx + 1}/{self.num_adaptive_passes}: "
                f"{len(uncertain)} uncertain node(s). Top: {top_preview}"
            )

            col_suggestions = self._get_adaptive_column_suggestions_llm(uncertain, all_cols)

            forced_cols_per_sample_pass: List[List[str]] = []
            max_forced = max(2, num_cols_per_sample - 1)
            for node, edges in uncertain:
                suggested = col_suggestions.get(node, [])
                candidates = [p for p, _, _ in edges]
                forced = [node] + candidates
                seen = set(forced)
                for c in suggested:
                    if c not in seen:
                        forced.append(c)
                        seen.add(c)
                forced = forced[:max_forced]
                for _ in range(self.num_targeted_samples_per_node):
                    forced_cols_per_sample_pass.append(list(forced))

            if not forced_cols_per_sample_pass:
                logging.info(
                    f"Adaptive bagging pass {pass_idx + 1}: no columns to force, stopping."
                )
                break

            targeted_edges = super()._get_candidate_edges(
                df,
                len(forced_cols_per_sample_pass),
                num_rows_per_sample,
                num_cols_per_sample,
                max_workers,
                sample_batch_size,
                forced_cols_per_sample=forced_cols_per_sample_pass,
            )
            accumulated_edges.extend(targeted_edges)

            for a, neighbors in self._co_occurrence_counts.items():
                for b, count in neighbors.items():
                    accumulated_cooc.setdefault(a, {})[b] = (
                        accumulated_cooc.get(a, {}).get(b, 0) + count
                    )

            logging.info(
                f"Adaptive bagging pass {pass_idx + 1}: ran "
                f"{len(forced_cols_per_sample_pass)} targeted bootstrap(s). "
                f"Cumulative AB-targeted pairs: {len(self._ab_targeted_pairs)}."
            )
            self._log_checkpoint(f"after_ab_pass_{pass_idx + 1}", accumulated_edges)

        self._co_occurrence_counts = accumulated_cooc
        return accumulated_edges

    def _resolve_cycle_llm(
        self,
        new_parent: str,
        new_child: str,
        cycle_paths: List[List[str]],
    ) -> List[Tuple[str, str]]:
        """
        Given a proposed new edge new_parent -> new_child that would close one or
        more cycles via the directed paths in cycle_paths (each from new_child to
        new_parent, inclusive), ask the LLM which existing edges to drop so that
        every path is broken.

        Returns the list of edges to remove (possibly empty, which signals "skip
        the new edge"). Caller is responsible for verifying the removal actually
        breaks every path - the LLM's choice is not authoritative.
        """
        edge_options: List[Tuple[str, str]] = []
        seen: set = set()
        for path in cycle_paths:
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge not in seen:
                    seen.add(edge)
                    edge_options.append(edge)

        if not edge_options:
            return []

        options_str = "\n".join(
            f"{i+1}. {p} -> {c}" for i, (p, c) in enumerate(edge_options)
        )
        paths_str = "\n".join(
            f"Path {i+1}: " + " -> ".join(path) for i, path in enumerate(cycle_paths)
        )

        all_vars: set = {new_parent, new_child}
        for path in cycle_paths:
            all_vars.update(path)
        variable_descriptions = self._format_variable_descriptions(list(all_vars))

        prompt = prompts.CYCLE_ARBITRATION_PROMPT.format(
            new_edge_parent=new_parent,
            new_edge_child=new_child,
            cycle_paths=paths_str,
            edge_options=options_str,
            variable_descriptions=variable_descriptions,
        )
        try:
            response = self._send_message(prompt, "cycle_arbitration").strip()
            list_match = re.search(r'\[([^\[\]]*)\]', response)
            if not list_match:
                return []
            choices = ast.literal_eval(list_match.group(0))
            if not isinstance(choices, list):
                return []
            removals: List[Tuple[str, str]] = []
            for c in choices:
                if not isinstance(c, int) or c < 1 or c > len(edge_options):
                    continue
                removals.append(edge_options[c - 1])
            return removals
        except Exception as e:
            logging.info(f"_resolve_cycle_llm error: {e}")
            traceback.print_exc()
            return []

    def _get_parent_ordering_llm(self, parent_counts, df: pd.DataFrame):
        """
        Use the LLM to rank candidate parents for each child by causal
        plausibility. Returns a ranked list per child: the LLM's chosen
        parents first (in LLM order), then any candidates the LLM omitted
        appended at the tail sorted by descending bootstrap count.

        When adaptive bagging is enabled, each AB-targeted candidate is
        rendered with a pre-AB / post-AB trajectory so the LLM can see how
        targeted resampling shifted the evidence (vs the original bootstrap
        alone).

        Args:
            parent_counts: Nested dict mapping child -> parent -> frequency count.
            df: Full data frame used by the non-LLM correlation-based fallback.

        Returns:
            Dict mapping each child to its ranked list of candidate parents.
        """
        logging.info(f"Total count of parents to be considered: {len(parent_counts.items())}")
        pre_cooc = self._pre_ab_co_occurrence_counts
        pre_pc = self._pre_ab_parent_counts
        ab_targeted = self._ab_targeted_pairs
        show_counts = self.parent_ordering_show_counts
        fallback_parent_counts = self._apply_parent_frequency_cutoff(parent_counts)
        fallback_ordering = super()._get_parent_ordering(fallback_parent_counts, df)

        def pct(n: int, d: int) -> str:
            return f"{n/d*100:.1f}%" if d > 0 else "n/a"

        def format_candidate(parent: str, count: int, child: str) -> str:
            total_cooc = self._co_occurrence_counts.get(child, {}).get(parent, 0)
            base = (
                f"{parent}: {pct(count, total_cooc)} "
                f"({count}/{total_cooc} networks where both co-occurred)"
            )
            pair = frozenset({parent, child})
            h = ab_targeted.get(pair)
            if h is None or pre_cooc is None or pre_pc is None:
                return base
            pre_count = pre_pc.get(child, {}).get(parent, 0)
            pre_cooc_n = pre_cooc.get(child, {}).get(parent, 0)
            post_count = max(0, count - pre_count)
            post_cooc_n = max(0, total_cooc - pre_cooc_n)
            return (
                f"{base}\n"
                f"    pre-AB:  {pct(pre_count, pre_cooc_n)} "
                f"({pre_count}/{pre_cooc_n} initial bootstrap networks where both co-occurred) "
                f"— AB triggered at H={h:.2f}\n"
                f"    post-AB: {pct(post_count, post_cooc_n)} "
                f"({post_count}/{post_cooc_n} AB-targeted samples where both co-occurred)"
            )

        def query(item: Tuple[str, Dict[str, int]]) -> Tuple[str, List[str], List[str]]:
            child, parent_candidates = item
            positive = [(p, c) for p, c in parent_candidates.items() if c > 0]
            by_count = sorted(positive, key=lambda x: (-x[1], x[0]))
            try:
                parent_nodes = [child] + [p for p in parent_candidates.keys()]
                variable_descriptions = self._format_variable_descriptions(parent_nodes)

                if show_counts:
                    parent_percentages = "\n".join(
                        format_candidate(parent, count, child) for parent, count in by_count
                    )
                    prompt = prompts.PARENT_SELECTION_PROMPT.format(
                        target_node=child,
                        parent_percentages=parent_percentages,
                        variable_descriptions=variable_descriptions
                    )
                    system_prompt = (
                        prompts.PARENT_SELECTION_SYSTEM_PROMPT_WITH_AB
                        if self.use_llm_adaptive_bagging
                        else prompts.PARENT_SELECTION_SYSTEM_PROMPT
                    )
                else:
                    prompt = prompts.PARENT_SELECTION_PROMPT_NO_COUNTS.format(
                        target_node=child,
                        candidate_parents="\n".join(f"- {parent}" for parent, _ in by_count),
                        variable_descriptions=variable_descriptions
                    )
                    system_prompt = prompts.PARENT_SELECTION_SYSTEM_PROMPT_NO_COUNTS
                response = self._send_message(
                    prompt, "parent_ordering", system_prompt=system_prompt
                )

                list_pattern = r'\[([^\[\]]*)\]'
                match = re.search(list_pattern, response)
                if not match:
                    raise Exception(f"No list found in model response: {response}")
                parsed = ast.literal_eval(match.group(0))
                positive_set = {p for p, _ in positive}
                selected: List[str] = []
                selected_set: set[str] = set()
                for parent in parsed:
                    parent = str(parent)
                    if parent in positive_set and parent not in selected_set:
                        selected.append(parent)
                        selected_set.add(parent)

                # Tail = candidates the LLM did not select but whose bootstrap
                # support clears the frequency cutoff. LLM endorsement bypasses
                # the cutoff entirely (selected list goes through unfiltered).
                cutoff = self.config.parent_frequency_cutoff
                tail: List[str] = []
                for p, count in by_count:
                    if p in selected_set:
                        continue
                    if cutoff > 0:
                        cooc = self._co_occurrence_counts.get(child, {}).get(p, 0)
                        if cooc == 0 or (count / cooc) < cutoff:
                            continue
                    tail.append(p)
                return child, selected, selected + tail
            except Exception as e:
                logging.info(f"Error in _get_parent_ordering_llm for {child}, falling back to default implementation")
                logging.info(e)
                fallback = fallback_ordering.get(child, [])
                return child, fallback, fallback

        sorted_parents: Dict[str, List[str]] = {}
        total_tail_appended = 0
        children_with_tail = 0
        with ThreadPoolExecutor(max_workers=self.llm_max_workers) as executor:
            for child, selected, ordered in executor.map(query, parent_counts.items()):
                sorted_parents[child] = ordered
                tail = ordered[len(selected):]
                if tail:
                    children_with_tail += 1
                    total_tail_appended += len(tail)
                ab_in_candidates = [
                    p for p in ordered
                    if frozenset({p, child}) in ab_targeted
                ]
                logging.info(
                    f"[PO_DECISION child={child} "
                    f"selected={selected} "
                    f"tail_appended={tail} "
                    f"ab_flagged_in_candidates={ab_in_candidates}]"
                )

        n = max(1, len(parent_counts))
        logging.info(
            f"[PO_SUMMARY n_children={len(parent_counts)} "
            f"children_with_tail_appends={children_with_tail} "
            f"total_tail_appended={total_tail_appended} "
            f"avg_tail_appended_per_child={total_tail_appended/n:.2f}]"
        )
        # Log after_cutoff here for the LLM-PO path: the edges PO actually
        # passes to greedy (LLM-selected unfiltered + tail filtered by cutoff).
        self._log_checkpoint(
            "after_cutoff",
            ((parent, child) for child, ps in sorted_parents.items() for parent in ps),
        )
        return sorted_parents

    def _refine_structure(
        self,
        processed_edges: List[Tuple[str, str]],
        nodes: set,
        parent_counts: Optional[Dict[str, Dict[str, int]]] = None,
        near_miss: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Tuple[str, str]]:
        """
        Iteratively refine the network structure using LLM feedback.

        The LLM can add edges, delete edges, or terminate early. Runs for at most
        num_refinement_iterations (defaults to the number of nodes).

        Args:
            processed_edges: List of (parent, child) tuples representing current edges.
            nodes: Set of all node names in the network.
            parent_counts: Nested dict child -> parent -> bootstrap count. Used to
                annotate the prompt with bootstrap support per edge.
            near_miss: (parent, child) tuples that PO+cutoff endorsed but greedy
                did not place (max_parents saturation or cycle). Surfaced as a
                separate block in the prompt so refinement can prefer them when
                adding edges, rather than inventing from scratch.

        Returns:
            Refined list of (parent, child) edge tuples.
        """
        num_iterations = self.num_refinement_iterations if self.num_refinement_iterations is not None else len(nodes)
        parent_counts = parent_counts or {}
        near_miss = near_miss or []

        def support(parent: str, child: str) -> float:
            count = parent_counts.get(child, {}).get(parent, 0)
            cooc = self._co_occurrence_counts.get(child, {}).get(parent, 0)
            return count / cooc if cooc > 0 else 0.0

        def support_str(parent: str, child: str) -> str:
            cooc = self._co_occurrence_counts.get(child, {}).get(parent, 0)
            count = parent_counts.get(child, {}).get(parent, 0)
            if cooc == 0:
                return "no bootstrap evidence"
            return f"{count/cooc:.2f} ({count}/{cooc} co-occurring networks)"

        dag_parents = defaultdict(list)
        dag_children = defaultdict(list)
        for parent, child in processed_edges:
            dag_parents[child].append(parent)
            dag_children[parent].append(child)

        variable_descriptions = self._format_variable_descriptions(list(nodes))
        in_graph_for_prompt = set(processed_edges)
        show_support = self.structure_refinement_show_support

        near_miss_sorted = sorted(
            (e for e in near_miss if e not in in_graph_for_prompt),
            key=lambda e: (-support(*e), e),
        )
        near_miss_str = "\n".join(
            f"{p} -> {c}  (support: {support_str(p, c)})"
            for p, c in near_miss_sorted
        ) if near_miss_sorted else "(none)"

        for i in range(num_iterations):
            current_edges = [
                (parent, child)
                for child, parents in dag_parents.items()
                for parent in parents
            ]
            if current_edges:
                if show_support:
                    edges_str = "\n".join(
                        f"{p} -> {c}  (support: {support_str(p, c)})"
                        for p, c in sorted(current_edges)
                    )
                else:
                    edges_str = "\n".join(
                        f"{p} -> {c}" for p, c in sorted(current_edges)
                    )
            else:
                edges_str = "(no edges)"

            if show_support:
                prompt = prompts.STRUCTURE_REFINEMENT_PROMPT.format(
                    variable_descriptions=variable_descriptions,
                    edges=edges_str,
                    near_miss=near_miss_str,
                )
                operation_type = "structure_refinement"
            else:
                prompt = prompts.VANILLA_STRUCTURE_REFINEMENT_PROMPT.format(
                    variable_descriptions=variable_descriptions,
                    edges=edges_str,
                )
                operation_type = "vanilla_structure_refinement"

            try:
                response = self._send_message(prompt, operation_type).strip()

                if response.startswith("terminate"):
                    logging.info(f"Structure refinement terminated by LLM at iteration {i+1}/{num_iterations}")
                    break

                action_match = re.match(r'(add_edge|delete_edge)\((.+),\s*(.+)\)', response)
                if not action_match:
                    logging.info(f"Structure refinement: could not parse response '{response}', skipping iteration {i+1}")
                    continue

                action = action_match.group(1)
                node_a = action_match.group(2).strip().strip("'\"")
                node_b = action_match.group(3).strip().strip("'\"")

                if node_a not in nodes or node_b not in nodes:
                    logging.info(f"Structure refinement: unknown node(s) in '{response}', skipping iteration {i+1}")
                    continue

                if action == "add_edge":
                    parent, child = node_a, node_b
                    if child in dag_parents and parent in dag_parents[child]:
                        logging.info(f"Structure refinement: edge {parent} -> {child} already exists, skipping")
                        continue
                    if has_path(child, parent, dag_children):
                        logging.info(f"Structure refinement: adding {parent} -> {child} would create a cycle, skipping")
                        continue
                    sup = support(parent, child)
                    is_near_miss = (parent, child) in set(near_miss)
                    dag_parents[child].append(parent)
                    dag_children[parent].append(child)
                    logging.info(
                        f"[REFINE_ACTION iter={i+1}/{num_iterations} action=add_edge "
                        f"parent={parent} child={child} support={sup:.3f} "
                        f"near_miss={is_near_miss}]"
                    )

                elif action == "delete_edge":
                    parent, child = node_a, node_b
                    if child not in dag_parents or parent not in dag_parents[child]:
                        logging.info(f"Structure refinement: edge {parent} -> {child} does not exist, skipping")
                        continue
                    parent_would_be_isolated = (
                        len(dag_children[parent]) == 1 and dag_children[parent] == [child]
                        and len(dag_parents.get(parent, [])) == 0
                    )
                    child_would_be_isolated = (
                        len(dag_parents[child]) == 1 and dag_parents[child] == [parent]
                        and len(dag_children.get(child, [])) == 0
                    )
                    if parent_would_be_isolated or child_would_be_isolated:
                        isolated_node = parent if parent_would_be_isolated else child
                        logging.info(f"Structure refinement: deleting {parent} -> {child} would isolate {isolated_node}, skipping")
                        continue
                    sup = support(parent, child)
                    dag_parents[child].remove(parent)
                    dag_children[parent].remove(child)
                    logging.info(
                        f"[REFINE_ACTION iter={i+1}/{num_iterations} action=delete_edge "
                        f"parent={parent} child={child} support={sup:.3f}]"
                    )

            except Exception as e:
                logging.info(f"Structure refinement: error at iteration {i+1}: {e}")
                traceback.print_exc()
                continue

        refined_edges = []
        for child, parents in dag_parents.items():
            for parent in parents:
                refined_edges.append((parent, child))
        return refined_edges

    def _redirect_confounded_edges(
        self,
        processed_edges: List[Tuple[str, str]],
        nodes: set,
        parent_counts: Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, str]]:
        """
        For each child with incoming edges, ask the LLM to flag any edges
        where a specific node in the variable list better explains the
        correlation as a common cause. Each flagged edge requires a named
        confounder; deletes without a structurally validated confounder are
        rejected.
        """
        dag_parents: Dict[str, List[str]] = defaultdict(list)
        dag_children: Dict[str, List[str]] = defaultdict(list)
        for parent, child in processed_edges:
            dag_parents[child].append(parent)
            dag_children[parent].append(child)

        def support_str(parent: str, child: str) -> str:
            cooc = self._co_occurrence_counts.get(child, {}).get(parent, 0)
            count = parent_counts.get(child, {}).get(parent, 0)
            if cooc == 0:
                return "no bootstrap evidence"
            return f"{count/cooc:.2f} ({count}/{cooc} co-occurring networks)"

        children_with_edges = [c for c in dag_parents if dag_parents[c]]

        def query(child: str) -> List[Tuple[str, str]]:
            parents_block = "\n".join(
                f"  {p} -> {child}  (support: {support_str(p, child)})"
                for p in sorted(dag_parents[child])
            )
            variable_descriptions = self._format_variable_descriptions(list(nodes))
            prompt = prompts.CONFOUNDER_REDIRECTION_PROMPT.format(
                variable_descriptions=variable_descriptions,
                target_node=child,
                target_parents=parents_block,
            )
            try:
                response = self._send_message(prompt, "confounder_redirection").strip()
                match = re.search(r"\[[^\[\]]*(?:\([^\[\]]*\)[^\[\]]*)*\]", response, re.DOTALL)
                if not match:
                    logging.info(
                        f"Confounder redirection: no list in response for {child}, skipping. Response: {response}"
                    )
                    return []
                parsed = ast.literal_eval(match.group(0))
                if not isinstance(parsed, list):
                    return []
                flagged: List[Tuple[str, str]] = []
                for item in parsed:
                    if not (isinstance(item, (list, tuple)) and len(item) == 2):
                        continue
                    parent, confounder = str(item[0]), str(item[1])
                    if parent not in dag_parents.get(child, []):
                        logging.info(
                            f"Confounder redirection [{child}]: skip — '{parent}' is not a current parent"
                        )
                        continue
                    if confounder not in nodes:
                        logging.info(
                            f"Confounder redirection [{child}]: skip — confounder '{confounder}' is not in variable list"
                        )
                        continue
                    if confounder == parent or confounder == child:
                        logging.info(
                            f"Confounder redirection [{child}]: skip — confounder '{confounder}' coincides with parent or target"
                        )
                        continue
                    flagged.append((parent, confounder))
                return flagged
            except Exception as e:
                logging.info(f"Confounder redirection error for {child}: {e}")
                traceback.print_exc()
                return []

        per_child_flags: Dict[str, List[Tuple[str, str]]] = {}
        with ThreadPoolExecutor(max_workers=self.llm_max_workers) as executor:
            futures = {executor.submit(query, c): c for c in children_with_edges}
            for fut in as_completed(futures):
                child = futures[fut]
                per_child_flags[child] = fut.result() or []

        total_proposed = 0
        applied: List[Tuple[str, str, str]] = []  # (parent, child, confounder)
        for child in children_with_edges:
            for parent, confounder in per_child_flags.get(child, []):
                total_proposed += 1
                if parent not in dag_parents.get(child, []):
                    continue
                parent_isolated = (
                    len(dag_children.get(parent, [])) == 1
                    and dag_children[parent] == [child]
                    and len(dag_parents.get(parent, [])) == 0
                )
                child_isolated = (
                    len(dag_parents.get(child, [])) == 1
                    and dag_parents[child] == [parent]
                    and len(dag_children.get(child, [])) == 0
                )
                if parent_isolated or child_isolated:
                    iso = parent if parent_isolated else child
                    logging.info(
                        f"[CONFOUNDER_REDIRECT skipped target={child} parent={parent} "
                        f"confounder={confounder} reason=would_isolate_{iso}]"
                    )
                    continue
                cooc = self._co_occurrence_counts.get(child, {}).get(parent, 0)
                count = parent_counts.get(child, {}).get(parent, 0)
                sup = count / cooc if cooc > 0 else 0.0
                dag_parents[child].remove(parent)
                dag_children[parent].remove(child)
                applied.append((parent, child, confounder))
                logging.info(
                    f"[CONFOUNDER_REDIRECT target={child} parent={parent} "
                    f"confounder={confounder} support={sup:.3f}]"
                )

        logging.info(
            f"[CONFOUNDER_REDIRECT_SUMMARY children_reviewed={len(children_with_edges)} "
            f"flagged_by_llm={total_proposed} applied={len(applied)}]"
        )

        new_edges: List[Tuple[str, str]] = []
        for child, parents in dag_parents.items():
            for parent in parents:
                new_edges.append((parent, child))
        return new_edges

    def _repair_orphan_nodes(
        self,
        processed_edges: List[Tuple[str, str]],
        nodes: set,
        max_parents: int,
    ) -> List[Tuple[str, str]]:
        """
        Propose parents for variables that currently have no incoming edges
        in the graph. The LLM picks from the variable list; bagging+PO had
        nothing for these nodes so the LLM is the only available signal.
        """
        dag_parents: Dict[str, List[str]] = defaultdict(list)
        dag_children: Dict[str, List[str]] = defaultdict(list)
        for parent, child in processed_edges:
            dag_parents[child].append(parent)
            dag_children[parent].append(child)

        orphans = sorted(c for c in nodes if not dag_parents.get(c))
        if not orphans:
            logging.info("[ORPHAN_REPAIR_SUMMARY orphans=0 applied=0]")
            return list(processed_edges)

        def query(target: str) -> List[str]:
            variable_descriptions = self._format_variable_descriptions(list(nodes))
            prompt = prompts.ORPHAN_REPAIR_PROMPT.format(
                variable_descriptions=variable_descriptions,
                target_node=target,
                max_parents=max_parents,
            )
            try:
                response = self._send_message(prompt, "orphan_repair").strip()
                match = re.search(r"\[[^\[\]]*\]", response, re.DOTALL)
                if not match:
                    logging.info(
                        f"Orphan repair: no list in response for {target}, skipping. Response: {response}"
                    )
                    return []
                parsed = ast.literal_eval(match.group(0))
                if not isinstance(parsed, list):
                    return []
                proposed: List[str] = []
                for item in parsed:
                    name = str(item)
                    if name not in nodes:
                        continue
                    if name == target:
                        continue
                    if name not in proposed:
                        proposed.append(name)
                return proposed[:max_parents]
            except Exception as e:
                logging.info(f"Orphan repair error for {target}: {e}")
                traceback.print_exc()
                return []

        per_target_proposals: Dict[str, List[str]] = {}
        with ThreadPoolExecutor(max_workers=self.llm_max_workers) as executor:
            futures = {executor.submit(query, t): t for t in orphans}
            for fut in as_completed(futures):
                target = futures[fut]
                per_target_proposals[target] = fut.result() or []

        applied = 0
        for target in orphans:
            proposed = per_target_proposals.get(target, [])
            added: List[str] = []
            for parent in proposed:
                if len(dag_parents[target]) >= max_parents:
                    break
                if parent in dag_parents.get(target, []):
                    continue
                if has_path(target, parent, dag_children):
                    logging.info(
                        f"Orphan repair [{target}]: skip {parent} -> {target} (would create cycle)"
                    )
                    continue
                dag_parents[target].append(parent)
                dag_children[parent].append(target)
                added.append(parent)
                applied += 1
            logging.info(
                f"[ORPHAN_REPAIR target={target} proposed={proposed} applied={added}]"
            )

        logging.info(
            f"[ORPHAN_REPAIR_SUMMARY orphans={len(orphans)} "
            f"orphans_with_proposals={sum(1 for o in orphans if per_target_proposals.get(o))} "
            f"total_edges_added={applied}]"
        )

        new_edges: List[Tuple[str, str]] = []
        for child, parents in dag_parents.items():
            for parent in parents:
                new_edges.append((parent, child))
        return new_edges

    def _fit_structure(self, df: pd.DataFrame) -> nx.DiGraph:
        all_nodes = list(df.columns)
        if self.use_llm_column_grouping and not self.config.candidate_edges_path:
            self.column_groupings = self._compute_column_groupings_llm(all_nodes)
            self.column_upweight_factor = self._requested_column_upweight_factor
        return super()._fit_structure(df)

    def _build_network_structure(
        self,
        edges: List[Tuple[str, str]],
        max_parents: int,
        df: pd.DataFrame
    ) -> List[Tuple[str, str]]:
        """
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

        # Step 3: Prepare and sort candidate parents. Without LLM parent
        # ordering, follow the base Bagging pipeline exactly: apply the
        # frequency cutoff before resolving directions and sorting by
        # correlation. The LLM path handles the cutoff inside
        # ``_get_parent_ordering_llm`` so that explicitly endorsed parents can
        # intentionally bypass it.
        if self.use_llm_parent_ordering:
            parent_counts = self._get_edge_directions(parent_counts)
            self._log_checkpoint(
                "after_direction",
                ((p, c) for c, parents in parent_counts.items()
                 for p, n in parents.items() if n > 0),
            )
            sorted_parents = self._get_parent_ordering_llm(parent_counts, df)
        else:
            parent_counts = self._apply_parent_frequency_cutoff(parent_counts)
            self._log_checkpoint(
                "after_cutoff",
                ((p, c) for c, parents in parent_counts.items()
                 for p, n in parents.items() if n > 0),
            )
            parent_counts = self._get_edge_directions(parent_counts)
            self._log_checkpoint(
                "after_direction",
                ((p, c) for c, parents in parent_counts.items()
                 for p, n in parents.items() if n > 0),
            )
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
                elif self.use_llm_cycle_arbitration:
                    cycle_paths = find_directed_paths(
                        child, parent, dag_children, max_paths=self._cycle_arbitration_max_paths
                    )
                    if not cycle_paths or len(cycle_paths) > self._cycle_arbitration_max_paths:
                        continue
                    edges_to_remove = self._resolve_cycle_llm(parent, child, cycle_paths)
                    if not edges_to_remove:
                        continue
                    removed: List[Tuple[str, str]] = []
                    for rm_parent, rm_child in edges_to_remove:
                        if rm_parent in dag_parents.get(rm_child, []):
                            dag_parents[rm_child].remove(rm_parent)
                            dag_children[rm_parent].remove(rm_child)
                            removed.append((rm_parent, rm_child))
                    if has_path(child, parent, dag_children):
                        for rm_parent, rm_child in removed:
                            dag_parents[rm_child].append(rm_parent)
                            dag_children[rm_parent].append(rm_child)
                        logging.info(
                            f"Cycle arbitration: LLM removals {removed} did not break all "
                            f"paths from {child} to {parent}; reverted and skipped {parent} -> {child}"
                        )
                        continue
                    dag_parents[child].append(parent)
                    dag_children[parent].append(child)
                    logging.info(
                        f"Cycle arbitration: removed {removed} to allow {parent} -> {child}"
                    )
        processed_edges: List[Tuple[str, str]] = []
        for child, parents in dag_parents.items():
            for parent in parents:
                processed_edges.append((parent, child))
        self._log_checkpoint("after_greedy", processed_edges)

        self._post_greedy_edges = list(processed_edges)
        self._post_greedy_nodes = set(nodes)
        self._post_po_parent_counts = {
            child: dict(parents) for child, parents in parent_counts.items()
        }
        self._post_po_sorted_parents = {
            child: list(parents) for child, parents in sorted_parents.items()
        }

        # Step 8: Use LLM-based structure refinement
        if self.use_llm_structure_refinement:
            processed_set = set(processed_edges)
            near_miss: List[Tuple[str, str]] = [
                (p, c)
                for c, parents_list in sorted_parents.items()
                for p in parents_list
                if (p, c) not in processed_set
            ]
            processed_edges = self._refine_structure(
                processed_edges, nodes, parent_counts, near_miss
            )
            self._log_checkpoint("after_refinement", processed_edges)

        if self.use_llm_confounder_redirection:
            processed_edges = self._redirect_confounded_edges(
                processed_edges, nodes, parent_counts
            )
            self._log_checkpoint("after_confounder_redirection", processed_edges)

        if self.use_llm_orphan_repair:
            processed_edges = self._repair_orphan_nodes(
                processed_edges, nodes, max_parents
            )
            self._log_checkpoint("after_orphan_repair", processed_edges)

        return processed_edges
