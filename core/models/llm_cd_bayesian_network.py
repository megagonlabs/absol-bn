"""LLMCD baseline (single-pass, target-free): port of Du et al.'s "Causal
Discovery through Synergizing LLM and Data-Driven Reasoning" (KDD 2025),
adapted to the \\name conventions in this repo.

Paper: https://doi.org/10.1145/3711896.3736874
Reference implementation: https://github.com/trytodoit227/LLMCD

Intentional deviation from the published algorithm: LLMCD wraps its pipeline
in an outer iteration loop that, between iterations, computes the parents of
a fixed target node (`GetFMB`) and restricts the data to those columns before
re-running PC. That refinement is target-conditional — its semantics are
defined only relative to a designated outcome variable. We evaluate full-graph
topology recovery against ground-truth `.bif` DAGs, where target-conditional
column pruning has no well-defined meaning, so we run a single pass over the
full variable set.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import Meek, UCSepset
from causallearn.utils.PCUtils.Helper import append_value

from ..llm_session import LlmCallBudgetExceeded, LLMSession
from .bayesian_network import BayesianNetwork
from .config import LLMCDConfig


logging.getLogger("pgmpy").setLevel(logging.WARNING)


_SYSTEM_PROMPT = (
    "You are an expert on causal reasoning with broad domain knowledge. "
    "You are assisting a constraint-based causal discovery algorithm by "
    "answering bounded questions about specific variables and relationships. "
    "Reason about direct causal mechanisms, distinguishing them from indirect "
    "effects, common-cause confounding, and coincidental correlation."
)


def _format_variables(names: List[str], descriptions: Dict[str, str]) -> str:
    lines = []
    for n in names:
        d = descriptions.get(n, "")
        lines.append(f"  - {n}: {d}" if d else f"  - {n}")
    return "\n".join(lines)


def _ci_prompt(x_name: str, y_name: str, cond_names: List[str],
               descriptions: Dict[str, str]) -> str:
    involved = [x_name, y_name] + list(cond_names)
    var_block = _format_variables(involved, descriptions)
    cond_repr = ", ".join(cond_names) if cond_names else "(empty)"
    return (
        f"# Variables\n{var_block}\n\n"
        f"# Task: Conditional Independence Judgment\n"
        f"Are '{x_name}' and '{y_name}' conditionally independent given the "
        f"conditioning set {{{cond_repr}}}?\n\n"
        f"Use your domain knowledge to estimate the probability that this "
        f"conditional independence holds.\n\n"
        f"# Output\n"
        f"Respond with a single number in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, "
        f"0.8, 0.9, 1.0] where higher means more likely that the independence "
        f"holds. Output only the number, nothing else."
    )


def _edge_check_prompt(parent_name: str, child_name: str,
                       descriptions: Dict[str, str]) -> str:
    var_block = _format_variables([parent_name, child_name], descriptions)
    return (
        f"# Variables\n{var_block}\n\n"
        f"# Task\n"
        f"A causal discovery algorithm has currently oriented the edge as "
        f"'{parent_name}' causes '{child_name}'. Choose the most likely "
        f"hypothesis among:\n"
        f"- KEEP: '{parent_name}' causes '{child_name}'\n"
        f"- FLIP: '{child_name}' causes '{parent_name}'\n"
        f"- REMOVE: there is no direct causal relationship\n\n"
        f"# Output\n"
        f"Output a JSON object whose values are integer probabilities (0-100) "
        f"summing to 100, with keys exactly KEEP, FLIP, REMOVE. Output only "
        f"the JSON, with no extra characters.\n\n"
        f"Example: {{\"KEEP\": 70, \"FLIP\": 25, \"REMOVE\": 5}}"
    )


def _orient_prompt(u_name: str, v_name: str,
                   descriptions: Dict[str, str]) -> str:
    var_block = _format_variables([u_name, v_name], descriptions)
    return (
        f"# Variables\n{var_block}\n\n"
        f"# Task\n"
        f"A causal discovery algorithm has identified a direct causal link "
        f"between '{u_name}' and '{v_name}' but could not determine the "
        f"direction. Choose the more likely direction.\n\n"
        f"# Output\n"
        f"Output 1 if '{u_name}' causes '{v_name}'; output 0 if '{v_name}' "
        f"causes '{u_name}'. Output only the digit 0 or 1, nothing else."
    )


def _cycle_prompt(cycle_names: List[str], descriptions: Dict[str, str]) -> str:
    var_block = _format_variables(cycle_names, descriptions)
    chain = " -> ".join(cycle_names + [cycle_names[0]])
    pairs = list(zip(cycle_names, cycle_names[1:] + cycle_names[:1]))
    options_block = "\n".join(
        f"  - Remove: {a} and {b}\n  - Reverse: {a} and {b}"
        for a, b in pairs
    )
    return (
        f"# Variables\n{var_block}\n\n"
        f"# Task\n"
        f"The following directed cycle currently exists in the inferred "
        f"causal graph (arrows go left-to-right; the last node connects back "
        f"to the first):\n  {chain}\n\n"
        f"Propose a single edit that eliminates the cycle. Choose either to "
        f"remove an edge that does not reflect a true direct causal link, or "
        f"to reverse an edge whose direction is more plausibly the other "
        f"way.\n\n"
        f"# Output\n"
        f"Output exactly one line in one of the following forms (pick edges "
        f"from the cycle above):\n{options_block}\n\n"
        f"Output only that single line, with no extra characters."
    )


_DECIMAL_RE = re.compile(r"-?\d+\.?\d*")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


def _parse_first_number(text: str) -> Optional[float]:
    m = _DECIMAL_RE.search(text)
    return float(m.group(0)) if m else None


def _parse_edge_check(text: str) -> Optional[Dict[str, float]]:
    cleaned = _JSON_FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out: Dict[str, float] = {}
    for k in ("KEEP", "FLIP", "REMOVE"):
        v = obj.get(k)
        if v is None:
            return None
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            return None
    return out


def _parse_orient(text: str) -> Optional[int]:
    cleaned = text.strip().splitlines()[0].strip() if text.strip() else ""
    m = re.search(r"\b([01])\b", cleaned)
    return int(m.group(1)) if m else None


_CYCLE_ACTION_RE = re.compile(
    r"^\s*(Remove|Reverse)\s*:\s*(.+?)\s+and\s+(.+?)\s*$", re.IGNORECASE
)


def _parse_cycle_action(text: str, cycle_names: List[str]) -> Optional[Tuple[str, str, str]]:
    """Returns (action, a, b) where action is 'Remove' or 'Reverse' and (a, b)
    are variable names from the cycle. None if parsing fails or the named
    variables are not in the cycle."""
    for line in text.strip().splitlines():
        m = _CYCLE_ACTION_RE.match(line)
        if not m:
            continue
        action = m.group(1).capitalize()
        a, b = m.group(2).strip(), m.group(3).strip()
        if a in cycle_names and b in cycle_names:
            return action, a, b
    return None


def _enumerate_directed_cycles(graph: nx.DiGraph, max_cycles: int) -> List[List[str]]:
    """Return up to `max_cycles` simple directed cycles in `graph` (node-name
    lists, without a repeated terminal node). Uses networkx's
    `simple_cycles`, which yields cycles lazily so we can cap enumeration."""
    out: List[List[str]] = []
    for cycle in nx.simple_cycles(graph):
        if len(cycle) >= 2:
            out.append(list(cycle))
            if len(out) >= max_cycles:
                break
    return out


def _causal_graph_from_node_names(node_names: List[str]) -> CausalGraph:
    cg = CausalGraph(len(node_names), node_names)
    return cg


def _causal_graph_to_directed_edges(cg: CausalGraph, node_names: List[str]) -> List[Tuple[str, str]]:
    """Extract directed (tail->arrow) edges from a CausalGraph, mapping by
    causallearn's internal node index back to the original variable name."""
    edges: List[Tuple[str, str]] = []
    for edge in cg.G.get_graph_edges():
        if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
            i = cg.G.nodes.index(edge.get_node1())
            j = cg.G.nodes.index(edge.get_node2())
            edges.append((node_names[i], node_names[j]))
    return edges


class LLMCDBayesianNetwork(BayesianNetwork):
    """Single-pass, target-free port of LLMCD (Du et al., KDD 2025).
    PC-based structure discovery where the LLM is invoked at four sites:

      1. Borderline CI tests during skeleton discovery (when the CI p-value
         falls within `ci_borderline_threshold` of `alpha`).
      2. Per directed edge after orientation: KEEP / FLIP / REMOVE.
      3. Per undirected edge after orientation: choose a direction.
      4. Per remaining cycle after orientation: remove or reverse one edge.

    The published algorithm's outer iteration loop is deliberately omitted —
    see the module docstring for rationale. Briefly: that loop is
    target-conditional (column-prunes to a fixed outcome's parents between
    passes), which has no well-defined meaning when evaluating full-graph
    topology recovery.

    A hard cap (`max_llm_calls`) on the total LLM call count is enforced via
    `LlmCallBudgetExceeded`; the run aborts rather than degrading to plain
    PC. Cycle enumeration is itself capped at `max_cycles_per_pass` to keep
    the call budget tractable on networks with many simple cycles.
    """

    def __init__(self, config: LLMCDConfig) -> None:
        super().__init__(config)
        self._llm = LLMSession(
            config.llm,
            cache_path=config.cache_path,
            max_calls=config.max_llm_calls,
            verbose=config.verbose,
        )
        self.alpha = config.alpha
        self.ci_test = config.ci_test
        self.ci_borderline_threshold = config.ci_borderline_threshold
        self.max_iterations = config.max_iterations
        self.max_llm_calls = config.max_llm_calls
        self.max_cycles_per_pass = config.max_cycles_per_pass

    # ---------------- LLM plumbing ----------------

    @property
    def llm_model_name(self) -> str:
        return self._llm.model_name

    @property
    def training_usage_details(self) -> Dict[str, Any]:
        return self._llm.usage_details

    @property
    def variable_descriptions(self) -> Dict[str, str]:
        return self._llm.variable_descriptions

    @property
    def _llm_calls_made(self) -> int:
        return self._llm.calls_made

    def _send_message(self, message: str, op: str) -> str:
        return self._llm.send(message, op, system_prompt=_SYSTEM_PROMPT)

    # ---------------- algorithm ----------------

    def _skeleton_discovery(
        self,
        data: np.ndarray,
        node_names: List[str],
    ) -> CausalGraph:
        """Stable skeleton discovery with an LLM-assisted borderline rule.

        Faithful to LLMCD's modified SkeletonDiscovery (their `SkeletonDiscovery.py`):
        for each depth and each (x, y) currently adjacent, iterate over
        conditioning sets S of size `depth`. If p > alpha and |p - alpha| is
        clearly above `ci_borderline_threshold`, remove the edge. If |p - alpha|
        is within the borderline threshold, defer to the LLM."""
        no_of_var = data.shape[1]
        cg = _causal_graph_from_node_names(node_names)
        indep_test = CIT(data, method=self.ci_test)
        cg.set_ind_test(indep_test)

        depth = -1
        from itertools import combinations
        while cg.max_degree() - 1 > depth:
            depth += 1
            edge_removal = []
            for x in range(no_of_var):
                Neigh_x = cg.neighbors(x)
                if len(Neigh_x) < depth - 1:
                    continue
                for y in Neigh_x:
                    Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                    sepsets = set()
                    for S in combinations(Neigh_x_noy, depth):
                        p = cg.ci_test(x, y, S)
                        if p > self.alpha and (p - self.alpha) > self.ci_borderline_threshold:
                            edge_removal.append((x, y))
                            edge_removal.append((y, x))
                            for s in S:
                                sepsets.add(s)
                        elif abs(p - self.alpha) < self.ci_borderline_threshold:
                            cond_names = [node_names[i] for i in S]
                            prompt = _ci_prompt(
                                node_names[x], node_names[y], cond_names,
                                self.variable_descriptions,
                            )
                            try:
                                response = self._send_message(prompt, "ci_borderline")
                            except LlmCallBudgetExceeded:
                                raise
                            score = _parse_first_number(response)
                            if score is not None and score >= 0.5:
                                edge_removal.append((x, y))
                                edge_removal.append((y, x))
                                for s in S:
                                    sepsets.add(s)
                    if (x, y) in edge_removal or not cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y]):
                        append_value(cg.sepset, x, y, tuple(sepsets))
                        append_value(cg.sepset, y, x, tuple(sepsets))
            for (x, y) in list(set(edge_removal)):
                edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                if edge1 is not None:
                    cg.G.remove_edge(edge1)
        return cg

    def _llm_revise_directed_edges(
        self,
        cg: CausalGraph,
        node_names: List[str],
    ) -> CausalGraph:
        """For each fully-directed edge, ask the LLM to KEEP / FLIP / REMOVE.

        Decisions are applied to a copy of the graph after enumerating the
        original directed-edge list (so revising one edge does not change
        the input set of another)."""
        directed: List[Tuple[int, int]] = []
        for edge in list(cg.G.get_graph_edges()):
            if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
                i = cg.G.nodes.index(edge.get_node1())
                j = cg.G.nodes.index(edge.get_node2())
                directed.append((i, j))

        for i, j in directed:
            prompt = _edge_check_prompt(node_names[i], node_names[j], self.variable_descriptions)
            response = self._send_message(prompt, "edge_check")
            scores = _parse_edge_check(response)
            if scores is None:
                continue
            decision = max(scores, key=scores.get)
            edge_obj = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
            if edge_obj is not None:
                cg.G.remove_edge(edge_obj)
            if decision == "KEEP":
                cg.G.add_edge(Edge(cg.G.nodes[i], cg.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW))
            elif decision == "FLIP":
                cg.G.add_edge(Edge(cg.G.nodes[j], cg.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW))
            # REMOVE: leave the edge removed.
        return cg

    def _llm_orient_undirected_edges(
        self,
        cg: CausalGraph,
        node_names: List[str],
    ) -> CausalGraph:
        """For each undirected edge after Meek, ask the LLM for a direction."""
        undirected_pairs = set()
        for edge in list(cg.G.get_graph_edges()):
            if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.TAIL:
                i = cg.G.nodes.index(edge.get_node1())
                j = cg.G.nodes.index(edge.get_node2())
                undirected_pairs.add(tuple(sorted((i, j))))

        for i, j in undirected_pairs:
            prompt = _orient_prompt(node_names[i], node_names[j], self.variable_descriptions)
            response = self._send_message(prompt, "edge_orient")
            choice = _parse_orient(response)
            edge_obj = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
            if edge_obj is not None:
                cg.G.remove_edge(edge_obj)
            if choice == 1:
                cg.G.add_edge(Edge(cg.G.nodes[i], cg.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW))
            elif choice == 0:
                cg.G.add_edge(Edge(cg.G.nodes[j], cg.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW))
            # Unparseable: drop the edge.
        return cg

    def _llm_resolve_cycles(
        self,
        cg: CausalGraph,
        node_names: List[str],
    ) -> CausalGraph:
        """Detect directed cycles and ask the LLM to remove or reverse one
        edge per cycle. Cycle enumeration is capped at `max_cycles_per_pass`
        per pass; after each pass the graph is rechecked, up to a small fixed
        number of passes."""
        for pass_idx in range(3):
            nx_graph = self._causal_graph_to_nx_directed(cg, node_names)
            cycles = _enumerate_directed_cycles(nx_graph, self.max_cycles_per_pass)
            if not cycles:
                return cg
            for cycle in cycles:
                prompt = _cycle_prompt(cycle, self.variable_descriptions)
                response = self._send_message(prompt, "cycle_resolve")
                parsed = _parse_cycle_action(response, cycle)
                if parsed is None:
                    continue
                action, a, b = parsed
                if a not in node_names or b not in node_names:
                    continue
                i = node_names.index(a)
                j = node_names.index(b)
                edge_obj = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
                if edge_obj is None:
                    continue
                cg.G.remove_edge(edge_obj)
                if action == "Reverse":
                    cg.G.add_edge(Edge(cg.G.nodes[j], cg.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW))
            logging.info("LLMCD cycle pass %d: resolved %d cycles", pass_idx + 1, len(cycles))
        # Final safety net: if cycles remain, break them deterministically by
        # dropping the lowest-priority edge per cycle. This is rare but keeps
        # the output a DAG.
        nx_graph = self._causal_graph_to_nx_directed(cg, node_names)
        leftover = _enumerate_directed_cycles(nx_graph, self.max_cycles_per_pass)
        for cycle in leftover:
            i = node_names.index(cycle[-1])
            j = node_names.index(cycle[0])
            edge_obj = cg.G.get_edge(cg.G.nodes[i], cg.G.nodes[j])
            if edge_obj is not None:
                cg.G.remove_edge(edge_obj)
        return cg

    @staticmethod
    def _causal_graph_to_nx_directed(cg: CausalGraph, node_names: List[str]) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(node_names)
        for edge in cg.G.get_graph_edges():
            if edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.ARROW:
                i = cg.G.nodes.index(edge.get_node1())
                j = cg.G.nodes.index(edge.get_node2())
                g.add_edge(node_names[i], node_names[j])
        return g

    # ---------------- training entrypoint ----------------

    def _fit_structure(self, df: pd.DataFrame) -> nx.DiGraph:
        """Run the LLM-CD pipeline and return its learned DAG."""
        train_df = df.astype(str)
        node_names = list(train_df.columns)

        # Integer-encode for causal-learn's chisq/fisherz machinery.
        encoded_cols: List[np.ndarray] = []
        for col in node_names:
            codes, _ = pd.factorize(train_df[col].astype(str), sort=True)
            encoded_cols.append(codes.astype(np.float64))
        data = np.column_stack(encoded_cols)

        logging.info(
            "LLMCD: starting on %d nodes (alpha=%.3f, ci_test=%s, max_llm_calls=%d)",
            len(node_names), self.alpha, self.ci_test, self.max_llm_calls,
        )

        cg: Optional[CausalGraph] = None
        for it in range(self.max_iterations):
            logging.info("LLMCD: iteration %d / %d", it + 1, self.max_iterations)
            cg = self._skeleton_discovery(data, node_names)
            cg = UCSepset.uc_sepset(cg)
            cg = Meek.meek(cg)
            cg = self._llm_revise_directed_edges(cg, node_names)
            cg = self._llm_orient_undirected_edges(cg, node_names)
            cg = self._llm_resolve_cycles(cg, node_names)

        assert cg is not None
        edges = _causal_graph_to_directed_edges(cg, node_names)

        graph = nx.DiGraph()
        graph.add_nodes_from(node_names)
        graph.add_edges_from(edges)
        self.matrix = pd.DataFrame(
            0, index=node_names, columns=node_names, dtype=int
        )
        for parent, child in edges:
            self.matrix.at[parent, child] = 1

        logging.info(
            "LLMCD: complete. %d edges, %d LLM calls used (cap=%d).",
            len(edges), self._llm_calls_made, self.max_llm_calls,
        )
        return graph
