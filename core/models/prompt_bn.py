"""PromptBN baseline.

Based on Zhang et al., "Bayesian Network Structure Discovery Using Large
Language Models," TMLR (2026): https://openreview.net/forum?id=G4mrO8LVix
"""

import json
import logging
from typing import Any, Dict, List

import networkx as nx
import pandas as pd

from ..llm_session import LLMSession
from .bayesian_network import BayesianNetwork
from .config import PromptBNConfig


logging.getLogger('pgmpy').setLevel(logging.WARNING)

SYSTEM_PROMPT = "You are a helpful assistant that constructs Bayesian Networks."

BASELINE_PROMPT = """
You are an expert in building Bayesian Networks. You will receive variables in table format with columns [node, var_name, var_description, var_distribution]. Your task is to construct a Bayesian Network as a Directed Acyclic Graph (DAG) based on the table following the instructions below.

[Instructions]
1. Parse the table to understand:
   - The node id (node).
   - The variable name (var_name).
   - The semantic meaning of each variable (var_description).
   - The distribution type (var_distribution).

2. Construct a Bayesian Network as a Directed Acyclic Graph (DAG) by:
   - Proposing parents (if any) for each node based on relevant domain knowledge or any clues from the table.
   - Ensuring no cycles exist.

3. Provide a strict JSON object with the following structure:
   {{
     "bn": {{
       "nodes": [
         {{
           "node_id": <integer>,
           "node_name": <string>,
           "parents": [<string>, ...],
           "description": <string>,
           "distribution": <string>,
           "conditional_probability_table": <string>
         }},
         ...
       ],
       "edges": [
         {{
           "from": <string>,
           "to": <string>,
           "justification": <string>
         }},
         ...
       ],
       "network_summary": <string>
     }}
   }}


[Output Format]
1. In "nodes":
   - "node_id": A unique ID or index for each node.
   - "node_name": The variable name from var_name column.
   - "parents": Array of node_name values for parent nodes.
   - "description": Short text from var_description.
   - "distribution": Data from var_distribution column.
   - "conditional_probability_table": e.g. "P(tub | asia)".

2. In "edges":
   - "from" and "to": References to the "node_name" fields, indicating parent-child relationships.
   - "justification": A concise reason for why this relationship exists.

3. "network_summary": A concise explanation of how the Bayesian Network structure was derived.

4. Output ONLY the valid JSON object, with no additional commentary or text.

[Variables]
{desc_variables}
"""


class PromptBN(BayesianNetwork):
    """
    Faithful recreation of PromptBNGenerator from
    https://github.com/sherryzyh/llmbn/blob/main/llmbn/generators/promptbn.py.

    Generates a Bayesian Network structure. The LLM is given a table of
    variables and returns a strict JSON object describing the full DAG (nodes
    with parent lists, edges with justifications). Edges are extracted from
    each node's parent list and stored as a networkx DiGraph (`self.graph`)
    and as a pandas DataFrame adjacency matrix (`self.matrix`).

    By default it preserves the original structure-only behavior. Setting
    ``fit_parameters=True`` fits the shared global inference backend afterward.
    """

    def __init__(self, config: PromptBNConfig) -> None:
        super().__init__(config)
        # The upstream baseline issues one uncached call, so no cache_path.
        self._llm = LLMSession(config.llm, verbose=config.verbose)

    @property
    def llm_model_name(self) -> str:
        return self._llm.model_name

    @property
    def training_usage_details(self) -> Dict[str, Dict[str, Any]]:
        return self._llm.usage_details

    @property
    def variable_descriptions(self) -> Dict[str, str]:
        return self._llm.variable_descriptions

    def _send_message(self, message: str, operation_type: str = "general") -> str:
        return self._llm.send(message, operation_type, system_prompt=SYSTEM_PROMPT)

    def _fit_structure(self, df: pd.DataFrame) -> nx.DiGraph:
        train_df = df.astype(str)
        edges = self._generate_structure(train_df)
        graph = nx.DiGraph(edges)
        graph.add_nodes_from(df.columns)
        self.matrix = self._construct_matrix(edges, list(df.columns))
        return graph

    def _format_desc_variables(self, df: pd.DataFrame) -> str:
        # Adapter only: the upstream generator received `desc_variables` already
        # formatted by its caller; this builds the same kind of table from a
        # DataFrame so train(df) can plug into our pipeline.
        explanations = self.variable_descriptions
        lines = ["| node | var_name | var_description | var_distribution |",
                 "| --- | --- | --- | --- |"]
        for i, col in enumerate(df.columns):
            description = explanations.get(col, "")
            states = sorted(df[col].dropna().astype(str).unique().tolist())
            distribution = "categorical(" + ", ".join(states) + ")"
            lines.append(f"| {i} | {col} | {description} | {distribution} |")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        content = response.replace("```json\n", "").replace("\n```", "").strip()
        try:
            return json.loads(content, strict=False)
        except json.JSONDecodeError as e:
            logging.debug("Unjsonified raw generation:\n%s", response)
            raise ValueError(f"JSON decode error: {e}") from e

    def _edges_from_generation(self, generation: Dict[str, Any], all_nodes: set) -> List[tuple]:
        bn = generation.get("bn", generation)
        edges: List[tuple] = []
        seen = set()
        for node in bn.get("nodes", []):
            child = node.get("node_name")
            if child not in all_nodes:
                continue
            for parent in node.get("parents", []) or []:
                if parent in all_nodes and parent != child and (parent, child) not in seen:
                    edges.append((parent, child))
                    seen.add((parent, child))
        return edges

    def _construct_matrix(
        self, edges: List[tuple], dag_variables: List[str]
    ) -> pd.DataFrame:
        # Mirrors `construct_matrix_from_nodes` in the original repo: a binary
        # adjacency DataFrame indexed/columned by `dag_variables`, where
        # entry [parent, child] == 1 iff parent -> child appears in `edges`.
        matrix = pd.DataFrame(
            0, index=dag_variables, columns=dag_variables, dtype=int
        )
        for parent, child in edges:
            matrix.at[parent, child] = 1
        return matrix

    def _generate_structure(self, train_df: pd.DataFrame) -> List[tuple]:
        all_nodes = set(train_df.columns)
        desc_variables = self._format_desc_variables(train_df)
        prompt = BASELINE_PROMPT.format(desc_variables=desc_variables)

        response = self._send_message(prompt, "structure_generation")
        generation = self._parse_response(response)
        self.generation = generation
        edges = self._edges_from_generation(generation, all_nodes)
        return edges
