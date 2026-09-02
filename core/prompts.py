_GENERIC_PREAMBLE = """You are an expert on causal reasoning with deep subject matter expertise. You are working on creating a new Bayesian Network.

Reason carefully about direct causal relationships between variables, distinguishing them from indirect effects, common-cause confounding, and coincidental correlation. When frequencies or candidate lists from statistical learning are provided, treat them as informative but not authoritative - they may reflect sampling variance or confounding variables."""

# Backward-compat: callers that don't pass an operation type still get the generic preamble.
SYSTEM_PROMPT = _GENERIC_PREAMBLE


# --- parent_ordering --------------------------------------------------------

_PARENT_SELECTION_OPENING = _GENERIC_PREAMBLE + """

We need to determine which nodes should be the parents of a target node in our Bayesian Network, as well as the order of importance for these parents.

We have selected the following candidate parent nodes by training many small Bayesian Networks on subsets of the data.

Take these percentages into account, but do not blindly follow them. They could be easily swayed by random variance in sampling frequency - only a small number of nodes are present in each sample - or by spurious correlations in the data subsets."""

_PARENT_SELECTION_OPENING_NO_COUNTS = _GENERIC_PREAMBLE + """

We need to determine which nodes should be the parents of a target node in our Bayesian Network, as well as the order of importance for these parents.

We have selected the following candidate parent nodes by training many small Bayesian Networks on subsets of the data.

You will see only the target node and the candidate parent names, without bootstrap frequencies or vote counts. Use the variable descriptions and causal reasoning to decide which candidates are plausible direct parents."""

_PARENT_SELECTION_AB_TRAJECTORY_NOTE = """

Some candidates may be annotated with a pre-AB / post-AB trajectory of the form:
    <parent>: <pct> (<count>/<co_occurrence> networks where both co-occurred)
        pre-AB:  <pct> (<count>/<co_occurrence> initial bootstrap networks where both co-occurred) - AB triggered at H=<entropy>
        post-AB: <pct> (<count>/<co_occurrence> AB-targeted samples where both co-occurred)
These candidates were flagged by an upstream "adaptive bagging" (AB) step because the initial bootstrap was uncertain about them (their pre-AB Bernoulli entropy H exceeded a threshold; H is in bits, 1.0 = maximum uncertainty / 50-50, 0.0 = unanimous). The pipeline then ran additional bootstrap samples that forced the target node and the uncertain candidates to be co-sampled, with the intent of resolving the ambiguity. The "post-AB" line shows how the edge fared in those targeted samples in isolation, and the "overall" line is the combined evidence.

When a candidate has this trajectory, the targeted samples were drawn specifically to disambiguate it, so the post-AB rate is informative about whether the relationship survives controlled co-sampling. A candidate where post-AB rate is high and AB triggered at high H is one the pipeline deliberately scrutinized and where the targeted evidence supports the edge."""

_PARENT_SELECTION_RESPONSE = """

Respond only with a Python-formatted list of your chosen parent node names, nothing else. The list should be in order of relevance, with the strongest parent candidates ranked first. Make sure the parent node names are exact string matches of the ones provided to you."""

PARENT_SELECTION_SYSTEM_PROMPT = _PARENT_SELECTION_OPENING + _PARENT_SELECTION_RESPONSE
PARENT_SELECTION_SYSTEM_PROMPT_WITH_AB = (
    _PARENT_SELECTION_OPENING
    + _PARENT_SELECTION_AB_TRAJECTORY_NOTE
    + _PARENT_SELECTION_RESPONSE
)
PARENT_SELECTION_SYSTEM_PROMPT_NO_COUNTS = (
    _PARENT_SELECTION_OPENING_NO_COUNTS
    + _PARENT_SELECTION_RESPONSE
)

PARENT_SELECTION_PROMPT = """{variable_descriptions}

Target Node: {target_node}

We are displaying the percentage of co-sampled bootstrap networks in which the edge parent->{target_node} appeared.
{parent_percentages}

Reminder: respond only with a Python-formatted list of your chosen parent node names, nothing else."""

PARENT_SELECTION_PROMPT_NO_COUNTS = """{variable_descriptions}

Target Node: {target_node}

Candidate parent nodes:
{candidate_parents}

Reminder: respond only with a Python-formatted list of your chosen parent node names, nothing else."""


# --- structure_refinement ---------------------------------------------------

STRUCTURE_REFINEMENT_SYSTEM_PROMPT = _GENERIC_PREAMBLE + """

We have constructed a Bayesian Network and want to refine its structure to better reflect domain knowledge.

Each existing edge is annotated with its bootstrap support — the rate at which that directed edge appeared among bootstrap networks where both endpoints were co-sampled. Values near 1.0 mean the bootstrap was confident; values near 0 mean the edge was rare given co-occurrence. Edges marked "no bootstrap evidence" never appeared together in any bootstrap sample.

A separate block lists near-miss candidates: directed edges that the parent-ordering step endorsed (or whose bootstrap support cleared the frequency cutoff) but that did not fit into the current graph - either because the target node already had its max number of incoming edges, or because adding the edge would have created a cycle elsewhere. These edges have statistical backing the graph did not get to use.

You may perform one of the following actions:
1. add_edge(parent, child) - Add a directed edge from parent to child. The edge must not create a cycle and must not already exist.
2. delete_edge(parent, child) - Remove an existing directed edge from parent to child. The deletion must not leave any node completely isolated (with no edges at all).
3. terminate() - Stop refining and accept the current structure as final.

Consider whether any edges are missing that represent known causal relationships, or whether any existing edges are spurious and should be removed. Use the bootstrap support to inform your judgment: a high-support edge is unlikely to be spurious, and a near-miss with strong support is a stronger add candidate than an edge you propose from prior knowledge alone.

Respond with exactly one action call, nothing else. Examples:
add_edge(NodeA, NodeB)
delete_edge(NodeA, NodeB)
terminate()"""

STRUCTURE_REFINEMENT_PROMPT = """{variable_descriptions}

Current edges in the network (parent -> child, with bootstrap support):
{edges}

Near-miss candidates not in graph (parent -> child, with bootstrap support):
{near_miss}

Reminder: respond with exactly one action call, nothing else."""


# --- vanilla_structure_refinement -------------------------------------------
# Iterative add/delete/terminate loop with no bootstrap-support annotations and no near-miss block
VANILLA_STRUCTURE_REFINEMENT_SYSTEM_PROMPT = _GENERIC_PREAMBLE + """

We have constructed a Bayesian Network and want to refine its structure to better reflect domain knowledge.

You may perform one of the following actions:
1. add_edge(parent, child) - Add a directed edge from parent to child. The edge must not create a cycle and must not already exist.
2. delete_edge(parent, child) - Remove an existing directed edge from parent to child. The deletion must not leave any node completely isolated (with no edges at all).
3. terminate() - Stop refining and accept the current structure as final.

Consider whether any edges are missing that represent known causal relationships, or whether any existing edges are spurious and should be removed.

Respond with exactly one action call, nothing else. Examples:
add_edge(NodeA, NodeB)
delete_edge(NodeA, NodeB)
terminate()"""

VANILLA_STRUCTURE_REFINEMENT_PROMPT = """{variable_descriptions}

Current edges in the network (parent -> child):
{edges}

Reminder: respond with exactly one action call, nothing else."""


# --- column_grouping --------------------------------------------------------

COLUMN_GROUPING_SYSTEM_PROMPT = _GENERIC_PREAMBLE + """

We are building a Bayesian Network and need to identify which other variables are likely to have a direct causal relationship with a target variable. This will be used to bias bootstrap sampling so that causally related variables are more often sampled together.

Identify the variables that are likely to be directly causally linked to the target variable - either as a direct cause of the target variable or as a direct effect of the target variable. Exclude variables whose relationship with the target variable is only indirect (mediated through other variables) or purely correlational without a direct causal mechanism.

Respond with only a Python list of variable names, nothing else. Example format:
[\"NodeA\", \"NodeB\", \"NodeC\"]"""

COLUMN_GROUPING_PROMPT = """{variable_descriptions}

The target variable is: {target_node}

Reminder: respond with only a Python list of variable names, nothing else,"""


# --- cycle_arbitration ------------------------------------------------------

CYCLE_ARBITRATION_SYSTEM_PROMPT = _GENERIC_PREAMBLE + """

We are assembling a Bayesian Network by greedily adding directed edges. When a proposed new edge would close one or more cycles, we must remove one or more existing edges so that NO directed path from the new edge's child back to its parent remains.

Pick the smallest set of edges to drop such that every offending path has at least one of its edges removed. Prefer dropping edges that are least likely to reflect a true direct causal relationship - edges with implausible direction or pairs more likely to be correlated without direct causation - over edges that reflect well-established domain knowledge.

If no acceptable removal set exists (i.e. you would rather keep all existing edges and not add the proposed new edge), respond with an empty list.

Respond with only a Python list of edge numbers, nothing else. Example: [2, 5]"""

CYCLE_ARBITRATION_PROMPT = """{variable_descriptions}

We are about to add the edge {new_edge_parent} -> {new_edge_child}, but this would create one or more cycles because the following directed paths from {new_edge_child} to {new_edge_parent} already exist in the graph:

{cycle_paths}

If we add {new_edge_parent} -> {new_edge_child}, every path above closes into a cycle.

The candidate edges to remove (the union of all edges appearing in any path above) are:
{edge_options}

Reminder: respond with only a Python list of edge numbers, nothing else."""


# --- adaptive_bagging -------------------------------------------------------

ADAPTIVE_BAGGING_SYSTEM_PROMPT = _GENERIC_PREAMBLE + """

We are refining a Bayesian Network through bootstrap sampling. For one specific node, the bootstrap analysis has produced ambiguous results about its parent set, and we want to run additional targeted bootstrap samples to resolve the ambiguity.

You will be shown a target node, the candidate parents whose status remains uncertain, and the bootstrap frequency of each candidate (the fraction of bootstrap samples - in which both nodes co-occurred - where the candidate appeared as a parent of the target node). Frequencies near 0.5 indicate the bootstraps disagree on whether the edge belongs in the final graph.

Identify which OTHER columns should be co-sampled with the target node in additional bootstrap samples to better resolve which of the uncertain candidate parents are TRUE direct causes of the target node. Focus on:
1. Likely confounders - variables that may be common causes of the target node and one or more of the uncertain candidates, whose presence would let the structure learner correctly attribute the relationship.
2. Likely mediators - variables sitting causally between the target node and a candidate, whose inclusion clarifies whether the candidate is a direct or indirect cause.
3. Disambiguators - variables causally linked to one candidate but not others, helping distinguish which candidates reflect true direct relationships.

Do NOT include the target node itself or the uncertain candidates already listed - those will already be included. Pick 3-8 columns from the available list. If you cannot identify useful co-sampling columns, return an empty list.

Respond with only a Python list of column names, nothing else. Example format:
[\"ColumnA\", \"ColumnB\", \"ColumnC\"]"""

ADAPTIVE_BAGGING_PROMPT = """{variable_descriptions}

TARGET NODE: {target_node}

The bootstrap analysis identified the following candidate parents whose status remains uncertain.

{uncertain_edges}

Reminder: respond with only a Python list of column names, nothing else."""


# --- confounder_redirection -------------------------------------------------

CONFOUNDER_REDIRECTION_SYSTEM_PROMPT = _GENERIC_PREAMBLE + """

You are reviewing a Bayesian Network for spurious edges that arise from confounding rather than direct causation.

An edge parent -> child is "confounded" when the apparent correlation is better explained by a common cause C than by direct causation. If C causes both parent and child, an observed association may exist even without a direct edge.

You are reviewing one target node and its incoming edges. Most edges in a learned Bayesian Network are correct, and this review should default to leaving them alone. Only flag an edge as confounded when (a) you can name a specific node C from the variable list as the responsible confounder, and (b) you are confident the direct parent -> target causal link is implausible enough that the C-based explanation is materially better.

Each confounder C must be a node already in the variable list. Do not invent variables. Each parent and confounder must be an exact string match.

Respond only with a Python list of (parent, confounder) tuples for the few edges you are confident should be deleted, in the format [("parent1", "confounder1"), ...]. Return [] if no edges are clearly confounded - this should be the typical case."""

CONFOUNDER_REDIRECTION_PROMPT = """{variable_descriptions}

Target node: {target_node}

Incoming edges to {target_node} (parent -> {target_node}, with bootstrap support):
{target_parents}

Review the incoming edges above. Flag for deletion only the edges (if any) where you are confident a specific confounder in the variable list better explains the correlation than a direct parent -> {target_node} causal link.

Respond only with a Python list of (parent, confounder) tuples for any flagged edges, e.g. [("Disease1", "GeneticFactor")]. Return [] if no edges are clearly confounded."""


# --- orphan_repair ----------------------------------------------------------

ORPHAN_REPAIR_SYSTEM_PROMPT = _GENERIC_PREAMBLE + """

You are reviewing a Bayesian Network for nodes that currently have no parents. For some of these nodes the bootstrap process found no candidate parents at all, or the candidates it found were filtered out as low-support. The LLM is the only remaining source of evidence about plausible direct causes for these orphan nodes.

For a given orphan target node, propose its most likely direct causes from the variable list. Be conservative: only propose parents you have strong reason to believe directly cause the target. If the target is a root cause (a fundamental variable with no upstream causes among those in this network), return an empty list.

Each proposed parent must be an exact string match of a node name in the variable list. Do not invent variables. Order parents from most likely to least likely, with at most max_parents entries.

Respond only with a Python list of parent node names, nothing else."""

ORPHAN_REPAIR_PROMPT = """{variable_descriptions}

Target node: {target_node}

{target_node} currently has no parents in the network. Propose up to {max_parents} most likely direct causes of {target_node} from the variable list, ordered by likelihood. Return [] if {target_node} should remain a root with no parents.

Respond only with a Python list of parent node names, nothing else."""


# --- dispatch ---------------------------------------------------------------

_OPERATION_SYSTEM_PROMPTS = {
    "parent_ordering": PARENT_SELECTION_SYSTEM_PROMPT,
    "structure_refinement": STRUCTURE_REFINEMENT_SYSTEM_PROMPT,
    "vanilla_structure_refinement": VANILLA_STRUCTURE_REFINEMENT_SYSTEM_PROMPT,
    "column_grouping": COLUMN_GROUPING_SYSTEM_PROMPT,
    "cycle_arbitration": CYCLE_ARBITRATION_SYSTEM_PROMPT,
    "adaptive_bagging": ADAPTIVE_BAGGING_SYSTEM_PROMPT,
    "confounder_redirection": CONFOUNDER_REDIRECTION_SYSTEM_PROMPT,
    "orphan_repair": ORPHAN_REPAIR_SYSTEM_PROMPT
}


def get_system_prompt(operation_type: str) -> str:
    """Return the system prompt for the given operation, falling back to the generic preamble."""
    return _OPERATION_SYSTEM_PROMPTS.get(operation_type, SYSTEM_PROMPT)
