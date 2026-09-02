import itertools
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from pgmpy.readwrite import BIFReader
from pgmpy.estimators import BDeu
import logging
import traceback
from tqdm import tqdm
import re
from typing import Any, Dict, List, Optional, Union, Tuple

from core.models import BayesianNetwork

# Historical paper-run estimates per 1M tokens, with optional long-context
# tiers. These are for reproducibility of logged estimates, not billing advice;
# provider prices can change.
# Each entry is a list of tiers sorted by ascending prompt-token threshold.
# A tier is (max_prompt_tokens, input, cached_input, output); the last tier
# should use float('inf'). The whole call (input + output) is priced at the
# tier its prompt_tokens fall into.
MODEL_PRICING: Dict[str, List[Tuple[float, float, float, float]]] = {
    "gpt-5.4": [
        (272_000,      2.50, 0.25, 15.00),
        (float('inf'), 5.00, 0.50, 22.50),
    ],
    "gemini-3.1-pro-preview": [
        (200_000,      2.00, 0.20, 12.00),
        (float('inf'), 4.00, 0.40, 18.00),
    ],
    "accounts/fireworks/models/qwen3p6-plus": [
        (float('inf'), 0.50, 0.10, 3.00),
    ],
    "accounts/fireworks/models/deepseek-v4-pro": [
        (float('inf'), 1.74, 0.15, 3.48),
    ],
}

def get_model_pricing(model: str) -> Optional[List[Tuple[float, float, float, float]]]:
    """Look up pricing tiers for a model, falling back to base model for dated variants."""
    pricing = MODEL_PRICING.get(model)
    if pricing is not None:
        return pricing
    base_model = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', model)
    return MODEL_PRICING.get(base_model)

def estimate_llm_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> Optional[float]:
    """Per-call cost in USD. Tier is selected by prompt_tokens; cached_prompt_tokens
    are billed at the tier's cached input rate. Returns None if model is unpriced."""
    tiers = get_model_pricing(model)
    if tiers is None:
        return None
    for threshold, input_price, cached_price, output_price in tiers:
        if prompt_tokens <= threshold:
            uncached = max(prompt_tokens - cached_prompt_tokens, 0)
            return (
                uncached * input_price
                + cached_prompt_tokens * cached_price
                + completion_tokens * output_price
            ) / 1_000_000
    return None

def log_training_usage_summary(model: str, training_usage_details: Dict) -> None:
    """Log a formatted summary of LLM token usage during training."""
    if not training_usage_details:
        logging.info("No LLM usage recorded yet.")
        return

    def _is_cached(op: str) -> bool:
        return op.endswith('_cached')

    total_calls = sum(s['calls'] for s in training_usage_details.values())
    total_prompt = sum(s['prompt_tokens'] for s in training_usage_details.values())
    total_completion = sum(s['completion_tokens'] for s in training_usage_details.values())
    total_cached_prompt = sum(s.get('cached_prompt_tokens', 0) for s in training_usage_details.values())
    total_tokens = sum(s['total_tokens'] for s in training_usage_details.values())

    # Price each call individually so tiered pricing is exact even when a
    # bucket spans the threshold. Buckets without per-call records come from
    # older runs and are skipped from cost totals.
    def _bucket_cost(stats: Dict) -> Optional[float]:
        records = stats.get('call_records')
        if records is None:
            return None
        total = 0.0
        for r in records:
            c = estimate_llm_cost(
                model,
                r['prompt_tokens'],
                r['completion_tokens'],
                cached_prompt_tokens=r.get('cached_prompt_tokens', 0),
            )
            if c is None:
                return None
            total += c
        return total

    bucket_costs = {op: _bucket_cost(s) for op, s in training_usage_details.items()}
    has_cost = all(c is not None for c in bucket_costs.values())
    if has_cost:
        total_cost = sum(bucket_costs.values())
        billed_cost = sum(c for op, c in bucket_costs.items() if not _is_cached(op))
        saved_cost = total_cost - billed_cost
    else:
        total_cost = billed_cost = saved_cost = None

    billed_calls = sum(s['calls'] for op, s in training_usage_details.items() if not _is_cached(op))
    cached_calls = total_calls - billed_calls
    total_truncated = sum(s.get('truncated', 0) for s in training_usage_details.values())

    logging.info("\n=== LLM Training Usage Summary ===")
    logging.info(f"Model: {model}")
    logging.info(f"Total API calls: {total_calls} (billed: {billed_calls}, local-cache hits: {cached_calls})")
    logging.info(f"Total tokens: {total_tokens} (input: {total_prompt} [server-cached: {total_cached_prompt}], output: {total_completion})")
    if billed_calls > 0:
        trunc_rate = 100.0 * total_truncated / billed_calls
        logging.info(f"Truncated (max_tokens hit): {total_truncated}/{billed_calls} billed calls ({trunc_rate:.1f}%)")
    if total_cost is not None:
        logging.info(f"Estimated cost without local caching: ${total_cost:.4f}")
        logging.info(f"Estimated cost with local caching:    ${billed_cost:.4f}")
        logging.info(f"Saved by local cache:                 ${saved_cost:.4f}")
    elif not has_cost:
        logging.info("(cost unavailable: this run predates per-call cost accounting)")
    logging.info("\nBreakdown by operation:")
    logging.info("-" * 110)

    for operation, stats in sorted(training_usage_details.items()):
        op_cost = bucket_costs.get(operation)
        cost_str = f" | Cost: ${op_cost:.4f}" if op_cost is not None else ""
        cached_in = stats.get('cached_prompt_tokens', 0)
        trunc = stats.get('truncated', 0)
        trunc_str = f" | Truncated: {trunc:3d}" if trunc else ""
        logging.info(f"{operation:24} | Calls: {stats['calls']:4d} | "
              f"Input: {stats['prompt_tokens']:8d} (cached: {cached_in:8d}) | "
              f"Output: {stats['completion_tokens']:8d} | "
              f"Total: {stats['total_tokens']:8d}{cost_str}{trunc_str}")


def evaluate_model_with_variance(trial_dirs: List[str], model_subdir: str) -> Optional[Dict]:
    """
    Aggregate evaluation.json across trials and compute mean, stdev, and 95% CI for each numeric metric.

    Args:
        trial_dirs: List of trial directories (e.g. [.../trial_0, .../trial_1, ...]).
        model_subdir: Model subdirectory name (e.g. "bagging_bn").

    Returns:
        Dict mirroring evaluation.json structure, with each numeric leaf replaced by
        {"mean", "std", "ci_95_low", "ci_95_high", "values"}, or None if no evaluations found.
    """
    eval_dicts = []
    for td in trial_dirs:
        path = os.path.join(td, model_subdir, "evaluation.json")
        if os.path.exists(path):
            with open(path) as f:
                eval_dicts.append(json.load(f))

    if not eval_dicts:
        return None

    def _aggregate(items):
        """Recursively aggregate a list of structurally-similar dicts/values."""
        if not items:
            return None
        first = items[0]
        if isinstance(first, dict):
            result = {}
            for key in first:
                child_items = [d[key] for d in items if key in d]
                if child_items:
                    result[key] = _aggregate(child_items)
            return result
        elif isinstance(first, (int, float)) and not isinstance(first, bool):
            values = [float(v) for v in items if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not values:
                return None
            arr = np.array(values)
            n = len(arr)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            # 95% CI using t-distribution approximation (1.96 for large n, exact for small)
            if n > 1:
                from scipy.stats import t as t_dist
                t_val = t_dist.ppf(0.975, df=n - 1)
                margin = t_val * std / np.sqrt(n)
            else:
                margin = 0.0
            return {
                "mean": mean,
                "std": std,
                "ci_95_low": mean - margin,
                "ci_95_high": mean + margin,
                "n_trials": n,
                "values": [float(v) for v in values],
            }
        elif isinstance(first, list):
            # Lists of dicts (e.g. propagation_evaluations) — aggregate element-wise
            max_len = max(len(lst) for lst in items if isinstance(lst, list))
            result = []
            for i in range(max_len):
                child_items = [lst[i] for lst in items if isinstance(lst, list) and i < len(lst)]
                if child_items:
                    result.append(_aggregate(child_items))
            return result
        else:
            # Non-numeric leaf (string, etc.) — return first value
            return first

    return _aggregate(eval_dicts)


def compute_classification_metrics(y_true: list, y_pred: list) -> dict:
    """Compute classification metrics for two lists of labels."""
    unique_classes = sorted(set(y_true + y_pred), key=str)
    n_classes = len(unique_classes)

    metrics = {
        "n_samples": len(y_true),
        "n_classes": n_classes,
    }

    if len(set(y_true)) < 2:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["note"] = "Only one class present in true labels"
        return metrics

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    if n_classes == 2:
        pos_label = unique_classes[1]
        metrics["precision"] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    else:
        metrics["precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return metrics


def compare_structure_against_ground_truth(learned_graph, bif_path, test_df) -> dict:
    """
    Compute structural comparison metrics between a learned graph and ground-truth BN.

    Args:
        learned_graph: A networkx DiGraph (e.g. model.graph from any learner)
        bif_path: Path to the ground-truth BIF file.
        test_df: A pd.DataFrame to calculate ground-truth BDeu score on.

    Returns:
        Dict with normalized_hamming_distance, causality F1/precision/recall (macro),
        and edge/node counts.
    """
    def _get_edge_relation(edges: set, a: str, b: str) -> str:
        if (a, b) in edges:
            return "A->B"
        elif (b, a) in edges:
            return "B->A"
        else:
            return "none"
    logging.info("Loading ground truth model...")
    reader = BIFReader(bif_path)
    gt_model = reader.get_model()
    logging.info("Ground truth model loaded.")

    learned_edges = set(learned_graph.edges())
    gt_edges = set(gt_model.edges())

    all_nodes = sorted(set(learned_graph.nodes()) | set(gt_model.nodes()))
    all_pairs = list(itertools.combinations(all_nodes, 2))

    # Normalized Structural Hamming Distance
    hamming = 0
    for a, b in all_pairs:
        gt_rel = _get_edge_relation(gt_edges, a, b)
        learned_rel = _get_edge_relation(learned_edges, a, b)
        if gt_rel != learned_rel:
            hamming += 1
    nhd = hamming / len(all_pairs) if all_pairs else 0.0

    # Causality F1 Score
    y_true = []
    y_pred = []
    for a, b in all_pairs:
        y_true.append(_get_edge_relation(gt_edges, a, b))
        y_pred.append(_get_edge_relation(learned_edges, a, b))

    labels = ["A->B", "B->A", "none"]
    causality_f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    causality_precision_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    causality_recall_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # Direction-sensitive edge precision and recall
    learned_edges_undirected = {frozenset(e) for e in learned_edges}
    gt_edges_undirected = {frozenset(e) for e in gt_edges}

    correct_directed = learned_edges & gt_edges
    reversed_edges = {
        (a, b) for (a, b) in learned_edges
        if (a, b) not in gt_edges and (b, a) in gt_edges
    }
    incorrect_edges = {
        (a, b) for (a, b) in learned_edges
        if frozenset((a, b)) not in gt_edges_undirected
    }
    missing_edges = {
        (a, b) for (a, b) in gt_edges
        if frozenset((a, b)) not in learned_edges_undirected
    }

    directed_precision = len(correct_directed) / len(learned_edges) if learned_edges else 0.0
    directed_recall = len(correct_directed) / len(gt_edges) if gt_edges else 0.0
    directed_f1 = (
        2 * directed_precision * directed_recall / (directed_precision + directed_recall)
        if (directed_precision + directed_recall) > 0 else 0.0
    )

    result = {
        "normalized_hamming_distance": nhd,
        "causality_f1_macro": causality_f1_macro,
        "causality_precision_macro": causality_precision_macro,
        "causality_recall_macro": causality_recall_macro,
        "directed_edge_precision": directed_precision,
        "directed_edge_recall": directed_recall,
        "directed_edge_f1": directed_f1,
        "num_incorrect_edges": len(incorrect_edges),
        "num_missing_edges": len(missing_edges),
        "num_reversed_edges": len(reversed_edges),
        "num_gt_edges": len(gt_edges),
        "num_learned_edges": len(learned_edges),
        "num_nodes": len(all_nodes),
        "num_pairs": len(all_pairs),
    }

    try:
        logging.info("Calculating ground-truth BDeu score")
        scorer = BDeu(test_df, equivalent_sample_size=1.0)
        bdeu_total = 0.0
        for node in all_nodes:
            parents = [p for p in gt_model.get_parents(node) if p in test_df.columns]
            bdeu_total += scorer.local_score(node, parents)
        result["gt_bdeu_score"] = bdeu_total
        logging.info("Ground-truth BDeu score calculated.")
    except Exception as e:
        logging.info(f"Exception encountered in calculated ground-truth BDeu score: {e}")

    return result


def compute_bdeu_score(
    model: BayesianNetwork,
    test_df: pd.DataFrame,
    equivalent_sample_size: float = 1.0
) -> float:
    """
    Compute the BDeu score of the model's graph structure on test_df.

    Only nodes present in both the graph and test_df are included.
    """
    graph_nodes = set(model.graph.nodes())
    available = set(test_df.columns)
    scorer = BDeu(test_df, equivalent_sample_size=equivalent_sample_size)

    total = 0.0
    for node in graph_nodes & available:
        parents = [p for p in model.graph.predecessors(node) if p in available]
        total += scorer.local_score(node, parents)
    return total


def evaluate_model_with_propagation(
    model: BayesianNetwork,
    prediction_questions: list[dict],
    max_workers: int = 1,
    max_hops: Optional[int] = 3,
    num_samples: int = 50,
) -> dict:
    """
    Evaluate classification metrics with belief propagation enabled.

    Args:
        model: A model fitted onto the Markov-blanket inference backend.
        prediction_questions: List of question dicts from generate_ground_truth_predictions.
        max_workers: Number of parallel workers.
        max_hops: Maximum path length for propagation. None allows unlimited.
        num_samples: Number of Monte Carlo samples for integrating soft evidence.

    Returns:
        Dict with classification_metrics, plus the propagation parameters used.
    """
    model_predictions = []
    gt_predictions = []
    synthetic_values = []

    queries = [(q["context"], q["target_node"]) for q in prediction_questions]

    results = model.predict_target_batch_with_propagation(
        queries, max_hops=max_hops, num_samples=num_samples,
        max_workers=max_workers,
    )
    for q, prob_dict in tqdm(zip(prediction_questions, results), total=len(prediction_questions), desc="Evaluating questions (with propagation)"):
        model_predictions.append("__unknown__" if prob_dict is None else str(max(prob_dict, key=prob_dict.get)))
        gt_predictions.append(str(q["ground_truth_prediction"]))
        synthetic_values.append(str(q["synthetic_data_value"]))

    classification_metrics = {}
    if model_predictions:
        classification_metrics["model_vs_ground_truth"] = compute_classification_metrics(
            gt_predictions, model_predictions
        )
        classification_metrics["model_vs_synthetic_data"] = compute_classification_metrics(
            synthetic_values, model_predictions
        )
        classification_metrics["ground_truth_vs_synthetic_data"] = compute_classification_metrics(
            synthetic_values, gt_predictions
        )

    return {
        "classification_metrics_with_propagation": classification_metrics,
        "propagation_params": {
            "max_hops": max_hops,
            "num_samples": num_samples,
        },
    }


def evaluate_model(
    model,
    prediction_questions: list[dict],
    test_df: pd.DataFrame,
    ground_truth_bif_path: str = None,
    max_workers: int = 1,
    evaluate_with_propagation: bool = False,
    skip_classification: bool = True,
) -> dict:
    """
    Evaluate a model using pre-generated prediction questions.

    Each question has: target_node, context, ground_truth_prediction, synthetic_data_value.
    The model predicts each target given the context, then classification metrics are
    computed for three comparisons:
      - model vs ground_truth_prediction
      - model vs synthetic_data_value
      - ground_truth_prediction vs synthetic_data_value

    Args:
        model: A trained model with a predict_target method.
        prediction_questions: List of question dicts from generate_ground_truth_predictions.
        test_df: Test DataFrame (used for BDeu scoring and structural comparison).
        ground_truth_bif_path: Optional path to ground-truth BIF file for structural metrics.
        max_workers: Number of parallel workers (used when model supports predict_target_batch).
        evaluate_with_propagation: If True and the model supports propagation, also
            evaluates classification metrics with belief propagation enabled.
        skip_classification: If True (default), skip per-question prediction and the
            resulting classification metrics.

    Returns:
        Dict with classification_metrics, model_bdeu_score, and optionally structural_metrics
        and classification_metrics_with_propagation.
    """
    model_predictions = []
    gt_predictions = []
    synthetic_values = []

    queries = [(q["context"], q["target_node"]) for q in prediction_questions]

    if skip_classification:
        logging.info("Skipping classification metrics (skip_classification=True).")
    elif model.parameters_fitted:
        results = model.predict_target_batch(queries, max_workers=max_workers)
        for q, prob_dict in tqdm(zip(prediction_questions, results), total=len(prediction_questions), desc="Evaluating questions"):
            model_predictions.append("__unknown__" if prob_dict is None else str(max(prob_dict, key=prob_dict.get)))
            gt_predictions.append(str(q["ground_truth_prediction"]))
            synthetic_values.append(str(q["synthetic_data_value"]))
    else:
        # Any model can be configured without CPTs and therefore cannot answer
        # predictive queries. Skip predictive evaluation —
        # structural metrics and BDeu score are still computed below.
        logging.info(
            "Model parameters were not fitted; skipping predictive evaluation."
        )

    classification_metrics = {}
    if model_predictions:
        classification_metrics["model_vs_ground_truth"] = compute_classification_metrics(
            gt_predictions, model_predictions
        )
        classification_metrics["model_vs_synthetic_data"] = compute_classification_metrics(
            synthetic_values, model_predictions
        )
        classification_metrics["ground_truth_vs_synthetic_data"] = compute_classification_metrics(
            synthetic_values, gt_predictions
        )

    eval_results = {
        "classification_metrics": classification_metrics,
        "num_questions": len(prediction_questions),
        "num_evaluated": len(model_predictions),
    }

    # BDeu score for the learned model
    try:
        eval_results["model_bdeu_score"] = compute_bdeu_score(model, test_df)
    except Exception as e:
        logging.warning(f"BDeu score computation failed: {e}")

    # Structural comparison
    if ground_truth_bif_path:
        if not os.path.exists(ground_truth_bif_path):
            logging.info(f"Error: Ground-Truth Model '{ground_truth_bif_path}' not found")
        else:
            logging.info("Comparing structure against ground truth...")
            try:
                structural_metrics = compare_structure_against_ground_truth(
                    model.graph, ground_truth_bif_path, test_df
                )
                eval_results["structural_metrics"] = structural_metrics
            except Exception as e:
                logging.info(f"Error encountered in ground-truth structural comparison: {e}")
                traceback.print_exc()

    # Optionally evaluate with propagation
    if evaluate_with_propagation and model.parameters_fitted and model.supports_propagation:
        logging.info("Evaluating with belief propagation enabled...")
        propagation_results = evaluate_model_with_propagation(
            model, prediction_questions, max_workers=max_workers
        )
        eval_results["propagation_evaluations"] = [propagation_results]

    return eval_results
