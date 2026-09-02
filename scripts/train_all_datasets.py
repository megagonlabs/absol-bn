#!/usr/bin/env python3
"""
Train all model types across multiple datasets defined in a JSON config file.

Usage:
    python scripts/train_all_datasets.py config1.json [config2.json ...]

The config file has three sections:
  - "defaults": shared hyperparameters applied to every dataset (merged under dataset overrides).
            Supports "num_trials" (default 1) to repeat each dataset×run N times.
            When num_trials > 1, outputs go to .../trial_0/, .../trial_1/, etc.
  - "runs": list of run configurations for sweeping over any hyperparameters. Each entry has a
            "name" plus any parameter overrides (e.g. llm, num_samples, parent_frequency_cutoff).
            If omitted, a single run using defaults is performed.
  - "datasets": array of dataset entries. Per-dataset values override defaults.
                Each dataset may include a "models" list to restrict which model types
                are trained. If omitted, all models are trained.

Valid model IDs for the "models" list:
  bagging, llm_bagging_all_augs,
  llm_bagging_parent_ordering,
  llm_bagging_column_grouping, llm_bagging_cycle_arbitration,
  llm_bagging_adaptive_bagging,
  llm_bagging_all_augs_plus_vanilla_refinement,
  llm_bagging_all_augs_plus_refinement_w_support,
  llm_bagging_all_augs_plus_narrow_refinement,
  ges, pc, prompt_bn, bfs, llm_cd

Leave-one-out ablation model IDs (train all augs except the named one):
  llm_bagging_loo_<aug>
  where <aug> is one of: parent_ordering, column_grouping,
  cycle_arbitration, adaptive_bagging (the entries in LLM_AUGS below).

Graph refinement is not part of "all augs" — it is available only through the
three llm_bagging_all_augs_plus_*_refinement model IDs above.

Adaptive-bagging tunables (read from config p, all optional):
  num_adaptive_passes (int, default 1)
  num_targeted_samples_per_node (int, default 10)
  uncertainty_entropy_threshold (float, default 0.7 bits)
  min_co_occurrence_for_uncertainty (int, default 5)

LLM provider config key (optional, used by LLM-backed models):
  "llm_provider": str — "openai" (default), "fireworks", or "gemini". Selects which
                       API key env var and base URL the OpenAI client targets.

Training timeout (optional, applies to every model):
  "train_timeout_seconds": int — wall-clock cap per trainer call (default 48h).
                                 Raised via SIGALRM; counted as a model failure on overrun.

Structure-search config keys (optional, used by GES-based models):
  "score_method": str          — local score alias (default "bdeu"; see core/scoring.py)
  "structure_algorithm": str   — "ges" (default) or "dges"
  "score_params": dict|null    — extra score parameters
  "lambda_value": float|null   — BIC penalty multiplier
  "sample_batch_size": int     — bootstrap samples per pool call (default 10).
                                 Keep this small: causal-learn leaks ~270 MB per GES
                                 call, and a small batch restarts pool workers often
                                 enough for the OS to reclaim it.

Structure-only by default:
  "fit_parameters" defaults to false, so models learn a graph but do not fit CPTs;
  evaluation reports structural metrics and a BDeu score, not predictive metrics.
  Set it to true to fit CPTs and enable inference.
"""

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_TRAIN_TIMEOUT_SECONDS = 48 * 60 * 60


@contextmanager
def train_timeout(seconds: int):
    """Raise TimeoutError if the wrapped block runs longer than `seconds` (POSIX, main thread only)."""
    def _handler(signum, frame):
        raise TimeoutError(f"Training exceeded {seconds // 3600}h timeout")
    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)

sys.path.append(".")
from core.evaluation_utils import evaluate_model, evaluate_model_with_variance, log_training_usage_summary

from core.models import (
    BFSBayesianNetwork,
    BFSConfig,
    BaggingBayesianNetwork,
    BaggingConfig,
    GESBayesianNetwork,
    GESConfig,
    GESSettings,
    LLMBaggingBayesianNetwork,
    LLMBaggingConfig,
    LLMCDBayesianNetwork,
    LLMCDConfig,
    LLMSettings,
    PCBayesianNetwork,
    PCConfig,
    PromptBN,
    PromptBNConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_variable_descriptions(path: str | None):
    if not path:
        return None
    if not os.path.exists(path):
        logging.error(f"Variable description file '{path}' not found")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def setup_logging(output_dir: str):
    """Reset logging for a new model run."""
    root = logging.getLogger()
    root.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "training.log")),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def evaluate_and_save(model, config, output_dir):
    """Evaluate a trained model using the dataset paths in its resolved config."""
    train_path = Path(config["csv_path"])
    test_csv_path = Path(
        config.get("test_csv_path") or train_path.with_name("test.csv")
    )
    if not test_csv_path.exists():
        logging.warning(f"Test CSV '{test_csv_path}' not found — skipping evaluation")
        return
    test_df = pd.read_csv(test_csv_path, keep_default_na=False)
    logging.info(f"Test data loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    questions_path = Path(
        config.get("prediction_questions_path")
        or train_path.with_name("prediction_questions.json")
    )
    prediction_questions = []
    if questions_path.exists():
        with questions_path.open() as f:
            prediction_questions = json.load(f)
    else:
        logging.warning(
            "Prediction questions '%s' not found; continuing with BDeu and "
            "structural evaluation only.",
            questions_path,
        )

    eval_results = evaluate_model(
        model, prediction_questions, test_df,
        ground_truth_bif_path=config.get("ground_truth_model_path"),
        max_workers=config["max_workers"],
        skip_classification=not model.parameters_fitted or not prediction_questions,
    )
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(eval_results, f, indent=2, default=str)


def save_config(output_dir: str, config: dict):
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def _base_config_kwargs(p):
    return {
        "ignore_columns": tuple(p.get("ignore_columns", ())),
        "random_seed": p.get("random_seed"),
        "verbose": p["verbose"],
        "fit_parameters": bool(p.get("fit_parameters", False)),
        "max_workers": p["max_workers"],
    }


def _llm_settings(p, max_tokens=None):
    return LLMSettings(
        model_name=p["llm"],
        provider=p.get("llm_provider", "openai"),
        max_tokens=int(max_tokens if max_tokens is not None else p["llm_max_tokens"]),
        temperature=p["llm_temperature"],
        reasoning_effort=p.get("llm_reasoning_effort"),
        variable_descriptions=p.get("variable_descriptions") or {},
    )


def _ges_settings(p):
    return GESSettings(
        score_method=p.get("score_method", "bdeu"),
        score_params=p.get("score_params"),
        lambda_value=p.get("lambda_value"),
        structure_algorithm=p.get("structure_algorithm", "ges"),
    )


def _bagging_config_kwargs(
    p, *, candidate_edges_path=None, save_candidate_edges_path=None
):
    return {
        **_base_config_kwargs(p),
        "num_samples": p["num_samples"],
        "num_rows_per_sample": p["num_rows_per_sample"],
        "num_columns_per_sample": p["num_cols_per_sample"],
        "max_parents_per_node": p["max_parents_per_node"],
        "max_nodes_per_markov_blanket": p.get("num_nodes_per_markov_blanket"),
        "parent_frequency_cutoff": p["parent_frequency_cutoff"],
        "candidate_edges_path": candidate_edges_path,
        "save_candidate_edges_path": save_candidate_edges_path,
        "sample_batch_size": p.get("sample_batch_size", 10),
        "ges": _ges_settings(p),
    }

def train_prompt_bn(df, output_dir, p):
    model = PromptBN(PromptBNConfig(
        **_base_config_kwargs(p), llm=_llm_settings(p, max_tokens=16384)
    )).fit(df)
    log_training_usage_summary(model.llm_model_name, model.training_usage_details)
    model.save(os.path.join(output_dir, "prompt_bn.pkl"))
    return model

def train_bfs(df, output_dir, p):
    model = BFSBayesianNetwork(BFSConfig(
        **_base_config_kwargs(p),
        llm=_llm_settings(p, max_tokens=8192),
        include_statistics=True,
    )).fit(df)
    log_training_usage_summary(model.llm_model_name, model.training_usage_details)
    model.save(os.path.join(output_dir, "bfs.pkl"))
    return model

def train_llm_cd(df, output_dir, p):
    model = LLMCDBayesianNetwork(LLMCDConfig(
        **_base_config_kwargs(p),
        llm=_llm_settings(p, max_tokens=int(p.get("llm_max_tokens", 1024))),
        alpha=float(p.get("llmCD_alpha", 0.05)),
        ci_test=str(p.get("llmCD_ci_test", "chisq")),
        ci_borderline_threshold=float(p.get("llmCD_ci_borderline_threshold", 0.001)),
        max_iterations=int(p.get("llmCD_max_iterations", 1)),
        max_llm_calls=int(p.get("llmCD_max_llm_calls", 20000)),
        max_cycles_per_pass=int(p.get("llmCD_max_cycles_per_pass", 200)),
        cache_path=p.get("llm_cache_path"),
    )).fit(df)
    log_training_usage_summary(model.llm_model_name, model.training_usage_details)
    model.save(os.path.join(output_dir, "llm_cd.pkl"))
    return model


def train_bagging(df, output_dir, p):
    model = BaggingBayesianNetwork(BaggingConfig(
        **_bagging_config_kwargs(
            p, save_candidate_edges_path=p.get("candidate_edges_path")
        )
    )).fit(df)
    model.save(os.path.join(output_dir, "bagging.pkl"))
    return model


def train_llm_bagging(df, output_dir, p, *,
                         use_llm_parent_ordering=False,
                         use_llm_structure_refinement=False,
                         structure_refinement_show_support=True,
                         use_llm_column_grouping=False,
                         column_upweight_factor=5.0, use_llm_cycle_arbitration=False,
                         use_llm_adaptive_bagging=False,
                         use_llm_confounder_redirection=False,
                         use_llm_orphan_repair=False,
                         max_tokens=None):
    max_tokens = max_tokens if max_tokens is not None else p["llm_max_tokens"]
    if use_llm_column_grouping or use_llm_adaptive_bagging:
        candidate_edges_path = None
    else:
        candidate_edges_path = p.get("candidate_edges_path")
    model = LLMBaggingBayesianNetwork(LLMBaggingConfig(
        **_bagging_config_kwargs(p, candidate_edges_path=candidate_edges_path),
        llm=_llm_settings(p, max_tokens=max_tokens),
        parent_ordering=use_llm_parent_ordering,
        parent_ordering_show_counts=bool(p.get("parent_ordering_show_counts", True)),
        structure_refinement=use_llm_structure_refinement,
        structure_refinement_show_support=structure_refinement_show_support,
        column_grouping=use_llm_column_grouping,
        column_upweight_factor=column_upweight_factor,
        cycle_arbitration=use_llm_cycle_arbitration,
        adaptive_bagging=use_llm_adaptive_bagging,
        num_adaptive_passes=int(p.get("num_adaptive_passes", 1)),
        num_targeted_samples_per_node=int(p.get("num_targeted_samples_per_node", 10)),
        uncertainty_entropy_threshold=float(p.get("uncertainty_entropy_threshold", 0.7)),
        min_co_occurrence_for_uncertainty=int(p.get("min_co_occurrence_for_uncertainty", 5)),
        confounder_redirection=use_llm_confounder_redirection,
        orphan_repair=use_llm_orphan_repair,
        cache_path=p.get("llm_cache_path"),
    )).fit(df)
    log_training_usage_summary(model.llm_model_name, model.training_usage_details)
    model.save(os.path.join(output_dir, "llm_bagging.pkl"))
    return model


def train_pc(df, output_dir, p):
    if len(df) > 100000:
        seed = p["random_seed"] if "random_seed" in p else None
        df = df.sample(n=100000, random_state=seed).reset_index(drop=True)
    model = PCBayesianNetwork(PCConfig(**_base_config_kwargs(p))).fit(df)
    model.save(os.path.join(output_dir, "pc.pkl"))
    return model


def train_ges(df, output_dir, p):
    if len(df) > 100000:
        seed = p["random_seed"] if "random_seed" in p else None
        df = df.sample(n=100000, random_state=seed).reset_index(drop=True)
    model = GESBayesianNetwork(GESConfig(
        **_base_config_kwargs(p),
        max_parents_per_node=p["max_parents_per_node"],
        ges=_ges_settings(p),
    )).fit(df)
    model.save(os.path.join(output_dir, "ges.pkl"))
    return model


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Single source of truth for the LLM augmentations that compose "all augs".
# Each entry: (aug_name, trainer kwarg).
# Graph refinement is excluded rather than treated as a fifth augmentation.
LLM_AUGS = [
    ("parent_ordering",        "use_llm_parent_ordering"),
    ("column_grouping",        "use_llm_column_grouping"),
    ("cycle_arbitration",      "use_llm_cycle_arbitration"),
    ("adaptive_bagging",       "use_llm_adaptive_bagging")
]


def _augs_kwargs(exclude=()):
    """All use_llm_* flags True except those whose aug name is in `exclude`."""
    excluded = set(exclude)
    return {kw: True for name, kw in LLM_AUGS if name not in excluded}


def _llm_common(p):
    return {"column_upweight_factor": p.get("column_upweight_factor", 5.0)}


def _llm_entries(prefix, display, subdir, trainer):
    """Generate all-augs, per-aug, and LOO entries for an LLM-augmented trainer."""
    entries = [(
        f"{prefix}_all_augs",
        f"{display} All Augs",
        f"{subdir}_all_augs",
        lambda df, d, p, _t=trainer: _t(df, d, p, **_augs_kwargs(), **_llm_common(p)),
    )]
    for aug, kw in LLM_AUGS:
        pretty = aug.replace("_", " ").title()
        entries.append((
            f"{prefix}_{aug}",
            f"{display} {pretty}",
            f"{subdir}_only_{aug}",
            lambda df, d, p, _kw=kw, _t=trainer:
                _t(df, d, p, **{_kw: True}, **_llm_common(p)),
        ))
    for left_out, _ in LLM_AUGS:
        kwargs = _augs_kwargs(exclude=[left_out])
        entries.append((
            f"{prefix}_loo_{left_out}",
            f"{display} LOO (no {left_out})",
            f"{subdir}_loo_{left_out}",
            lambda df, d, p, _kw=kwargs, _t=trainer:
                _t(df, d, p, **_kw, **_llm_common(p)),
        ))
    return entries


MODEL_DEFS = [
    ("bagging", "Bagging Bayesian Network", "bagging",
     lambda df, d, p: train_bagging(df, d, p)),
    *_llm_entries("llm_bagging", "LLM Bagging Bayesian Network", "llm_bagging", train_llm_bagging),
    ("llm_bagging_all_augs_plus_vanilla_refinement",
     "LLM Bagging All Augs + Vanilla Refinement",
     "llm_bagging_all_augs_plus_vanilla_refinement",
     lambda df, d, p: train_llm_bagging(
         df, d, p,
         **_augs_kwargs(),
         use_llm_structure_refinement=True,
         structure_refinement_show_support=False,
         **_llm_common(p),
     )),
    ("llm_bagging_all_augs_plus_refinement_w_support",
     "LLM Bagging All Augs + Refinement w/ Bootstrap Support",
     "llm_bagging_all_augs_plus_refinement_w_support",
     lambda df, d, p: train_llm_bagging(
         df, d, p,
         **_augs_kwargs(),
         use_llm_structure_refinement=True,
         structure_refinement_show_support=True,
         **_llm_common(p),
     )),
    ("llm_bagging_all_augs_plus_narrow_refinement",
     "LLM Bagging All Augs + Narrow-Scope Refinement",
     "llm_bagging_all_augs_plus_narrow_refinement",
     lambda df, d, p: train_llm_bagging(
         df, d, p,
         **_augs_kwargs(),
         use_llm_confounder_redirection=True,
         use_llm_orphan_repair=True,
         **_llm_common(p),
     )),
    ("ges", "GES Bayesian Network", "ges",
     lambda df, d, p: train_ges(df, d, p)),
    ("pc", "PC Bayesian Network", "pc",
     lambda df, d, p: train_pc(df, d, p)),
    ("prompt_bn", "PromptBN", "prompt_bn",
     lambda df, d, p: train_prompt_bn(df, d, p)),
    ("bfs", "BFS Bayesian Network", "bfs",
     lambda df, d, p: train_bfs(df, d, p)),
    ("llm_cd", "LLM-CD", "llm_cd",
     lambda df, d, p: train_llm_cd(df, d, p)),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_config(config_path: str) -> list[str]:
    """Run all training for a single config file. Returns list of failed model labels."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    defaults = config.get("defaults", {})
    runs = config.get("runs", [{}])
    datasets = config["datasets"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = config.get("name")
    suffix = config_name if config_name else "multi"
    overall_log_dir = f"logs/{timestamp}_{suffix}"
    os.makedirs(overall_log_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(overall_log_dir, "training_config.json"))

    num_trials = int(defaults.get("num_trials", 1))

    print(f"=== Training all models across {len(datasets)} dataset(s), {len(runs)} run(s), {num_trials} trial(s) ===")
    print(f"Config: {config_path}")
    print(f"Overall log directory: {overall_log_dir}\n")

    overall_failed = []

    for run_cfg in runs:
        run_name = run_cfg.get("name", "default")
        print("=" * 60)
        print(f"Run: {run_name}")
        print("=" * 60 + "\n")

        for dataset in datasets:
            name = dataset["name"]
            csv_path = dataset["csv_path"]

            # Merge: defaults < dataset overrides < run overrides (all keys except "name")
            p = {**defaults, **dataset}
            for k, v in run_cfg.items():
                if k != "name":
                    p[k] = v

            # Coerce types
            p["num_samples"] = int(p.get("num_samples", 1000))
            p["num_cols_per_sample"] = int(p.get("num_cols_per_sample", 15))
            p["num_rows_per_sample"] = int(p.get("num_rows_per_sample", 10000))
            p["max_parents_per_node"] = int(p.get("max_parents_per_node", 5))
            p["parent_frequency_cutoff"] = float(p.get("parent_frequency_cutoff", 0.5))
            p["max_workers"] = int(p.get("max_workers", 14))
            p["llm_max_tokens"] = int(p.get("llm_max_tokens", 4096))
            p["train_timeout_seconds"] = int(p.get("train_timeout_seconds", DEFAULT_TRAIN_TIMEOUT_SECONDS))
            p["verbose"] = bool(p.get("verbose", False))
            raw_temp = p.get("llm_temperature", None)
            p["llm_temperature"] = float(raw_temp) if raw_temp is not None else None
            # Causal-learn defaults
            p.setdefault("score_method", "bdeu")
            p.setdefault("structure_algorithm", "ges")
            p.setdefault("sample_batch_size", 10)

            # Per-dataset trial count (dataset override > run override > global default)
            dataset_num_trials = int(p.get("num_trials", num_trials))

            p["variable_descriptions"] = load_variable_descriptions(p.get("variable_description_path"))

            print(f"--- Dataset: {name} (run: {run_name}, {dataset_num_trials} trial(s)) ---")
            print(f"  CSV:    {csv_path}")

            if not os.path.exists(csv_path):
                print(f"*** CSV not found: {csv_path} — skipping ***", file=sys.stderr)
                overall_failed.append(f"{run_name}/{name} (csv not found)")
                continue

            df = pd.read_csv(csv_path, keep_default_na=False)
            allowed_models = p.get("models")
            if allowed_models:
                unknown_models = sorted(
                    set(allowed_models) - {model_id for model_id, *_ in MODEL_DEFS}
                )
                if unknown_models:
                    raise ValueError(
                        "Unknown model ID(s): "
                        + ", ".join(unknown_models)
                        + ". See MODEL_DEFS for valid IDs."
                    )

            for trial in range(dataset_num_trials):
                # Output directory — include trial subfolder when running multiple trials
                if len(runs) > 1:
                    base_dataset_dir = os.path.join(overall_log_dir, run_name, name)
                else:
                    base_dataset_dir = os.path.join(overall_log_dir, name)
                os.makedirs(base_dataset_dir, exist_ok=True)
                dataset_dir = base_dataset_dir
                if dataset_num_trials > 1:
                    dataset_dir = os.path.join(dataset_dir, f"trial_{trial}")
                os.makedirs(dataset_dir, exist_ok=True)

                if bool(p.get("reuse_candidate_edges", True)):
                    p['candidate_edges_path'] = os.path.join(dataset_dir, "candidate_edges.pkl")
                else:
                    p['candidate_edges_path'] = None

                # LLM response cache: shared across models and trials of this (run, dataset).
                if bool(p.get("use_llm_cache", True)):
                    p['llm_cache_path'] = os.path.join(base_dataset_dir, "llm_cache.sqlite")
                else:
                    p['llm_cache_path'] = None

                if dataset_num_trials > 1:
                    print(f"\n  === Trial {trial} ===")
                print(f"  Output: {dataset_dir}")

                failed_models = []

                all_model_defs = MODEL_DEFS
                for model_id, display_name, subdir, trainer_fn in all_model_defs:
                    if allowed_models and model_id not in allowed_models:
                        continue
                    output_dir = os.path.join(dataset_dir, subdir)
                    os.makedirs(output_dir, exist_ok=True)
                    setup_logging(output_dir)

                    print(f"\n  Training {display_name}...")
                    try:
                        with train_timeout(p["train_timeout_seconds"]):
                            model = trainer_fn(df, output_dir, p)
                        save_config(output_dir, {**{k: v for k, v in p.items()
                                                    if k not in ("variable_descriptions",)},
                                                 "data_shape": list(df.shape),
                                                 "trial": trial,
                                                 "timestamp": datetime.now().isoformat()})
                        evaluate_and_save(model, p, output_dir)
                        print(f"  {display_name} completed successfully")
                    except Exception:
                        traceback.print_exc()
                        print(f"*** {display_name} FAILED ***", file=sys.stderr)
                        failed_models.append(display_name)

                trial_label = f"{run_name}/{name}/trial_{trial}" if dataset_num_trials > 1 else f"{run_name}/{name}"
                if failed_models:
                    print(f"\n*** [{trial_label}] Failed: {', '.join(failed_models)} ***")
                    overall_failed.extend(f"{trial_label}/{m}" for m in failed_models)
                else:
                    print(f"\n  [{trial_label}] All models completed successfully.")

            # Aggregate metrics across trials
            if dataset_num_trials > 1:
                if len(runs) > 1:
                    base_dataset_dir = os.path.join(overall_log_dir, run_name, name)
                else:
                    base_dataset_dir = os.path.join(overall_log_dir, name)
                trial_dirs = [os.path.join(base_dataset_dir, f"trial_{t}") for t in range(dataset_num_trials)]

                for model_id, display_name, subdir, _ in MODEL_DEFS:
                    if allowed_models and model_id not in allowed_models:
                        continue
                    variance_result = evaluate_model_with_variance(trial_dirs, subdir)
                    if variance_result is not None:
                        out_path = os.path.join(base_dataset_dir, subdir)
                        os.makedirs(out_path, exist_ok=True)
                        with open(os.path.join(out_path, "evaluation_with_variance.json"), "w") as f:
                            json.dump(variance_result, f, indent=2, default=str)
                        print(f"  Saved variance report: {out_path}/evaluation_with_variance.json")
            print()

    print("=" * 60)
    print(f"All done. Outputs: {overall_log_dir}")
    if overall_failed:
        print("\n*** FAILED: ***")
        for m in overall_failed:
            print(f"  - {m}")

    return overall_failed


def main():
    parser = argparse.ArgumentParser(description="Train all model types across datasets")
    parser.add_argument("config_paths", nargs="+",
                        help="One or more JSON config files (run sequentially, each gets its own timestamp)")
    args = parser.parse_args()

    all_failed = {}
    for config_path in args.config_paths:
        print(f"\n{'#' * 60}")
        print(f"# Config: {config_path}")
        print(f"{'#' * 60}\n")
        failed = run_config(config_path)
        if failed:
            all_failed[config_path] = failed

    if all_failed:
        print(f"\n{'#' * 60}")
        print("# SUMMARY — FAILURES ACROSS ALL CONFIGS:")
        for cfg, failures in all_failed.items():
            print(f"#  {cfg}:")
            for m in failures:
                print(f"#    - {m}")
        print(f"{'#' * 60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
