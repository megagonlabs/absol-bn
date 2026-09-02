#!/usr/bin/env python3
"""
Generate a set of prediction questions from a ground-truth Bayesian network and synthetic test data.

For each row in the test set, picks a random target variable and a random subset
of other variables as context. The ground-truth BN predicts the target given the
context. The number of context variables is drawn uniformly from [0, num_other_nodes].

Example usage:
    python scripts/generate_ground_truth_predictions.py \
        data/hepar2/model.bif \
        data/hepar2/synthetic/test.csv \
        --output data/hepar2/synthetic/prediction_questions.json \
        --num_questions 1000 \
        --seed 42
"""

import argparse
import gc
import json
import logging
import random
import sys
from tqdm import tqdm
import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pgmpy")


GC_INTERVAL = 500
BATCH_SIZE = 1000


def generate_questions(gt_model, test_df, num_questions, rng, output_path):
    """Generate prediction questions from the ground-truth BN and test data.

    For each question:
      1. Sample a row from test_df.
      2. Pick a random target variable.
      3. Pick a random number of context variables (uniform over [0, n-1]).
      4. Use the ground-truth BN to predict the target given the context.

    Results are written to disk in batches to limit memory usage.
    """
    gt_nodes = sorted(gt_model.nodes())
    available_nodes = sorted(set(gt_nodes) & set(test_df.columns))

    inference_engine = VariableElimination(gt_model)

    batch = []
    total_written = 0

    # Write opening bracket, then append batches as we go
    with open(output_path, "w") as f:
        f.write("[\n")

    for i in tqdm(range(num_questions)):
        row_idx = rng.randint(0, len(test_df) - 1)
        row = test_df.iloc[row_idx]

        target = rng.choice(available_nodes)
        other_nodes = [n for n in available_nodes if n != target]

        # Uniform over [0, len(other_nodes)] context variables
        num_context = rng.randint(0, len(other_nodes))
        context_nodes = rng.sample(other_nodes, num_context)
        context = {node: row[node] for node in context_nodes}

        try:
            result = inference_engine.query([target], evidence=context if context else {})
            predicted_value = str(result.state_names[target][result.values.argmax()])
        except Exception as e:
            logging.warning(f"Prediction failed for target={target}, context keys={list(context.keys())}: {e}")
            continue

        batch.append({
            "target_node": target,
            "context": context,
            "ground_truth_prediction": predicted_value,
            "synthetic_data_value": row[target],
        })

        if len(batch) >= BATCH_SIZE:
            _flush_batch(output_path, batch, total_written)
            total_written += len(batch)
            batch = []
            gc.collect()
        elif (i + 1) % GC_INTERVAL == 0:
            gc.collect()

    # Flush remaining
    if batch:
        _flush_batch(output_path, batch, total_written)
        total_written += len(batch)

    # Close the JSON array
    with open(output_path, "a") as f:
        f.write("\n]")

    return total_written


def _flush_batch(output_path, batch, previously_written):
    """Append a batch of questions to the output JSON file."""
    with open(output_path, "a") as f:
        for j, item in enumerate(batch):
            if previously_written > 0 or j > 0:
                f.write(",\n")
            json.dump(item, f, indent=2)
    logging.info(f"Flushed batch — {previously_written + len(batch)} questions written so far")


def main():
    parser = argparse.ArgumentParser(
        description="Generate prediction questions from a ground-truth BN and test data"
    )
    parser.add_argument("bif_path", type=str, help="Path to ground-truth BN in BIF format")
    parser.add_argument("test_csv", type=str, help="Path to synthetic test CSV file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: same dir as test CSV)")
    parser.add_argument("--num_questions", type=int, default=1000,
                        help="Number of questions to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.output is None:
        import os
        args.output = os.path.join(os.path.dirname(args.test_csv), "prediction_questions.json")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    rng = random.Random(args.seed)

    logging.info(f"Loading ground-truth BN from {args.bif_path}")
    reader = BIFReader(args.bif_path)
    gt_model = reader.get_model()
    logging.info(f"Ground-truth BN has {len(gt_model.nodes())} nodes and {len(gt_model.edges())} edges")

    logging.info(f"Loading test data from {args.test_csv}")
    test_df = pd.read_csv(args.test_csv, keep_default_na=False)
    test_df = test_df.astype(str)
    logging.info(f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    logging.info(f"Generating {args.num_questions} questions...")
    total = generate_questions(gt_model, test_df, args.num_questions, rng, args.output)
    logging.info(f"Saved {total} questions to {args.output}")


if __name__ == "__main__":
    main()
