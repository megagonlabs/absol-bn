#!/usr/bin/env python3
"""Sample reproducible train/test tables from a discrete Bayesian network.

The source can be either a local BIF file or the name of a model bundled with
pgmpy (for example insurance, hepar2, diabetes, or munin).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pgmpy.readwrite import BIFReader, BIFWriter
from pgmpy.utils import get_example_model
from sklearn.model_selection import train_test_split


def load_model(source: str):
    """Return (model, default output directory, source-is-local)."""
    source_path = Path(source)
    if source_path.is_file():
        return BIFReader(str(source_path)).get_model(), source_path.parent, True

    model = get_example_model(source)
    return model, Path("data") / source, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="local .bif path or pgmpy example-model name (for example 'insurance')",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        help="dataset directory (default: the BIF's directory or data/<model-name>)",
    )
    parser.add_argument(
        "--n-samples",
        "--n_samples",
        dest="n_samples",
        type=int,
        default=1_000_000,
        help="total rows to simulate (default: 1000000)",
    )
    parser.add_argument(
        "--train-size",
        "--train_size",
        dest="train_size",
        type=float,
        default=0.8,
        help="fraction of rows used for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        "--random_state",
        dest="seed",
        type=int,
        default=42,
        help="sampling and split seed (default: 42)",
    )
    args = parser.parse_args()

    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    if not 0.0 < args.train_size < 1.0:
        parser.error("--train-size must be between 0 and 1")

    print(f"Loading model '{args.source}'...")
    model, default_output_dir, source_is_local = load_model(args.source)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {len(model.nodes())} nodes, {len(model.edges())} edges")

    print(f"Simulating {args.n_samples:,} rows with seed {args.seed}...")
    samples = model.simulate(args.n_samples, seed=args.seed)
    # pgmpy can expose node order through hash-dependent graph iteration. A
    # stable column order keeps seeded generation identical across processes.
    samples = samples.reindex(sorted(samples.columns), axis=1)
    train_df, test_df = train_test_split(
        samples,
        train_size=args.train_size,
        random_state=args.seed,
    )

    train_path = synthetic_dir / "train.csv"
    test_path = synthetic_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved {len(train_df):,} train rows -> {train_path}")
    print(f"Saved {len(test_df):,} test rows  -> {test_path}")

    # Named pgmpy sources do not have a local BIF path, so preserve the exact
    # network used for sampling. Never overwrite an existing reference model.
    bif_path = output_dir / "model.bif"
    if not source_is_local and not bif_path.exists():
        BIFWriter(model).write_bif(filename=str(bif_path))
        print(f"Saved model -> {bif_path}")


if __name__ == "__main__":
    main()
