"""
Aggregate DeepEnzyme EMULaToR results across seeds.

Walks DeepEnzyme result directories, reads final_results_{train,val,test}.csv
for every complete seed run, and writes mean + std per metric per TVT
split.

Directory layouts handled:
  <results_dir>/<split_group>/threshold_X/seed_N
  <results_dir>/<split_group>/all/seed_N
where <results_dir> is <work_dir>/<task>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_WORK_DIR


SPLITS = ("train", "val", "test")
METRIC_ALIASES = {
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R2",
    "pcc": "PCC",
    "scc": "SCC",
    "loss": "Loss",
}


def parse_path(seed_dir: Path, results_dir: Path) -> dict[str, str | None]:
    """Extract split metadata from a seed_* path under the results root."""
    parts = seed_dir.relative_to(results_dir).parts
    if len(parts) < 3:
        raise ValueError(f"Expected <split_group>/<split>/<seed>, got {seed_dir}")

    split_group = parts[0]
    split_name = parts[-2]
    threshold = split_name if split_name.startswith("threshold_") else None
    return {
        "split_group": split_group,
        "threshold": threshold,
        "split_name": split_name,
        "seed": seed_dir.name,
    }


def load_seed_results(seed_dir: Path) -> dict[str, pd.Series] | None:
    """Return {split: metrics_series} for one seed dir, or None if incomplete."""
    results = {}
    for split in SPLITS:
        fpath = seed_dir / f"final_results_{split}.csv"
        if not fpath.exists():
            return None
        df = pd.read_csv(fpath)
        if df.empty:
            return None
        series = df.iloc[0].copy()

        pred_path = seed_dir / f"pred_label_{split}.csv"
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            if {"pred", "label"}.issubset(pred_df.columns) and len(pred_df) > 1:
                series["SCC"] = pred_df["pred"].corr(pred_df["label"], method="spearman")

        results[split] = series
    return results


def build_rows(results_dir: Path) -> list[dict]:
    rows = []
    for seed_dir in sorted(results_dir.rglob("seed_*")):
        if not seed_dir.is_dir():
            continue

        try:
            meta = parse_path(seed_dir, results_dir)
        except ValueError as exc:
            print(f"  [skip] {exc}")
            continue

        split_results = load_seed_results(seed_dir)
        if split_results is None:
            print(f"  [skip] incomplete: {seed_dir.relative_to(results_dir)}")
            continue

        for tvt_split, series in split_results.items():
            row = {**meta, "tvt_split": tvt_split}
            for out_name, source_name in METRIC_ALIASES.items():
                if source_name in series.index:
                    row[out_name] = series[source_name]
            rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics = list(METRIC_ALIASES)
    group_keys = ["split_group", "threshold", "split_name", "tvt_split"]
    agg = df.groupby(group_keys, dropna=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = df.groupby(group_keys, dropna=False).size()
    return agg.reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_WORK_DIR / DEFAULT_BASE_DIR.name, help="<work_dir>/<task> written by run_split_benchmarks.py.")
    parser.add_argument("--output", type=Path, default=None, help="Default: <results_dir>/aggregate_deepenzyme_results.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output = (args.output or results_dir / "aggregate_deepenzyme_results.csv").expanduser().resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    rows = build_rows(results_dir)
    if not rows:
        print(f"No complete DeepEnzyme results found under {results_dir}")
        return

    summary = aggregate(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Saved {len(summary)} rows from {len(rows)} split result files to {output}")


if __name__ == "__main__":
    main()
