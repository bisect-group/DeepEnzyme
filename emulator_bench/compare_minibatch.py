"""Minibatch against batch-1. Experiment A: per-epoch test metrics of seed 666 per split. Experiment B: last-epoch test metrics on the train subset, mean +- std over seeds."""
import sys
from pathlib import Path

import pandas as pd

OUT = Path("/home/adhil/github/EMULaToR/outputs/baselines/DeepEnzyme")
STUDY, METRICS = OUT / "kcat_minibatch_study", ["RMSE_test", "R2_test", "PCC_test"]
SPLITS = ["random_splits_grouped_sequence/all", "random_splits_grouped_smiles/all", "uniprot_time_splits/all", "enzyme_sequence_splits/threshold_0.01", "enzyme_structure_splits/threshold_0.625", "substrate_splits/threshold_0.505"]


def log(path):
    return pd.read_csv(path / "logfile.csv").set_index("Epoch") if (path / "logfile.csv").exists() else None


def experiment_a():
    for split in SPLITS:
        b1 = log(OUT / "kcat" / split / "seed_666")
        runs = {d.name.split("_lr")[1]: log(d / split / "seed_666") for d in sorted(STUDY.glob("expA_B*_lr*")) if (d / split / "seed_666" / "logfile.csv").exists()}
        if b1 is None or not runs:
            continue
        print(f"\n== A: {split} seed 666 (test RMSE R2 PCC; batch 1 vs B at lr)")
        for epoch in sorted(set().union(*(r.index for r in runs.values()))):
            cells = [f"b1 " + " ".join(f"{b1.loc[epoch, m]:.3f}" for m in METRICS) if epoch in b1.index else "b1 -"]
            cells += [f"lr{lr} " + " ".join(f"{r.loc[epoch, m]:.3f}" for m in METRICS) if epoch in r.index else f"lr{lr} -" for lr, r in runs.items()]
            print(f"epoch {epoch:3d} | " + " | ".join(cells))


def last(path_pattern):
    rows = [log(p) for p in sorted(STUDY.glob(path_pattern)) if (p / "completed.json").exists()]
    return pd.DataFrame([r.iloc[-1][METRICS] for r in rows if r is not None])


def experiment_b():
    base = last("subset1500_B1/seed_*")
    print(f"\n== B: train subset, last epoch, test (seeds: batch 1 n={len(base)})")
    if len(base):
        print("batch 1      " + "  ".join(f"{m} {base[m].mean():.4f}+-{base[m].std():.4f}" for m in METRICS))
    for d in sorted({p.parent.name for p in STUDY.glob("subset1500_B[0-9]*_lr*/seed_*") if p.parent.name != "subset1500_B1"}):
        mb = last(f"{d}/seed_*")
        if len(mb) and len(base):
            print(f"{d:22s} n={len(mb)} " + "  ".join(f"{m} {mb[m].mean():.4f}+-{mb[m].std():.4f} gap/std1 {(mb[m].mean() - base[m].mean()) / base[m].std():+.1f}" for m in METRICS))


if __name__ == "__main__":
    experiment_a()
    experiment_b()
