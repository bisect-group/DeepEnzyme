#!/usr/bin/env bash
# Builds the v6 feature folders of the seven default kcat splits, one split after the other (CPU, about 10 GB RAM peak).
set -euo pipefail
cd "$(dirname "$0")/.."
EM=/home/adhil/github/EMULaToR
OUT=$EM/outputs/baselines/DeepEnzyme
exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/adhil/.conda/envs/mldb/bin/python emulator_bench/cache_features.py \
  --base_dir $EM/data/processed/datasets/kinetic_dataset/kcat --manifests_dir $OUT/kcat/manifests \
  --features_dir $OUT/kcat/features --cache_dir $OUT/embeddings
