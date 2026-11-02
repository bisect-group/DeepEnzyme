#!/usr/bin/env bash
# Runs the 21 default kcat runs (7 splits x seeds 666 777 888) on one GPU with N concurrent single-run trainer processes.
# Usage: CUDA_VISIBLE_DEVICES=1 run_queue.sh [N=6]. Needs the feature folders (build_all_features.sh).
# The longest sequence (7,096 residues) needs 2.8 GB of GPU memory. Each process is capped at 14% of a 22 GB GPU
# (DEEPENZYME_GPU_FRACTION) with expandable segments, so 6 processes fit. A run that fails is tried again from its checkpoint, 3 times at most.
# A restart continues every unfinished run from its checkpoint_last.pt. A finished run is skipped by the trainer.
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:-6}
EM=/home/adhil/github/EMULaToR
OUT=$EM/outputs/baselines/DeepEnzyme
PY=/home/adhil/.conda/envs/mldb/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEEPENZYME_GPU_FRACTION=0.14
SPLITS=(random_splits_grouped_sequence/all random_splits_grouped_smiles/all uniprot_time_splits/all enzyme_sequence_splits/threshold_0.01 enzyme_structure_splits/threshold_0.625 substrate_splits/threshold_0.505 conformer_cosine_splits/threshold_0.996)
# Wave 1 first: seed 666 of every split and seed 777 of the first split. Then the other seeds.
JOBS=()
for s in "${SPLITS[@]}"; do JOBS+=("$s 666"); done
JOBS+=("${SPLITS[0]} 777")
for seed in 777 888; do
  for s in "${SPLITS[@]}"; do [[ "$s $seed" == "${SPLITS[0]} 777" ]] || JOBS+=("$s $seed"); done
done
for job in "${JOBS[@]}"; do
  read -r split seed <<<"$job"
  while [[ $(jobs -rp | wc -l) -ge $N ]]; do wait -n; done
  until [[ -e $OUT/kcat/features/$split/test/feature_summary.json && -e $OUT/kcat/features/$split/val/feature_summary.json && -e $OUT/kcat/features/$split/train/feature_summary.json ]]; do sleep 30; done
  run=$OUT/kcat/$split/seed_$seed
  mkdir -p "$run"
  log=$OUT/logs/$(tr / _ <<<"$split")_seed$seed.log
  echo "$(date '+%F %T') start $split seed $seed -> $log"
  (
    for attempt in 1 2 3; do
      $PY emulator_bench/run_capped.py emulator_bench/train_single_target_tvt.py --train_dir $OUT/kcat/features/$split/train --val_dir $OUT/kcat/features/$split/val \
        --test_dir $OUT/kcat/features/$split/test --out_dir "$run" --dict_dir Data/Input --device cuda:0 --seed "$seed" --fused_adamw \
        --hide_sample_progress --progress_interval 1000 --torch_num_threads 1 --torch_num_interop_threads 1 >>"$log" 2>&1 && break
      echo "$(date '+%F %T') attempt $attempt of $split seed $seed failed" >>"$log"
    done
  ) &
done
wait
echo "$(date '+%F %T') queue finished"
