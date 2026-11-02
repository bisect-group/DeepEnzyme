#!/usr/bin/env bash
# Runs the 21 default kcat runs (7 splits x seeds 666 777 888) with packed_trainer.py, bf16, batch 512, lr 0.001.
# 4 worker slots: 2 on GPU 1 and 2 on GPU 3. A slot takes the next run from the list under a lock. Runs with completed.json are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
EM=/home/adhil/github/EMULaToR
OUT=$EM/outputs/baselines/DeepEnzyme
PY=/home/adhil/.conda/envs/mldb/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLITS=(random_splits_grouped_sequence/all random_splits_grouped_smiles/all uniprot_time_splits/all enzyme_sequence_splits/threshold_0.01 enzyme_structure_splits/threshold_0.625 substrate_splits/threshold_0.505 conformer_cosine_splits/threshold_0.996)
JOBS=()
for seed in 666 777 888; do for s in "${SPLITS[@]}"; do JOBS+=("$s $seed"); done; done
NEXT=$(mktemp -p "${TMPDIR:-/tmp}" packed_queue.XXXXXX)
echo 0 >"$NEXT"
LOCK=$NEXT.lock

worker() {
  local gpu=$1 i job split seed run log rc
  while :; do
    i=$(flock "$LOCK" bash -c 'i=$(<"$1"); echo $((i+1)) >"$1"; echo $i' _ "$NEXT")
    [[ $i -lt ${#JOBS[@]} ]] || break
    read -r split seed <<<"${JOBS[$i]}"
    run=$OUT/kcat/$split/seed_$seed
    [[ -e $run/completed.json ]] && { echo "$(date '+%F %T') skip $split seed $seed"; continue; }
    mkdir -p "$run"
    log=$OUT/logs/packed_$(basename "$(dirname "$split")")_$(basename "$split")_$seed.log
    echo "$(date '+%F %T') start gpu $gpu $split seed $seed -> $log"
    CUDA_VISIBLE_DEVICES=$gpu $PY emulator_bench/packed_trainer.py --train_dir $OUT/kcat/features/$split/train --val_dir $OUT/kcat/features/$split/val \
      --test_dir $OUT/kcat/features/$split/test --out_dir "$run" --dict_dir Data/Input --device cuda:0 --seed "$seed" \
      --batch_size 512 --lr 0.001 --bf16 --fused_adamw >>"$log" 2>&1
    rc=$?
    echo "$(date '+%F %T') end gpu $gpu $split seed $seed exit $rc"
  done
}

for gpu in 1 1 3 3; do worker $gpu & done
wait
echo "queue done"
