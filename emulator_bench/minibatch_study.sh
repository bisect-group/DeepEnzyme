#!/usr/bin/env bash
# Usage: minibatch_study.sh b1_subset | mb_subset B LR | expA B LR EPOCHS SPLIT. GPU 3 only. Logs go to $S/logs.
set -uo pipefail
cd "$(dirname "$0")/.."
EM=/home/adhil/github/EMULaToR
OUT=$EM/outputs/baselines/DeepEnzyme
S=$OUT/kcat_minibatch_study
PY=/home/adhil/.conda/envs/mldb/bin/python
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$S/logs"
case $1 in
  b1_subset)
    jobs=(); for s in 666 777 888; do jobs+=(--job $S/subset1500/train $S/subset1500/val $S/subset1500/test $S/subset1500_B1/seed_$s $s); done
    $PY emulator_bench/run_b1_subset.py --device cuda:0 --eval_every 20 "${jobs[@]}" >>"$S/logs/b1_subset.log" 2>&1 ;;
  mb_subset)
    B=$2; LR=$3
    for s in 666 777 888; do
      $PY emulator_bench/minibatch_trainer.py --train_dir $S/subset1500/train --val_dir $S/subset1500/val --test_dir $S/subset1500/test \
        --out_dir $S/subset1500_B${B}_lr$LR/seed_$s --seed $s --batch_size $B --lr $LR --eval_every 20 --fused_adamw --device cuda:0 >>"$S/logs/mb_subset_B${B}_lr$LR.log" 2>&1
    done ;;
  expA)  # one split; a rerun with more epochs resumes from the checkpoint
    B=$2; LR=$3; EP=$4
    for sp in $5; do
      F=$OUT/kcat/features/$sp
      $PY emulator_bench/minibatch_trainer.py --train_dir $F/train --val_dir $F/val --test_dir $F/test --out_dir $S/expA_B${B}_lr$LR/$sp/seed_666 \
        --seed 666 --batch_size $B --lr $LR --iteration $EP --fused_adamw --device cuda:0 >>"$S/logs/expA_B${B}_lr${LR}_$(tr / _ <<<"$sp").log" 2>&1
    done ;;
esac
