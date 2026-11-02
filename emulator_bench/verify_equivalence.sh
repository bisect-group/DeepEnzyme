#!/usr/bin/env bash
# Checks that the DeepEnzyme wrapper matches the original DeepEnzyme code. Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: mldb (torch, rdkit, biotite, pandas, scipy, scikit-learn, pytest). The script does not create envs or install packages.
# CPU mode: synthetic fixtures built in a temporary folder, the public dictionaries and example structure under Data/Input.
#   It compares the wrapper with Code/Model and Code/Example on the CPU. No private data and no downloads. Any skipped test fails the run.
# --gpu mode: also runs the GPU tests on the visible CUDA device (bit-exact checks, original loop, bf16 against fp32).
#   The precision tests (bf16, fp16, TF32, fused AdamW) pass when the gap to fp32 is at most K = 10 times a noise floor, the effect of
#   one rounding of the initial weights at the production precision. Rejected candidates are xfail (strict). The ratios are printed.
#   Without a CUDA device the GPU tests skip and the script says so.
# --negative-control mode: runs only the precision tests with the production weights multiplied by 1.05. It exits 0 only if every
#   test fails, which shows that the bound can detect a real difference.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
pytest_args=(-q -rs -p no:cacheprovider --color=no -o addopts="")
out=$(CUDA_VISIBLE_DEVICES="" python -m pytest emulator_bench/tests "${pytest_args[@]}" -m "not gpu" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
if [[ ${1:-} == --gpu ]]; then
    gpu_out=$(python -m pytest emulator_bench/tests "${pytest_args[@]}" -s -m gpu 2>&1) || { echo "$gpu_out"; exit 1; }
    echo "$gpu_out"
    if grep -Eq "[0-9]+ skipped" <<<"$gpu_out"; then
        echo "NOTE: GPU tests were skipped because no CUDA device is visible"
    fi
fi
if [[ ${1:-} == --negative-control ]]; then
    neg_out=$(EQUIV_NEGATIVE_CONTROL=1 python -m pytest emulator_bench/tests "${pytest_args[@]}" -s -m noise 2>&1) || true
    grep -E "^(bf16|fp16|tf32|fused_adamw|packed_fp32|packed_bf16):" <<<"$neg_out" || true
    if grep -Eq "[0-9]+ passed" <<<"$neg_out" || ! grep -Eq "[0-9]+ failed" <<<"$neg_out"; then
        echo "$neg_out" | tail -5
        echo "FAIL: a precision test passed with the production weights changed, so its bound is vacuous (or no test ran)"
        exit 1
    fi
    echo "OK: every precision test fails with the production weights changed"
fi
