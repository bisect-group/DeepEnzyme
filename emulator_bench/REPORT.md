# DeepEnzyme wrapper report

## Method
DeepEnzyme predicts kcat from a substrate and an enzyme structure. It has three branches.
A GCN reads the substrate graph (Weisfeiler-Lehman fingerprints, radius 2). A GCN reads the protein contact map
(C-alpha distance under 10 A, from a structure). A transformer reads protein 4-gram tokens.
The three mean-pooled vectors are joined and pass through 3 linear layers and a final linear output.
Loss is MSE. Paper: `deepenzyme.pdf` (10 pages). The paper does not list
training hyperparameters. The code is the source of truth.
Tasks in the paper (ablations.tex): kcat only.

## Original default settings
| Setting | Value | Source |
|---|---|---|
| lr | 0.001 | Code/Model/train.py:158 |
| epochs | 200 | train.py:159 |
| optimiser | AdamW(lr), no weight_decay argument, so torch default 0.01 | train.py:54 (`weight_decay=1e-6` at :160 is never used) |
| scheduler | StepLR(step 3, gamma 0.9); step only if epoch>=10 and epoch%10>0 | train.py:55, :211-212 |
| dropout | 0.3 | train.py:161 |
| dim / hidden1 / hidden2 | 64 / 64 / 64 | train.py:162-165 |
| layer_output | 3 | train.py |
| heads / hid_size / transformer layers | 4 / 64 / 3 | train.py |
| batch size | 1 (per-sample updates); the data is shuffled once with seed 666 and keeps that order in every epoch | train.py:34-37, :183; the shuffle inside the loop is commented out (:59) |
| loss | MSE | train.py:56, :92 |
| early stopping | none | train.py |
| checkpoint | model weights overwritten each epoch, so the final model is the last epoch | train.py:222 |
| precision | fp32 | train.py |
| target | log2(kcat) (`logkcat_0612.npy`) | train.py:176 |
| split | random 0.8/0.1/0.1, seed 666 | train.py:183-186 |

Paper and code: the paper has no hyperparameter table, so there is no conflict. The old wrapper differed from
the code in three places (see below).

## What the wrapper does now
`run_split_benchmarks.py` (defaults: `--base_dir` kcat task dir, `--work_dir` `EMULaToR/outputs/baselines/DeepEnzyme`, called `OUT`) runs, for every split folder found on disk:
`prepare_splits.py` (structure per row), `cache_features.py` (static features, persistent cache),
then `train_single_target_tvt.py` for each seed (default 666 777 888).
Split folders: random_splits_grouped_sequence, random_splits_grouped_smiles, uniprot_time_splits,
enzyme_sequence_splits/threshold_0.01, enzyme_structure_splits/threshold_0.625, substrate_splits/threshold_0.505, conformer_cosine_splits/threshold_0.996 (thresholds found on disk). 7 jobs, 21 runs.
All outputs go under `--work_dir`: runs in `OUT/<task>/<split_group>/<split_name>/seed_N`, feature cache in `OUT/embeddings/features_v6`. Nothing is written under `data/`. The model trains on log2(kcat); metrics and predictions are in log10.
Test metrics come from the last-epoch model.

## Files removed
`tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `run_from_config.py`
and `benchmark_config.example.json` (both used grid/Optuna-style keys and stale argument names),
`build_tvt_features.py` (unused CSV path, no caller). In `../Plan.md` the two Optuna lines were removed.

## Files changed
- `common.py`: removed default data/result paths and unused constants. Default seeds 666 777 888.
  Default split groups: the seven groups above.
- `run_split_benchmarks.py`: `--base_dir` and `--work_dir` have EMULaToR defaults. Removed `_load_hparams` and `--hparams_json`
  (no tuned settings). Removed fixed GPU default 2. `--amp` opt-in. `--iteration` stays as a smoke-test override.
- `train_single_target_tvt.py`: weight decay default 0.01 (original AdamW default). Final model is the last epoch
  (best-validation checkpoint removed). fp32 by default (`--amp` opt-in). TF32 switches removed.
  `run_summary.json` has `final_epoch` and `final_val_rmse`.
- `prepare_splits.py`, `structures.py`: one structure per unique sequence with the CatPred rule (see "Structure coverage"). Rows without a structure stay. Paths derive from the EMULaToR root, not a home dir.
- `cache_features.py`: path arguments are required. Default dictionary dir is `<submodule>/Data/Input`. The feature folder key holds the radius, n-gram size, distance threshold, dictionary hashes and structure rule version. Counts of dropped (by stage), full, partial, low-identity and no-structure rows go to `feature_summary.json`.
- `aggregate_deepenzyme_results.py`: `--results_dir` required. Output default `<results_dir>/aggregate_deepenzyme_results.csv`.
- `train_single_target_tvt.py` (later change): the train rows are shuffled once from the seed; loss and predictions stay on the device until the end of each pass. `modeling_fast.normalize_adj` scales rows and columns.
- `README.md`, `commands.txt`: rewritten for the EMULaToR layout.

## Bugs fixed
- ESM structure lookup used a folder and prefix that do not exist. All ESM-only rows were dropped.
- Wrapper weight decay was 1e-6. The original code trains with AdamW default 0.01.
- Wrapper selected the best-validation epoch. The original code keeps the last epoch.
- Wrapper used bf16/fp16 AMP and TF32. The original code uses fp32.
- Hardcoded home paths and default GPU 2 removed.

## Deviations
- Model code is `modeling_fast.py` (no `.cuda()`, `mm` instead of `spmm`). Math is the same. Reviewed line by line against `Code/Model`.
- Features are cached (sequence, SMILES, contact map). They are static inputs, so training is unchanged.
- Target scale: the model trains on log2(kcat) as the original does. The wrapper converts predictions and targets to log10 (times log10(2)) before it computes every reported metric and saves predictions, so metrics compare with the other baselines. `Loss` in the logs is the log2 training MSE. There is no option to report log2 metrics.
- Original protein adjacency was sparse; the wrapper uses dense arrays (same values).

## Structure coverage

Structure rule (the CatPred rule, `structures.py`, one structure per unique sequence): experimental mmCIF chain with the best `pdb_coverage`, else AlphaFold (newest version), else ESMFold2 (full id, then accession), else ESM3, else none. The candidates are the rows of the sequence in `kcat_kinetic_params_3d.parquet`. Rows without a structure stay in the data. Their protein map has self-loops only.

The contact map is built over the dataset sequence. The chain residues (C-alpha atoms of chain A, read by PDB fixed columns) are aligned to the sequence with a global alignment (`Bio.Align.PairwiseAligner`: match +2, mismatch -1, gap open -4, gap extend -1, free end gaps). Each aligned C-alpha goes to its matched position. Chain residues that do not align (tags, insertions) are cut. A position without an aligned C-alpha has a self-loop only. No row is dropped for its structure.

Counts on the real kcat splits (`prepare_splits.py` and the featurization code run on the CPU). "Substrate drop" is a row where the featurizer raises. "Rows" are the rows that remain. Sources are counted over the remaining rows: experimental / alphafold / esmfold2 / esm3 / none. "Full" is a row with a C-alpha atom at every sequence position. "Partial" is a row with a structure and fewer aligned positions. "Coverage" is the mean fraction of aligned positions over the rows with a structure. "Low id." is the number of rows with a structure where less than 90% of the aligned pairs are identical (kept).

| Split (kcat) | part | rows in | substrate drop | rows | sources | full | partial | coverage | low id. |
|---|---|---|---|---|---|---|---|---|---|
| random_splits_grouped_sequence | train | 50,289 | 0 | 50,289 | 31,348 / 18,400 / 414 / 91 / 36 | 25,582 | 24,671 | 0.959 | 7 |
|  | val | 2,794 | 0 | 2,794 | 1,846 / 916 / 29 / 3 / 0 | 1,323 | 1,471 | 0.960 | 1 |
|  | test | 2,795 | 0 | 2,795 | 1,844 / 932 / 15 / 3 / 1 | 1,414 | 1,380 | 0.960 | 0 |
| random_splits_grouped_smiles | train | 50,301 | 0 | 50,301 | 31,537 / 18,234 / 409 / 89 / 32 | 25,572 | 24,697 | 0.959 | 7 |
|  | val | 2,788 | 0 | 2,788 | 1,781 / 968 / 32 / 4 / 3 | 1,363 | 1,422 | 0.965 | 0 |
|  | test | 2,789 | 0 | 2,789 | 1,720 / 1,046 / 17 / 4 / 2 | 1,384 | 1,403 | 0.957 | 1 |
| uniprot_time_splits | train | 50,526 | 0 | 50,526 | 32,355 / 17,757 / 306 / 82 / 26 | 25,265 | 25,235 | 0.958 | 8 |
|  | val | 2,800 | 0 | 2,800 | 1,240 / 1,527 / 18 / 12 / 3 | 1,705 | 1,092 | 0.977 | 0 |
|  | test | 2,552 | 0 | 2,552 | 1,443 / 964 / 134 / 3 / 8 | 1,349 | 1,195 | 0.963 | 0 |
| enzyme_sequence_splits/threshold_0.01 | train | 50,293 | 0 | 50,293 | 32,237 / 17,631 / 331 / 60 / 34 | 25,077 | 25,182 | 0.961 | 4 |
|  | val | 2,793 | 0 | 2,793 | 1,449 / 1,288 / 55 / 1 / 0 | 1,610 | 1,183 | 0.952 | 1 |
|  | test | 2,792 | 0 | 2,792 | 1,352 / 1,329 / 72 / 36 / 3 | 1,632 | 1,157 | 0.931 | 3 |
| enzyme_structure_splits/threshold_0.625 | train | 50,308 | 0 | 50,308 | 30,796 / 18,952 / 426 / 97 / 37 | 26,213 | 24,058 | 0.964 | 4 |
|  | val | 2,793 | 0 | 2,793 | 2,046 / 727 / 20 / 0 / 0 | 1,215 | 1,578 | 0.936 | 1 |
|  | test | 2,777 | 0 | 2,777 | 2,196 / 569 / 12 / 0 / 0 | 891 | 1,886 | 0.908 | 3 |
| substrate_splits/threshold_0.505 | train | 50,307 | 0 | 50,307 | 31,370 / 18,372 / 437 / 93 / 35 | 25,648 | 24,624 | 0.958 | 7 |
|  | val | 2,793 | 0 | 2,793 | 1,814 / 967 / 9 / 3 / 0 | 1,359 | 1,434 | 0.965 | 1 |
|  | test | 2,778 | 0 | 2,778 | 1,854 / 909 / 12 / 1 / 2 | 1,312 | 1,464 | 0.972 | 0 |
| conformer_cosine_splits/threshold_0.996 | train | 50,292 | 0 | 50,292 | 31,392 / 18,337 / 432 / 95 / 36 | 25,557 | 24,699 | 0.958 | 7 |
|  | val | 2,793 | 0 | 2,793 | 1,849 / 935 / 7 / 1 / 1 | 1,388 | 1,404 | 0.975 | 1 |
|  | test | 2,793 | 0 | 2,793 | 1,797 / 976 / 19 / 1 / 0 | 1,374 | 1,419 | 0.971 | 0 |

Notes:
- No row is dropped for its substrate. The 107 unique SMILES that make the original featurizer raise now featurize (see "Featurization and dtype rules").
- About half of the rows with a structure are partial. Their unresolved residues have self-loops only. Experimental tags are cut.
- Rows resolved as ESMFold2, ESM3 or AlphaFold use a predicted wild-type structure, like the original DeepEnzyme.
- No row has a non-positive `value`.

## Changes after review

- Target: train on log2(kcat); metrics and predictions in log10. `cache_features.py` builds the label from `value` (`log2_value`) and marks the feature folder; old feature folders with another label are rebuilt. The `--target_transform` options were removed from `cache_features.py` and `run_split_benchmarks.py`.
- Summaries report std, not variance: `run_split_benchmarks.summarize` (`*_mean`, `*_std`) and `aggregate_deepenzyme_results.py` (both used variance).
- `--base_dir` and `--work_dir` have defaults (kcat task dir; `EMULaToR/outputs/baselines/DeepEnzyme`, computed from the file location). Paths are resolved to absolute before the subprocesses run in the submodule root. The aggregate script default `--results_dir` is `<default work_dir>/<task>` and it adds the repo root to `sys.path`.
- README and `commands.txt` updated (no `runs/kcat` path, no variance, no `--target_transform`).
- Stale-cache guards: prepared manifests carry a fingerprint of the structure rule version, the source file and the chosen structure paths. Feature folders store the label, the manifest size and mtime, and a configuration fingerprint (radius, n-gram size, distance threshold, dictionary hashes, structure rule version). A rerun of the same command skips both stages.

## Commands
Run from `EMULaToR/baselines/DeepEnzyme`, with `PY=/home/adhil/.conda/envs/mldb/bin/python`:

```
PY=/home/adhil/.conda/envs/mldb/bin/python
$PY emulator_bench/run_split_benchmarks.py --seeds 666 777 888 --gpus 0 1 --runs_per_gpu 2
$PY emulator_bench/aggregate_deepenzyme_results.py
```
Only kcat applies. km, ki, ic50, kd and ec50 are not DeepEnzyme tasks in the paper.

## Open issues
- Last-epoch model versus best-validation. The original saves the last epoch, so the wrapper does too. The
  paper appendix says validation-based checkpointing where the original has it. DeepEnzyme has none.
- Precision. The camera-ready appendix says bf16 for kinetic baselines. The original code is fp32, so the
  default is fp32. Use `--amp` to match the appendix, at some risk to the original numerics.
- Speed. Training is per-sample (batch size 1, 200 epochs, test evaluated every epoch as in the original).
  A full kcat run (47k train rows) is slow. Only exact speed-ups are applied (see "Speed-ups"); none changes batch behaviour.

## Featurization and dtype rules

The wrapper copies the original featurization and arithmetic. Where it differed, it now follows the original.

- Contact map dtype. A map that covers every sequence position is an integer map (int32), as the original equal-length map; it becomes float32 in `adj + torch.eye`. A map with missing positions is float64 with a unit diagonal, like the original padded map, and is normalised in float64 (`GCN.py:24-26`, `example.py:141`). The model casts the normalised matrix to float32 before `mm`, as the original does. For chains that equal a prefix of the sequence, the map matches the original bit for bit.
- Fingerprints. `extract_fingerprints` follows the original for every molecule that the original featurizes; the result is bit-identical on all of them (9,316 unique SMILES tested). Three repairs cover the 107 SMILES where the original raises. (1) An atom or bond type that is not in the dictionary gets a fresh id, `len(table)`. The id is not added to the dictionary. Every fingerprint that contains it is unseen, so it maps to id 0, the same as any unseen fingerprint in the original. A single-atom fingerprint that is not in the dictionary also maps to 0. (2) An atom without a bond gets an empty entry in the bond dictionary, so the fingerprint has one entry per atom. (3) The bond dictionary follows the atom order when a molecule has atoms without a bond. All 107 SMILES featurize and the fingerprint length equals the atom count.
- SMILES cache key. The atom order depends on the input string, so the cache key uses the raw SMILES, not the canonical SMILES.
- Structures. The chain is aligned to the dataset sequence (see "Structure coverage"). A row without a structure gets a map with self-loops only. The original reader does not read residues numbered 1000 and above; the wrapper reads them.
- Dropped rows. No row is dropped for its substrate. `cache_features.py` still fails a row (`failed_rows.csv`, `rows_failed` in `feature_summary.json`, `rows_dropped` per split in `run_summary.json`) if the SMILES is invalid or the fingerprint length differs from the atom count. `failed_rows.csv` has a `stage` column and `run_summary.json` has `failed_by_stage`. On the real kcat splits both are 0. Rows without a structure are kept.
- Cache version. `CACHE_VERSION` is `v6`. It is in every cache key and in `feature_summary.json`, so old caches and feature folders are rebuilt. The sequence and SMILES keys hold the dictionary hashes. A feature folder is rebuilt when the radius, n-gram size, distance threshold, a dictionary file or the structure rule version changes.
- The wrapper reads PDB chain A only, as the original. Experimental chains are extracted from mmCIF to chain A files first.

## Equivalence check

- 96-row run (kcat, random_splits_grouped_sequence, seed 666, CPU, wrapper CLI with default flags) against the original functions and classes with a minimal driver. 3 and 13 epochs: initial and final weights differ by 0, test predictions by 4.4e-16 (log2), epoch metrics by at most 1.8e-15.
- 9,316 unique real kcat SMILES: 9,209 work in the original and the wrapper gives the same fingerprints and adjacency (0 differences). The other 107 raise in the original; the wrapper featurizes all of them with the repairs above, and each fingerprint length equals the atom count.
- 270 PDB rows (chain A), checked before the alignment change: the original raised on 4 and 0 contact maps differed. The wrapper now differs from the original on purpose where a chain has gaps, tags or residues numbered 1000 and above. Tests cover the equal and prefix cases against the original and the other cases against hand-built maps.

## Speed-ups

All are exact. `verify_equivalence.sh` tests them (see `README.md`).
- `normalize_adj` scales rows and columns (`adj * r[:, None] * r[None, :]`) instead of two dense `L x L` products with a diagonal matrix. The products only multiply by zeros off the diagonal. Result: bit-exact against the original function (float32 and float64, sizes 1, 7, 120, 2000, and a matrix with a zero row).
- Loss and predictions of a pass stay on the device and move to the host once. The loss is summed in order on the host, so logged values are the same bits. Labels are read once per split. On CUDA the per-sample adjacency is copied from pinned memory with `non_blocking=True`.
- Not changed: batch size 1, no padding, `smiles_transformer` stays (its weights draw from the random generator), the order of module construction.

## Equivalence tests (CPU)

`emulator_bench/verify_equivalence.sh`: 48 tests pass in about 17 s, 0 skipped. 7 GPU tests are written and were not run. They are the bit-exact `normalize_adj` and forward on CUDA, the training loop against the original on CUDA (loss 1e-5 relative), and bf16 against fp32.
- The original `train.py` does not run as is: it passes the whole sample tuple (with the label) to `forward`, which unpacks four values, and it calls `float()` on one-element arrays, which fails on numpy 2. The test uses its `Trainer`, `Validater` and `Tester` with a model subclass that drops the label and returns a scalar.
- 14 epochs on 6 train rows (batch size 1, dropout 0.3, shuffled once, StepLR from epoch 11): train and validation losses and learning rates match the original loop exactly. The final weights match exactly. The saved test predictions are the original predictions times log10(2).
- A run interrupted after epoch 3 and resumed gives the same log, weights and predictions as an uninterrupted run.


## GPU speed audit (local A10, 31 Oct 2026)

Setup: random_splits_grouped_sequence features (train 50,289 / val 2,794 / test 2,795 rows, all kept, 0 substrate or structure drops), one A10, `CUDA_VISIBLE_DEVICES=1`. The features took 6 min 16 s on the CPU (peak 10.5 GB RAM) and use 5.4 GB on disk (`OUT/kcat/features`), plus 0.3 GB of cache (`OUT/embeddings`). A training process holds 6.4 to 6.6 GB RAM (all features in memory), 1.5 to 1.9 GB GPU memory, and one CPU core at 99 to 100%. The model is launch-bound on the CPU, so the GPU is not the limit.

Times come from the production `run_pass` on real rows (300 timed train steps and 300 eval steps after warm-up, one process). Epoch = 50,289 train steps + 5,589 eval steps (val and test). Projection = 200 epochs.

| Config | train ms/step | eval ms/step | min/epoch | h/run | h for 21 runs, one at a time |
|---|---|---|---|---|---|
| fp32 (original) | 26.0 | 3.4 | 22.1 | 73.7 | 1,548 |
| TF32 | 28.3 | 3.2 | 24.0 | 80.1 | 1,681 |
| bf16 autocast | 31.7 | 4.4 | 27.0 | 89.9 | 1,889 |
| fp16 autocast + GradScaler | 35.3 | 4.3 | 30.0 | 100.0 | 2,099 |
| fused AdamW (fp32) | 25.6 | 3.9 | 21.8 | 72.7 | 1,527 |
| bf16 + fused AdamW | 34.1 | 3.7 | 28.9 | 96.4 | 2,025 |

Concurrent runs on one GPU (1,500 timed steps, the timing windows aligned):

| Config | runs per GPU | train ms/step (mean) | aggregate train steps/s | h per run (wall) | h per run, effective | h for 21 runs on one GPU |
|---|---|---|---|---|---|---|
| fp32 | 1 | 26.6 | 37.6 | 75.3 | 75.3 | 1,582 |
| fp32 | 2 | 26.7 | 75.2 | 75.5 | 37.7 | 793 |
| fp32 | 3 | 25.8 | 116.0 | 73.0 | 24.3 | 511 |
| fp32 | 4 | 27.0 | 148.8 | 76.6 | 19.1 | 402 |
| fused AdamW | 1 | 26.3 | 38.1 | 74.7 | 74.7 | 1,568 |
| fused AdamW | 2 | 22.0 | 90.8 | 62.5 | 31.2 | 656 |
| fused AdamW | 3 | 21.9 | 137.2 | 62.2 | 20.7 | 435 |
| fused AdamW | 4 | 22.0 | 181.8 | 62.6 | 15.6 | 329 |

Results:
- bf16, fp16 and TF32 are slower than fp32. The step is launch-bound, and autocast adds kernels.
- Fused AdamW saves about 3% alone and 15 to 20% with several runs on the GPU.
- Throughput grows almost linearly to 4 runs per GPU. The limit is one CPU core per run, and 4 runs need about 26 GB RAM. The GPU has room (1.9 GB per run).
- The loaded node (load average about 27 on 112 cores) adds noise of a few ms per step to all numbers.

Noise-control tests (`tests/test_gpu_noise.py`, K = 10, ratios = gap / floor; toy rows with 300 to 420 residues, dropout 0, default model size; dropout is off because a dropout mask drawn for a 16-bit tensor differs from the one for a float32 tensor of the same seed):

| Candidate | step 0 | largest step | final val loss | final test pred | result |
|---|---|---|---|---|---|
| bf16 | 1.28 | 0.96 | 1.42 | 1.38 | pass |
| fp16 | 0.12 | 36 | 7.2 | 11.5 | rejected |
| TF32 | 0.03 | 1.8 | 11.9 | 1.10 | rejected |
| fused AdamW | 0.00 | 0.00 | 0.01 | 0.02 | pass |

- With the production weights multiplied by 1.05 (negative control), all four fail (step 0 ratios 30, 123, 273 and 33,854), so the bounds can detect a real change.
- fp16: the GradScaler starts at scale 65536 and skips the first 5 updates of 8 (the scale falls to 4096). The fp32 run does not skip. With a start scale of 2048 the ratios are 0.12, 1.9, 7.4 and 1.4. This is the normal warm-up. It matters only in a toy run of 48 steps.
- TF32: only the final validation loss exceeds K (11.9 against 10).
- The floor is the mean of 3 perturbation draws. One draw gave ratios from 0 to 14 for fused AdamW, because a single floor value can be small by chance.

## More runs per GPU, CPU-only and the batched trainer (local A10, 31 Oct to 1 Nov 2026)

More processes per GPU (fp32, fused AdamW, 1,500 timed steps, the timing windows aligned):

| runs per GPU | train ms/step | aggregate train steps/s | GPU memory | RAM | h per run (wall) |
|---|---|---|---|---|---|
| 4 | 22.0 | 181.8 | 7.6 GB | 26 GB | 62.6 |
| 6 | 22.3 | 270.2 | 9.9 GB | 39 GB | about 63 |
| 8 | 28.5 | 280.8 | 17.1 GB | 53 GB | about 81 |

The GPU is at 100% from 6 processes on. A 12-process run would need more than 20 GB of GPU memory and was not run. The feature folders cannot be shared between processes by memory mapping: the arrays are pickled object arrays (`np.save(..., allow_pickle=True)`), and `np.load(mmap_mode=...)` does not work on them. A shared layout would need flat arrays with offsets. Each process therefore holds its own copy (6.5 GB).

CPU-only training (fp32): 395 ms/step with 1 thread, 242 ms/step with 2 threads. Eight CPU processes run at about 405 ms/step each, which is 19.7 steps/s in total. This is about 7% of the GPU rate, so the CPU adds little.

Batched trainer (`batched_trainer.py`, `run_split_benchmarks.py --batched`). One process trains all runs. Every run keeps its own parameters, AdamW state and seeded order, and makes its own batch-1 AdamW update every step. The maths per run is that of `train_single_target_tvt.py`. The runs share only the process, the CUDA streams and one copy of each split in memory.
- Model: `batched_model.py`. The parameters of R runs are stacked, and the layers use `bmm`, `baddbmm` and one embedding lookup with offsets. Padded atoms and residues do not reach real ones: padded adjacency rows and columns are zero, attention ignores padded keys, and the mean pooling uses the real positions only.
- Padding: each run pads to its own sample. Residues go to the next size of 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096 and then to a multiple of 1,024. Atoms go to 64, 128, 256, 512, 1,024. On the real train split the mean sequence is 442 residues and the padded length is 533 (1.21 times the residues, 1.41 times for the attention cost). The mean molecule has 43.5 atoms and the padded size is 83.9 (2.7 times for the substrate adjacency, which is small).
- Speed: one CUDA graph per padded shape and run, with static input buffers. The graph holds the forward, the backward and the AdamW update (`torch._fused_adamw_` with the step added before and `found_inf` taken off after, as `torch.optim` does). A run that has no sample in a step (its train set is smaller) has `found_inf` = 1 and keeps its weights, moments and step. Runs wait for the longest epoch.
- Precision: fp32. TF32 and bf16 are off.
- Output: the same files as the single-run trainer (`logfile.csv`, `checkpoint_last.pt` with the same keys, `final_results_*.csv`, `pred_label_*.csv`, `run_summary.json`, `completed.json`) in the same folders. The checkpoint holds the optimizer state in the torch AdamW format. Runs resume together; all runs must be at the same epoch.
- Dropout: the masks come from one RNG stream for the whole process, not from the stream of each run. Only the stream differs; the distribution is the same. All comparisons below use dropout 0.

Throughput on real rows (random_splits_grouped_sequence train, one run per CUDA graph, fp32, 300 timed steps):

| runs R | streams | ms/step (all runs) | aggregate updates/s | GPU memory | eval ms/step | checkpoints, all runs |
|---|---|---|---|---|---|---|
| 3 | 3 | 16.5 | 182 | 3.1 GB | 3.1 | 2.9 s |
| 7 | 7 | 31.9 | 219 | 5.6 GB | 8.0 | 6.3 s |
| 21 | 8 | 101.1 | 208 | 13.8 GB | 26.3 | 12.9 s |
| 21, worst case (the 40 longest sequences first) | 8 | 713 | 29 | 17.1 GB (peak) | 212 | 13.4 s |

Runs that share a padded batch (group size above 1) were slower and needed more memory: R = 7 in one batch gave 134 updates/s and 15.7 GB, and R = 21 in groups of 3 ran out of GPU memory. The default is one run per graph (`--group_size 1`). The step is bound by the GPU: fused AdamW over the dense embedding tables (about 35 M parameters per run) and attention take most of the time.

Projection for 21 runs of 200 epochs on one A10 (real split sizes 50,289 / 2,794 / 2,795): one epoch = 50,289 x 101 ms train + 5,589 x 26.3 ms eval + 13 s checkpoints = 5,240 s. 200 epochs = 291 h, about 12.1 days. Eight processes per GPU (table above) take about 81 h per run in waves of 8, 8 and 5 runs, about 9.4 days. The batched trainer saves RAM (one copy of each split; about 45 GB for 7 splits against 6.5 GB per process) but is not faster than 8 processes.

Tests (`tests/test_batched.py`):
- Padded forward and gradient against the single-run model for each sample: predictions within 1e-5, gradients within 1e-4 of the largest gradient.
- Three runs with unequal train sizes, 14 epochs (the learning rate decays at epoch 13), dropout 0, against three separate single-run trainer runs, on the CPU and on the GPU (`-m gpu`). Per-epoch losses, test RMSE and final weights are inside the noise-control bound: gap = |batched - single|, floor = mean over 3 draws of |single with initial weights perturbed by u = 1e-6 - single|, K = 10. The learning rates are equal. The output folders have the same files. Ratios (gap / floor): CPU 0.23 to 0.82 for all four quantities, GPU 0.00 to 0.73.
- Negative control: initial weights multiplied by 1.05 make the bound fail.
- Resume after 3 of 6 epochs gives the same log and weights as 6 epochs without a stop (exact).
- Groups of 3 runs give the same log as groups of 1 (rtol 1e-5).

## Exact lazy AdamW for the word tables (local A10, 1 Nov 2026)

Code: `lazy_adamw.py` (Triton kernels, pure-torch reference for CPU), `lazy_train.py` (single-run trainer, same flags and output files as `train_single_target_tvt.py`, plus `--sparse_adjacency`), tests in `tests/test_lazy_adamw.py`. The two word tables (`embed_wordGCN`, `embed_wordTrans`) get AdamW updates only on the rows of the sample. Each row keeps the step of its last update. Before a row is used, it runs the skipped steps with g = 0 (same recurrence, the learning rate of each step from the StepLR history, torch eps placement, weight decay each step). After 256 skipped steps, m is below 1e-12 of its start value, and the rest of the steps use the closed form. All other parameters keep fused AdamW. Every checkpoint holds the fully caught-up state, so it equals the dense state, and the dense and lazy trainers load each other's `checkpoint_last.pt` (tested in both directions).

Tests (noise-control bound as above: K = 10, floor = mean over 3 draws of the effect of 1e-6 on the initial weights):
- Toy table, 2,000 steps with learning rate changes. CPU reference against dense torch AdamW passes. GPU kernel against fused dense AdamW (the fused kernel uses float32 constants, so it is the right reference for the kernel): ratios p 2.3, m 2.3, v 5.2. Without catch-up the bound fails.
- Full model, 14 epochs, dropout 0: lazy trainer against the single-run trainer passes on CPU and on the GPU; the sparse adjacency variant passes too. Initial weights times 1.05 fail the bound. Resume equals an uninterrupted run.
- Shorter exact loops fail: with 128 loop steps the ratios are p 2.3, m 2.3, v 5.2 (pass); with 64: p 11.4, m 6.1, v 67.8; with 32: p 59, m 245, v 3158; with 16: p 92, m 1407, v 21078 (all fail). So the loop must stay at 128 or more.

Speed (real rows of random_splits_grouped_sequence, fp32, A10 GPU 3, one process per step stream, same host load for all modes):

| mode | N processes | train ms/step | eval ms/step | aggregate train steps/s |
|---|---|---|---|---|
| dense, fused AdamW | 1 | 17.0 | | 59 |
| lazy | 1 | 19.0 | | 53 |
| dense | 4 | 21.4 | 3.6 | 187.3 |
| lazy | 4 | 23.9 | 6.9 | 167.6 |
| lazy + sparse adjacency | 4 | 27.7 | 5.8 | 108.4 (3 of 4 processes recorded) |
| dense | 8 | 28.3 | 5.7 | 282.7 |
| lazy | 8 | 35.6 | 10.5 | about 225 |
| lazy + sparse adjacency | 8 | 36.6 | 10.1 | about 218 |

Profile of one lazy step (device time per step): `_advance_kernel` 1.82 ms (two launches), attention 0.77 ms, regular fused AdamW 0.33 ms, H2D 0.19 ms, GEMMs and reductions 0.5 ms. Total 4.1 ms, against 3.2 ms for the dense step. The lazy step removes the dense fused AdamW (1.09 ms), the fills and the embedding backward (together about 1.3 ms), but the catch-up loop costs more. Cause: the rows of a sample are mostly rows that were idle for many steps, so they need the full loop of sequential steps. The catch-up work of the unique rows of one sample is larger than the dense update of all 137,455 rows. With a loop of 128 (the shortest that passes) the kernel needs 1.27 ms and the step 3.55 ms, still above dense. The step is also longer on the host (more launches and Python per step).

Result: exact lazy AdamW is not faster than fused dense AdamW for this model and data. The projection for 21 runs is worse than the dense queue (at the GPU limit, 244 steps/s against 312 steps/s from the device times). The live runs stay on the dense trainer. The lazy code and tests stay in the folder for reference. The sparse adjacency lowers GPU memory (1.2 GB against 1.5 GB reserved per process at N = 4) but costs about 7 ms of host time per step. It is not a speed-up.

Not run: N = 12 (does not fit in GPU memory, see above) and the batched trainer with the lazy optimizer (the batched layout stacks the tables of all runs; the lazy step has more device work than dense, so it cannot be faster). `nn.MultiheadAttention` already runs the memory-efficient SDPA kernels in training (`fmha_cutlassF/B_f32` in the profile), so `F.scaled_dot_product_attention` gives the same kernels and no gain.

Wave 1 (single-run trainer, fp32, fused AdamW, 200 epochs, `run_queue.sh`, `run_capped.py`): 6 processes on A10 GPU 1 (8 do not fit: peak 2.8 GB allocated per process, reserved memory grows to 5 to 6.6 GB without a cap). Each process has a cap of 14% of GPU memory and `expandable_segments`. About 1,130 s per epoch (63 h per run). 21 runs in waves of 6: about 9.2 days.
