# DeepEnzyme EMULaToR bench

This wrapper trains DeepEnzyme with its original published settings on the default EMULaToR splits.
Task: kcat only (`data/processed/datasets/kinetic_dataset/kcat`). `OUT` below is `EMULaToR/outputs/baselines/DeepEnzyme`. Seeds: 666, 777, 888.
Splits: 7 split folders (random grouped by sequence, random grouped by SMILES, UniProt time, enzyme sequence, enzyme structure, substrate, conformer cosine), so 21 runs.

## Pipeline

1. `prepare_splits.py` picks one structure per unique sequence (the rule of the CatPred baseline, `structures.py`):
   1. Experimental: the `pdb_chain` of the dataset row in the mmCIF file (`mmcif/<ID>_updated.cif`, else `<ID>.cif`), with the best `pdb_coverage` among the rows of the sequence in `kcat_kinetic_params_3d.parquet` (the first one wins a tie).
      The author chain, model 1, amino-acid atoms, residues renumbered 1 to n, is written to `OUT/structures/chains/<ID>_<chain>.pdb`.
   2. AlphaFold: `alphafold/AF-<accession>-F1-model_v<n>.pdb`, newest version.
   3. ESMFold2: `esmfold2_pdb/ESMFold2-<id>.pdb`, the full id (with mutations) first, then the accession.
   4. ESM3: `esm/ESM3-open-small-<id>.pdb`.
   Structure folders: `data/intermediate/{mmcif,alphafold,esmfold2_pdb,esm}`.
   A sequence without a structure keeps its rows. They get source `none`.
2. `cache_features.py` builds the static inputs: sequence 4-gram IDs, substrate fingerprints and adjacency,
   protein contact maps (C-alpha, 10 A) from the chain PDB file.
3. `train_single_target_tvt.py` trains one split and seed. The model is `modeling_fast.py`, a copy of
   `Code/Model` without hardcoded `.cuda()` calls.
4. `run_split_benchmarks.py` runs steps 1 to 3 for all split folders and seeds.
5. `aggregate_deepenzyme_results.py` writes mean and std over seeds.

## Original settings

lr 0.001, 200 epochs, AdamW (default weight decay 0.01), StepLR(3, 0.9) applied as in the original loop,
dropout 0.3, dim 64, 3 output layers, hidden 64/64, 4 heads, hid_size 64, 3 transformer layers,
MSE loss, batch size 1, fp32. The final model is the last epoch.
The original shuffles the data once and keeps that order in every epoch. The wrapper shuffles the train rows once with `np.random.RandomState(seed).permutation` and keeps the order in every epoch and after a resume.
Validation and test are evaluated after every epoch, in file order.
The model trains on log2(kcat) as the original code does. Reported metrics and saved predictions are in log10, so they compare with the other baselines. `Loss` is the log2 training MSE.

All outputs go under `--work_dir` (default `OUT`). Nothing is written under `data/`.
- Runs: `OUT/<task>/<split_group>/<split_name>/seed_N`. `<split_name>` is `threshold_*` or `all` for flat groups.
- Manifests and features: `OUT/<task>/manifests`, `OUT/<task>/features`. Extracted chains: `OUT/structures/chains`.
- Feature cache: `OUT/embeddings/features_v6` (the cache version is in the folder name).
- Summaries: `OUT/<task>/deepenzyme_summary_runs.csv`, `deepenzyme_summary_thresholds.csv`, `aggregate_deepenzyme_results.csv`.
- `<task>` is the name of `--base_dir` (`kcat`).
Runs resume from `checkpoint_last.pt`. A completed run has `completed.json` and is skipped.

`prepare_splits.py`, `cache_features.py` and `aggregate_deepenzyme_results.py` need explicit folder arguments (`--manifests_dir`, `--chains_dir`, `--features_dir`, `--cache_dir`, `--results_dir` as they apply). Use `run_split_benchmarks.py` to get the defaults.

## Features and dropped rows

Featurization follows the original code, except for the protein contact map.
- The contact map is built over the dataset sequence. The chain residues (C-alpha atoms of chain A, read by PDB fixed columns, so residues numbered 1000 or more are read) are aligned to the sequence with a global alignment (Biopython `PairwiseAligner`: match +2, mismatch -1, gap open -4, gap extend -1, free end gaps). Each aligned C-alpha goes to its matched position. Tags and insertions are cut. A position without a C-alpha has a self-loop only.
- A map that covers every position is int32; any other map is float64 with a unit diagonal, normalised in float64, as in the original.
- A row without a structure gets a map with self-loops only. These rows are counted in `n_no_structure`.
- No row is dropped for its structure.
- The fingerprint code follows the original and gives the same result wherever the original works. Three repairs cover the molecules where it raises: an unknown atom or bond type gets a fresh id that no fingerprint has, so it maps to id 0; an atom without a bond gets an empty entry; the bond dictionary follows the atom order when such atoms exist. No substrate row is dropped.

`cache_features.py` writes dropped rows to `failed_rows.csv` with the stage (`substrate`, `protein`, `label`).
`run_summary.json` has `rows_dropped` and `failed_by_stage` per partition, and `structure` per partition: `n_rows`, `n_no_structure`, `n_aligned_full`, `n_partial`, `mean_coverage`, `n_low_identity` and `sources` (experimental, alphafold, esmfold2, esm3, none).
A feature folder is rebuilt when the manifest, the radius, the n-gram size, the distance threshold, a dictionary file or the structure rule version changes.

## Speed

Opt-in flags (default off): `--amp` (bf16), `--fp16`, `--tf32`, `--fused_adamw`. On an A10 none of them is faster than fp32; see `REPORT.md`, "GPU speed audit".

The per-sample batch size and the model are unchanged. `normalize_adj` scales rows and columns instead of two dense products (the same bits). Loss and predictions stay on the device until the end of each pass, and on CUDA the adjacency is copied from pinned memory without blocking. Logged values are the same.

## Verify equivalence

Run `emulator_bench/verify_equivalence.sh` from any folder (env `mldb`). `--gpu` adds the CUDA tests. `--negative-control` runs only the precision tests with the production weights multiplied by 1.05. It exits 0 only if every one of them fails.
CPU mode uses synthetic data in a temporary folder and the public files in `Data/Input`. It checks:
- Weights after the same seed, forward outputs (eval and train mode) and `normalize_adj` (bit-exact, sizes up to 2000) against `Code/Model`.
- Losses, learning rates and final weights of 14 epochs against the loop of `Code/Model/train.py` (exact). The test model drops the label from the forward input and returns a scalar, because `train.py` passes the whole sample tuple and calls `float()` on one-element arrays, which fails on numpy 2.
- Fingerprints, adjacency, 4-gram tokens and contact maps against `Code/Example/example.py`.
- The train order (one permutation from the seed, the same in every epoch and after a resume), resume against an uninterrupted run (exact), and the log10 conversion.
- The structure rule on synthetic mmCIF, AlphaFold, ESMFold2 and ESM3 files, the cache key, and the counts of dropped, aligned, partial, low-identity and no-structure rows.
GPU mode also checks `normalize_adj`, the forward pass (bit-exact) and the fp32 training loop against the original (within 1e-5). The precision tests (`test_gpu_noise.py`: bf16, fp16, TF32, fused AdamW against fp32) use a noise-control bound: the gap to fp32 must be at most K = 10 times the effect of one rounding of the initial weights at the production precision (mean of 3 perturbation draws; step 0, largest step, final validation loss and final test predictions). fp16 and TF32 are recorded as rejected (strict xfail)
`tests/test_packed.py` (CPU) checks the packed-token trainer against `minibatch_trainer.py`: per-step losses, predictions and final weights, micro-batching by `--max_tokens`, eval-mode forwards with dropout configured, packing against batch-1 forwards, resume against an uninterrupted run, and the completion guard. `tests/test_packed_gpu.py` (GPU, same noise-control bound) checks packed fp32 against the padded trainer (u = 1e-6) and packed bf16 against packed fp32 (u = 2**-8).
Not exact: GPU runs (see above), and the 107 SMILES where the original raises (repaired, see above).

See `commands.txt` for commands.

## Batched trainer

`batched_trainer.py` trains many runs in one process (fp32, one CUDA graph per run) with the same per-run batch-1 AdamW maths and the same output files as `train_single_target_tvt.py`. Use `run_split_benchmarks.py --batched` (first GPU in `--gpus`), or call the script with one `--job TRAIN VAL TEST OUT SEED` per run. Dropout masks come from one RNG stream for the process, so only the stream differs from single runs. Tests are in `tests/test_batched.py`; the GPU test runs with `-m gpu`. See REPORT.md for speed, memory and the comparison with several processes per GPU.

## Lazy AdamW (not faster)

`lazy_train.py` and `lazy_adamw.py` update the two word tables only on the rows of each sample, and catch the other rows up exactly. The checkpoints are compatible with the single-run trainer in both directions. It passes the noise-control tests, but it is slower than fused dense AdamW for this model (see REPORT.md), so `run_queue.sh` uses `train_single_target_tvt.py`. Tests: `tests/test_lazy_adamw.py` (CPU tests run in `verify_equivalence.sh`; GPU tests with `-m gpu`).

## Packed-token trainer

`packed_trainer.py` has the CLI and outputs of `minibatch_trainer.py` and the same update, without padded residues: the protein transformer attends within each sample on concatenated tokens (nested-tensor attention). One micro-batch per step; a batch above `--max_tokens` (default 400000 residues) is cut into chunks with summed gradients. `--bf16` is an opt-in autocast (GCN sparse products and segment means stay fp32). Batches are prepared by three prefetch threads and the GPU never waits for them. A run whose `completed.json` has another `final_epoch` than `--iteration` raises an error; use `--overwrite` or a new `--out_dir`. The peak memory of the first epoch is printed and saved in `run_summary.json`.
