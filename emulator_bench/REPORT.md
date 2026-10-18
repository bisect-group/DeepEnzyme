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
| batch size | 1 (per-sample updates, no shuffling) | train.py:64-70, shuffle commented out :62 |
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
enzyme_sequence_splits/threshold_*, substrate_splits/threshold_* (thresholds found on disk). 5 jobs, 15 runs.
All outputs go under `--work_dir`: runs in `OUT/<task>/<split_group>/<split_name>/seed_N`, feature cache in `OUT/embeddings/features_v3`. Nothing is written under `data/`. The model trains on log2(kcat); metrics and predictions are in log10.
Test metrics come from the last-epoch model.

## Files removed
`tune_optuna.py`, `launch_parallel_optuna.py`, `launch_parallel_retrain_from_optuna.py`, `run_from_config.py`
and `benchmark_config.example.json` (both used grid/Optuna-style keys and stale argument names),
`build_tvt_features.py` (unused CSV path, no caller). In `../Plan.md` the two Optuna lines were removed.

## Files changed
- `common.py`: removed default data/result paths and unused constants. Default seeds 666 777 888.
  Default split groups: the five groups above (removed enzyme_structure and conformer_cosine, not on disk).
- `run_split_benchmarks.py`: `--base_dir` and `--work_dir` have EMULaToR defaults. Removed `_load_hparams` and `--hparams_json`
  (no tuned settings). Removed fixed GPU default 2. `--amp` opt-in. `--iteration` stays as a smoke-test override.
- `train_single_target_tvt.py`: weight decay default 0.01 (original AdamW default). Final model is the last epoch
  (best-validation checkpoint removed). fp32 by default (`--amp` opt-in). TF32 switches removed.
  `run_summary.json` has `final_epoch` and `final_val_rmse`.
- `prepare_splits.py`: ESM folder is now `data/intermediate/esmfold2_pdb` with prefix `ESMFold2-`
  (the old `esm/ESM3-open-small-*` folder does not exist). Paths derive from the EMULaToR root, not a home dir.
  Structure cache lives in `<work_dir>/<task>/manifests/_structure_alignment_cache`.
- `cache_features.py`: path arguments are required. Default dictionary dir is `<submodule>/Data/Input`.
- `aggregate_deepenzyme_results.py`: `--results_dir` required. Output default `<results_dir>/aggregate_deepenzyme_results.csv`.
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

The resolver used to drop 50-60% of the rows. Cause: only experimental entries in `processed_exp_pdb` (identity >= 90%) and candidates keyed by the `pdbs` column were checked. The ESMFold2 and AlphaFold structures were looked up only for rows whose source was predicted. Rows with an experimental PDB id that has no processed file (64% of the experimental rows) were dropped, although a predicted structure for their UniProt accession exists.

The resolver now uses every structure source on disk, in this order:
1. Best aligned experimental PDB in `processed_exp_pdb` (identity >= 90%).
2. AlphaFold by UniProt accession. The accession comes from `pdbs` (`<pdb>:<accession>`, `<accession>`) or from `uniprot_id` (mutation suffix removed).
3. ESMFold2 by accession. Index = `esmfold2_pdb` (259,299 files) plus `esmfold2_cif` (259,921 files). The PDB file wins when both exist. 622 accessions exist only as CIF. A small mmCIF reader (`feature_utils._get_ca_coords_cif`) reads the CIF files. Biopython's parser fails on the ESMFold2 CIFs (no `_atom_site.occupancy`). The reader matches the PDB coordinates for the same protein. `manifest.parquet` adds no accession that the file names do not already give, so it is not used.
4. Raw experimental mmCIF in `data/intermediate/mmcif` (`<PDB>_updated.cif`), with the chain and coverage from the 3D candidate table (best coverage).
Rows without any structure are dropped.

Row coverage (rows kept / rows in the split file; kcat; real files, all three stages measured with the same code on the same data):

| Split (kcat) | part | rows | before | ESMFold2 by accession (steps 1-3) | after (steps 1-4) | dropped now |
|---|---|---|---|---|---|---|
| random_splits_grouped_sequence | train | 55,018 | 27,108 (49.3%) | 51,540 (93.7%) | 54,872 (99.7%) | 146 |
| | val | 3,056 | 1,202 (39.3%) | 2,703 (88.5%) | 3,041 (99.5%) | 15 |
| | test | 3,057 | 1,207 (39.5%) | 2,747 (89.9%) | 3,052 (99.8%) | 5 |
| random_splits_grouped_smiles | train | 55,018 | 26,376 (47.9%) | 51,276 (93.2%) | 54,890 (99.8%) | 128 |
| | val | 3,056 | 1,566 (51.2%) | 2,861 (93.6%) | 3,041 (99.5%) | 15 |
| | test | 3,057 | 1,575 (51.5%) | 2,853 (93.3%) | 3,034 (99.2%) | 23 |
| uniprot_time_splits | train | 55,029 | 26,371 (47.9%) | 51,313 (93.2%) | 54,890 (99.7%) | 139 |
| | val | 3,068 | 1,843 (60.1%) | 2,824 (92.0%) | 3,052 (99.5%) | 16 |
| | test | 3,034 | 1,303 (42.9%) | 2,853 (94.0%) | 3,023 (99.6%) | 11 |
| enzyme_sequence_splits/threshold_0.01 | train | 55,026 | 26,295 (47.8%) | 51,359 (93.3%) | 54,901 (99.8%) | 125 |
| | val | 3,056 | 1,654 (54.1%) | 2,825 (92.4%) | 3,055 (100.0%) | 1 |
| | test | 3,049 | 1,568 (51.4%) | 2,806 (92.0%) | 3,009 (98.7%) | 40 |
| substrate_splits/threshold_0.205 | train | 55,389 | 26,890 (48.5%) | 51,510 (93.0%) | 55,245 (99.7%) | 144 |
| | val | 3,056 | 1,509 (49.4%) | 2,899 (94.9%) | 3,056 (100.0%) | 0 |
| | test | 2,686 | 1,118 (41.6%) | 2,581 (96.1%) | 2,664 (99.2%) | 22 |

The same stages for km (not a DeepEnzyme task in the paper; measured because the candidate table exists): train 54-55% before and 99.7-99.8% after; val 50-64% before and 99.6-99.8% after; test 51-64% before and 99.0-100% after. The wrapper stays kcat-only.

Notes:
- "Dropped now" counts rows with no structure on disk. These are the exact per-split drop counts of the structure step. A few more rows can fail in `cache_features.py`.
- The percentages of the "before" column match the earlier review (about half of the rows).
- The identity filter (>= 90%) of step 1 is unchanged. Experimental rows below it now fall through to steps 2-4.
- Rows resolved as ESMFold2 or AlphaFold use a predicted wild-type structure, like the original DeepEnzyme.

## Changes after review

- Target: train on log2(kcat); metrics and predictions in log10. `cache_features.py` builds the label from `value` (`log2_value`) and marks the feature folder; old feature folders with another label are rebuilt. The `--target_transform` options were removed from `cache_features.py` and `run_split_benchmarks.py`.
- Rows: new resolver (see "Structure coverage"), CIF/mmCIF reader, `--esm_cif_dir` and `--mmcif_dir` options in `prepare_splits.py`. `_candidate_lookup` reads `<task>_kinetic_params_3d.parquet` by task folder name.
- Stale-cache guards: prepared manifests carry an `index_fingerprint` (structure indexes plus a resolver version) and are rebuilt when it differs. Feature folders store the label, the manifest size and mtime, and `--limit_rows`, and are rebuilt when any of them differs. A rerun of the same command skips both stages (checked).
- Summaries report std, not variance: `run_split_benchmarks.summarize` (`*_mean`, `*_std`) and `aggregate_deepenzyme_results.py` (both used variance).
- `--base_dir` and `--work_dir` have defaults (kcat task dir; `EMULaToR/outputs/baselines/DeepEnzyme`, computed from the file location). Paths are resolved to absolute before the subprocesses run in the submodule root. The aggregate script default `--results_dir` is `<default work_dir>/<task>` and it adds the repo root to `sys.path`.
- README and `commands.txt` updated (no `runs/kcat` path, no variance, no `--target_transform`).
- Smoke after the changes (mldb, GPU 1, 1-2 epochs, 100 rows per split, kcat random_splits_grouped_sequence subset): prepare, cache, train, and metrics worked. Predictions are in log10 (the `Loss` column is about 25, the log2 MSE; RMSE about 1.5). A second smoke used 25 rows per split that were resolved as raw mmCIF and ESMFold2 CIF: 0 featurization failures, 1 epoch trained. Outputs: `scratchpad/bench_smoke/DeepEnzyme/{w2,w3}`. Coverage runs: `.../{before,after_esm_only,after,before_km,after_km}`.
- Speed note for the coverage run: PDB chain sequences for 9,697 experimental entries were parsed once in parallel (scratch script) and seeded into the resolver cache. The wrapper itself parses them one at a time and caches them, so a first `prepare_splits.py` run is slow (tens of minutes).
- `prepare_splits.py`: a prepared manifest stores `source_fingerprint` (source split path, size, mtime, `--limit_rows`). `prepared_manifest_is_valid` checks it, so a smoke manifest is never reused by a full run. `RESOLVER_VERSION` is 3.
- `prepare_splits.py`: ESMFold2 files named `<acc>|<mut>` are tried first with the exact key (mutation kept), then the wild-type accession.
- `README.md`: lists the raw mmCIF fallback step and the `mmcif` folder.
- Smoke (mldb, CPU): `--limit_rows 8` twice gave 0 then 3 skipped splits; `--limit_rows 12` rebuilt all 3. A fake ESM folder: `P12345|A12G` picks the mutant file, `P12345|B1C` and `Q99999|X1Y` fall back to the wild-type file. Output: `scratchpad/bench_smoke/DeepEnzyme`.

## Commands
Run from `EMULaToR/baselines/DeepEnzyme`, with `PY=/home/adhil/.conda/envs/mldb/bin/python`:

```
PY=/home/adhil/.conda/envs/mldb/bin/python
$PY emulator_bench/run_split_benchmarks.py --seeds 666 777 888 --gpus 0 1 --runs_per_gpu 2
$PY emulator_bench/aggregate_deepenzyme_results.py
```
Only kcat applies. km, ki, ic50, kd and ec50 are not DeepEnzyme tasks in the paper.

## Smoke test
Env: mldb, CPU (first smoke, before the review fixes). Data: copy of 300/100/100 rows of kcat random_splits_grouped_sequence plus the matching 3d rows,
in `scratchpad/bench_smoke/DeepEnzyme/task`. Command: `--limit_rows 100 --iteration 1 --seeds 666 --device cpu`.
Result: prepare kept 23/65/55 rows (train/val/test), features cached, one epoch trained, final
train/val/test metrics, predictions and `completed.json` written, aggregate script ran, a second run
skipped the completed run. A dry run on the real kcat folder lists 15 training jobs (5 splits x 3 seeds).
No full training done. Nothing was written under `EMULaToR/data`.

## Open issues
- Structure coverage: see "Structure coverage". The remaining 0.3% of rows have no structure on disk and are dropped.
- Last-epoch model versus best-validation. The original saves the last epoch, so the wrapper does too. The
  paper appendix says validation-based checkpointing where the original has it. DeepEnzyme has none.
- Precision. The camera-ready appendix says bf16 for kinetic baselines. The original code is fp32, so the
  default is fp32. Use `--amp` to match the appendix, at some risk to the original numerics.
- Speed. Training is per-sample (batch size 1, 200 epochs, test evaluated every epoch as in the original).
  A full kcat run (55k train rows) is slow. No speed-up changes batch behaviour, so none was applied.
- The 3d parquet is read in full by `prepare_splits.py` (476k rows). It is read once per run of the script.
- Structures for mutant sequences are the wild-type UniProt structure (AlphaFold, ESMFold2) as in the original, which also used one predicted structure per UniProt entry. The contact map is zero-padded to the sequence length. A structure with more C-alpha atoms than residues is dropped (see "Featurization and dtype rules").
- Raw mmCIF chains (3,975 of 54,872 train rows in random_splits_grouped_sequence) use the chain and coverage mapped in the 3D candidate table. There is no sequence-identity check for them, and residue gaps can shift positions. Same crude length fix as for the other sources.
- Featurization failures in `cache_features.py` drop more rows (listed in `failed_rows.csv` per split, counted in `run_summary.json`). I did not run the full cache.
- Contact maps are zero-padded at the end when the structure is shorter than the sequence (many experimental and raw mmCIF rows), and the mmCIF step has no minimum coverage. The PDB reader drops residues numbered >= 1000 (original `example.py` behaviour). The mmCIF reader loads all atom rows into memory.

## Featurization and dtype rules

The wrapper copies the original featurization and arithmetic. Where it differed, it now follows the original.

- Contact map dtype. The original pads the map with `np.zeros`, so a padded map is float64 and is normalised in float64 (`GCN.py:24-26`, `example.py:141`). A map with equal length is an integer map, which becomes float32 in `adj + torch.eye`. The wrapper stores padded maps as float64 and equal-length maps as int32. The model keeps the dtype and casts the normalised matrix to float32 before `mm`, as the original does. With this rule the weights and test predictions of a 96-row run match the original to 1e-12 or better (see "Equivalence check").
- Fingerprints. `extract_fingerprints` is a copy of the original. It emits entries only for atoms that have a bond, in bond-dict order. For a disconnected molecule the result is shorter than the atom count. Unknown atoms, bonds and short lists raise (`KeyError`, `IndexError`). There is no workaround. Unknown fingerprints and edges map to id 0 as in the original, without changing the dictionaries.
- SMILES cache key. The atom order depends on the input string, so the cache key uses the raw SMILES, not the canonical SMILES.
- Structures. A structure with more C-alpha atoms than residues raises, as in the original. The wrapper does not truncate.
- Dropped rows. The original raises on these rows, so no original-code run uses them. The wrapper drops them in `cache_features.py` (`failed_rows.csv`, `rows_failed` in `feature_summary.json`) and `run_summary.json` has `rows_dropped` per split (train, val, test). A row also counts as dropped when the fingerprint length differs from the atom count, because the original model fails on it (matrix shape mismatch). Train rows the original would filter and val/test rows the featurizer cannot build are treated the same way, as in the DLKcat wrapper. Rows without any structure file are dropped earlier, in `prepare_splits.py`.
- Cache version. `CACHE_VERSION` is `v3` (it was `v2`). It is in every cache key and in `feature_summary.json`, so old caches and feature folders are rebuilt.
- Not matched: the wrapper falls back to any chain when the requested chain has no C-alpha atoms, and it reads mmCIF files. The original reads PDB chain A only.

## Equivalence check

- 96-row run (kcat, random_splits_grouped_sequence, seed 666, CPU, wrapper CLI with default flags) against the original functions and classes with a minimal driver. 3 and 13 epochs: initial and final weights differ by 0, test predictions by 4.4e-16 (log2), epoch metrics by at most 1.8e-15.
- 2967 real kcat train SMILES: the original raises on 33 (the wrapper rejects the same 33). 0 fingerprint arrays and 0 adjacency arrays differ on the rest. For 6 molecules the fingerprint length is not the atom count in both implementations. The original model cannot run on them, so the wrapper drops them.
- 270 PDB rows (chain A): the original raises on 4, the wrapper on the same 4. 0 contact maps differ (248 integer maps, 18 float64 maps).
- Drop smoke (6 rows per split: 4 good, 1 disconnected SMILES, 1 structure with 3880 C-alpha atoms for 194 residues): `rows_dropped` is 2 for train, val and test, and training runs.
