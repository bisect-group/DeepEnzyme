# DeepEnzyme EMULaToR bench

This wrapper trains DeepEnzyme with its original published settings on the default EMULaToR splits.
Task: kcat only (`data/processed/datasets/kinetic_dataset/kcat`). `OUT` below is `EMULaToR/outputs/baselines/DeepEnzyme`. Seeds: 666, 777, 888.

## Pipeline

1. `prepare_splits.py` picks one structure per row: best aligned experimental PDB (identity >= 90%),
   then AlphaFold, then ESMFold2 (PDB or CIF file). Predicted structures are found by UniProt accession.
   If none of these exists, the raw experimental mmCIF entry (`data/intermediate/mmcif`, chain and coverage from the 3D candidate table) is the last fallback.
   An ESMFold2 file named `<accession>|<mutation>` is preferred over the wild-type accession file.
   Rows without any structure file are dropped.
   Structure folders: `data/intermediate/{processed_exp_pdb,alphafold,esmfold2_pdb,esmfold2_cif,mmcif}`.
2. `cache_features.py` builds the static inputs: sequence 4-gram IDs, substrate fingerprints and adjacency,
   protein contact maps (C-alpha, 10 A).
3. `train_single_target_tvt.py` trains one split and seed. The model is `modeling_fast.py`, a copy of
   `Code/Model` without hardcoded `.cuda()` calls.
4. `run_split_benchmarks.py` runs steps 1 to 3 for all split folders and seeds.
5. `aggregate_deepenzyme_results.py` writes mean and std over seeds.

## Original settings

lr 0.001, 200 epochs, AdamW (default weight decay 0.01), StepLR(3, 0.9) applied as in the original loop,
dropout 0.3, dim 64, 3 output layers, hidden 64/64, 4 heads, hid_size 64, 3 transformer layers,
MSE loss, batch size 1, no shuffling, fp32. The final model is the last epoch.
The model trains on log2(kcat) as the original code does. Reported metrics and saved predictions are in log10, so they compare with the other baselines. `Loss` is the log2 training MSE.

All outputs go under `--work_dir` (default `OUT`). Nothing is written under `data/`.
- Runs: `OUT/<task>/<split_group>/<split_name>/seed_N`. `<split_name>` is `threshold_*` or `all` for flat groups.
- Manifests and features: `OUT/<task>/manifests`, `OUT/<task>/features`.
- Feature cache: `OUT/embeddings/features_v3` (the cache version is in the folder name).
- Summaries: `OUT/<task>/deepenzyme_summary_runs.csv`, `deepenzyme_summary_thresholds.csv`, `aggregate_deepenzyme_results.csv`.
- `<task>` is the name of `--base_dir` (`kcat`).
Runs resume from `checkpoint_last.pt`. Completed runs (`final_results_test.csv` exists) are skipped.

`prepare_splits.py` stops with an error when one of the structure folders is missing or empty.
`prepare_splits.py`, `cache_features.py` and `aggregate_deepenzyme_results.py` need explicit folder arguments (`--manifests_dir`, `--features_dir`, `--cache_dir`, `--results_dir` as they apply). Use `run_split_benchmarks.py` to get the defaults.

Featurization follows the original code. Padded contact maps are float64 and are normalised in float64; equal-length maps are integer maps that become float32 (as in `example.py` and `GCN.py`). The fingerprint code is a copy of the original, so a disconnected molecule can give a fingerprint shorter than its atom count. Rows where the original code raises are dropped, not repaired: unknown atoms or bonds, disconnected molecules whose fingerprint length differs from the atom count, and structures with more C-alpha atoms than residues. `cache_features.py` writes them to `failed_rows.csv`, and `run_summary.json` counts them per split in `rows_dropped`. Cache version: `v3`.

See `commands.txt` for commands.
