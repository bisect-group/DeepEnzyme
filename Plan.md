# DeepEnzyme EMULaToR bench plan

## Summary

Create a bench-owned EMULaToR workflow for DeepEnzyme that retrains on explicit train/val/test split files using the original DeepEnzyme model settings by default. The default command path targets physical GPU 2 through `CUDA_VISIBLE_DEVICES=2`, while Python scripts use `cuda:0` internally.

## Implementation

- Add parquet-native split discovery, manifest preparation, static feature caching, and result aggregation under `emulator_bench/`.
- Resolve one structure per row by choosing the best aligned experimental PDB, then AlphaFold fallback, then ESM fallback.
- Cache only static DeepEnzyme inputs once: sequence token IDs, substrate fingerprints, substrate adjacency, and protein contact maps.
- Use a bench-local DeepEnzyme-equivalent model implementation that removes hardcoded `.cuda()` calls.
- Keep original default settings: `lr=0.001`, `iteration=200`, `dropout=0.3`, `dim=64`, `layer_output=3`, `hidden_dim1=64`, `hidden_dim2=64`, `nhead=4`, `hid_size=64`, `layers_trans=3`.
- Save resumable checkpoints with model, optimizer, scheduler, AMP scaler, RNG state, metric history, and best validation checkpoint.

## Test Plan

- Verify `mldb` imports and CUDA 2 availability.
- Run a tiny CUDA 2 smoke through prepare, cache, train, and metrics export.
- Re-run the smoke without overwrite flags to verify cache reuse and completed-run skipping.
- Verify `last_checkpoint.pt`, `best_checkpoint.pt`, final metrics, and prediction CSVs are written.

## Assumptions

- The default target is EMULaToR `log10_value`.
- Original DeepEnzyme `log2(value)` training is available through `--target_transform log2_from_value`.
- CUDA 2 means `CUDA_VISIBLE_DEVICES=2` externally and `cuda:0` internally.

