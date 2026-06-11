# Stone Relic Damage Segmentation Experiments

This repository is the GitHub-backed workspace for the local project under `D:\mask`.

## What Is Tracked

- `baseline_runs/*.py`: training, inference, dataset-building, review, and post-processing scripts.
- `baseline_runs/*.csv`: small experiment summary tables.
- `results/20260611_fcf/`: compact metrics from the FCF vs RSG comparison.

## What Is Not Tracked

Large datasets, generated image previews, masks, overlays, and model checkpoints are intentionally excluded by `.gitignore`.
Keep those locally under `D:\mask` or external storage, and commit only scripts plus small reproducibility metadata.

## Current Note

The 2026-06-11 FCF module improved crack recall slightly over RSG:

- RSG `mIoU_damage`: 0.540100
- FCF tuned `mIoU_damage`: 0.542488
- Main effect: more crack pixels recovered; spalling stayed almost unchanged.
