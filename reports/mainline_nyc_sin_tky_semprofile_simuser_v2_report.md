# Mainline Dataset Report: NYC / SIN / TKY

Date: 2026-07-08

## Scope

Current mainline datasets are:

- NYC: `dataset_clsprec/NYC`, assets in `retrieval_assets_clsprec/NYC`
- SIN: `dataset_clsprec/SIN`, assets in `retrieval_assets_clsprec/SIN`
- TKY: `dataset_getnext_clsprec/TKY`, assets in `retrieval_assets_getnext_clsprec/TKY`

The current model-input version follows `sample_model_input_semprofile_simuser_v2.txt`.
Training still uses `--input-template semantic_profile_simuser_v1`; this template dynamically builds the semprofile/similar-user input from structured columns.

## Leakage / Usability Audit

Detailed JSON audit:

- `reports/dataset_audit_mainline_nyc_sin_tky.json`

Result:

- `ok: true`
- `issues: []`
- No split overlap was detected.
- No per-user chronology violation was detected.
- Val/test users and POIs are covered by train-side vocab.
- Graph candidate files are present for train/val/test.
- Graph candidate pools are generated with train-side index assets.
- Similar-user profile augmentation uses train rows as the retrieval bank, excluding the same sample and same user.

## Built Parquet Outputs

All outputs are under each city's `joined_poi_classification` directory:

- `train_joined_top100.parquet`
- `val_joined_top100.parquet`
- `test_joined_top100.parquet`

Full raw backups are kept as:

- `*_joined_top100_fullraw.parquet`

Compressed current files use:

- `trajectory_lines=12`
- `transitions=6`
- `nearby=6`
- `top_categories=4`
- `revisited_pois=4`

## Dataset Summary

| City | Split | Rows | Graph hit@100 | Own profile insufficient | Similar profile nonempty | Expected similar profile injected by training |
|---|---:|---:|---:|---:|---:|---:|
| NYC | train | 4502 | 0.693025 | 823 | 4502 | 823 |
| NYC | val | 579 | 0.683938 | 2 | 579 | 2 |
| NYC | test | 884 | 0.648190 | 7 | 884 | 7 |
| SIN | train | 4755 | 0.701788 | 815 | 4755 | 815 |
| SIN | val | 627 | 0.728868 | 1 | 627 | 1 |
| SIN | test | 935 | 0.685561 | 5 | 935 | 5 |
| TKY | train | 12319 | 0.828233 | 1318 | 12319 | 1318 |
| TKY | val | 1568 | 0.808673 | 21 | 1568 | 21 |
| TKY | test | 2236 | 0.772361 | 34 | 2236 | 34 |

## Notes

- `similar_user_semantic_profile` is stored for all samples.
- Training injects it only when `user_profile_insufficient=True` and `similar_user_profile_has_signal=True`.
- Refined prompt files are absent for these new assets and were intentionally allowed with `--allow-missing-refined`.
- This is acceptable for the current semantic-profile template because the reranker input is built from recent context, `USER_SEMANTIC_PROFILE`, optional gated `SIMILAR_USER_SEMANTIC_PROFILE`, and `POI_HYPOTHESIS`.
- Do not use the old `retrieval_assets/NewYork` path for this version.
