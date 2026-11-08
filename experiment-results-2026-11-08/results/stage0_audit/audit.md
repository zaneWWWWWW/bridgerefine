# Stage 0 audit

- Root: `E:\tyllltl\Cross-modal conversion`
- Checks: 71 passed, 0 failed
- Split counts: `{'train': 144, 'val': 18, 'test': 18}`
- Cached name counts: `{'train': 27309, 'val': 3473, 'test': 3415}`

## Failed checks

None.

## All checks

| Check | Result |
|---|---|
| split_file_exists | PASS |
| split_counts | PASS |
| patient_split_disjoint | PASS |
| patient_union_count | PASS |
| names_train_exists | PASS |
| names_train_nonempty | PASS |
| names_train_unique | PASS |
| names_train_patients_match_split | PASS |
| names_val_exists | PASS |
| names_val_nonempty | PASS |
| names_val_unique | PASS |
| names_val_patients_match_split | PASS |
| names_test_exists | PASS |
| names_test_nonempty | PASS |
| names_test_unique | PASS |
| names_test_patients_match_split | PASS |
| cache_ct_train_exists | PASS |
| cache_ct_train_finite | PASS |
| cache_ct_train_row_count | PASS |
| cache_ct_train_single_channel | PASS |
| cache_ct_val_exists | PASS |
| cache_ct_val_finite | PASS |
| cache_ct_val_row_count | PASS |
| cache_ct_val_single_channel | PASS |
| cache_ct_test_exists | PASS |
| cache_ct_test_finite | PASS |
| cache_ct_test_row_count | PASS |
| cache_ct_test_single_channel | PASS |
| cache_mr_train_exists | PASS |
| cache_mr_train_finite | PASS |
| cache_mr_train_row_count | PASS |
| cache_mr_train_single_channel | PASS |
| cache_mr_val_exists | PASS |
| cache_mr_val_finite | PASS |
| cache_mr_val_row_count | PASS |
| cache_mr_val_single_channel | PASS |
| cache_mr_test_exists | PASS |
| cache_mr_test_finite | PASS |
| cache_mr_test_row_count | PASS |
| cache_mr_test_single_channel | PASS |
| cache_pix2pix_prior_train_exists | PASS |
| cache_pix2pix_prior_train_finite | PASS |
| cache_pix2pix_prior_train_row_count | PASS |
| cache_pix2pix_prior_train_single_channel | PASS |
| cache_pix2pix_prior_test_exists | PASS |
| cache_pix2pix_prior_test_finite | PASS |
| cache_pix2pix_prior_test_row_count | PASS |
| cache_pix2pix_prior_test_single_channel | PASS |
| cache_coarse_2x1_train_exists | PASS |
| cache_coarse_2x1_train_finite | PASS |
| cache_coarse_2x1_train_row_count | PASS |
| cache_coarse_2x1_train_single_channel | PASS |
| cache_detreg_train_exists | PASS |
| cache_detreg_train_finite | PASS |
| cache_detreg_train_row_count | PASS |
| cache_detreg_train_single_channel | PASS |
| cache_detreg_val_exists | PASS |
| cache_detreg_val_finite | PASS |
| cache_detreg_val_row_count | PASS |
| cache_detreg_val_single_channel | PASS |
| cache_detreg_test_exists | PASS |
| cache_detreg_test_finite | PASS |
| cache_detreg_test_row_count | PASS |
| cache_detreg_test_single_channel | PASS |
| cache_bridge_10x2_test_exists | PASS |
| cache_bridge_10x2_test_finite | PASS |
| cache_bridge_10x2_test_row_count | PASS |
| cache_bridge_10x2_test_single_channel | PASS |
| metrics_patient_file_exists | PASS |
| metrics_patient_count | PASS |
| metrics_patients_match_test_split | PASS |
