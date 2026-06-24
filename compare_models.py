# Cross-model intercomparison script.
# Compares Individual, Unified-joint (Stage 1), and Unified-fine-tuned (Stage 2)
# on each domain's held-out test set using identical evaluation metrics.
#
# Prerequisites:
#   - Individual domain pipelines fully run (predictions in outputs/{domain}_domain/predictions/)
#   - Multi-domain pipeline fully run (predictions in outputs/multi_domain/predictions/)
#
# Loads per-domain metrics CSVs produced by individual domain 04_evaluate.py and
# multi-domain 04_evaluate.py, merges them, and produces comparison tables and figures.
# Outputs written to outputs/intercomparison/.
#
# Schema reconciliation required when merging:
#   Individual domain CSVs (outputs/{domain}_domain/evaluation/metrics.csv) have columns:
#     {domain_id_cols, target, RMSE, NSE, KGE, PBIAS}  — no 'stage' column
#   Multi-domain CSVs ({domain}_stage{1,2}_metrics.csv) have the same columns plus a 'stage'
#     column ('stage1' or 'stage2').
#   Arctic individual metrics additionally include 'ssp' and 'period' columns.
#   When merging, assign stage='individual' to individual domain rows before concatenating,
#   so all rows share a uniform {domain_id_cols, target, stage, RMSE, NSE, KGE, PBIAS} schema.
