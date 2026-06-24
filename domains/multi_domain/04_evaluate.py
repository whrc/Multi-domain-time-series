# Evaluates Stage 1 (joint pretrain) and Stage 2 (fine-tuned heads) independently,
# per domain. For each stage: load test predictions from outputs/multi_domain/predictions/,
# compute per-unit RMSE/NSE/KGE/PBIAS metrics, and produce scatter plots and metric
# boxplots — same style as individual domain pipelines. Each stage gets its own
# output directory (evaluation/stage1/, evaluation/stage2_{domain}/).
# Cross-model comparison with Individual baselines is handled by a separate
# root-level intercomparison script (compare_models.py).
