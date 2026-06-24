# Two-stage training for the multi-domain model.
#
# Stage 1 (pretrain): joint pre-training with round-robin domain batching
#   (Arctic → Amazon → Rangeland → repeat). Trains the full architecture —
#   shared transformer, all input projections, and all domain heads simultaneously.
#   Saves best unified checkpoint to outputs/multi_domain/models/stage1_best.pt.
#
# Stage 2 (finetune): loads best Stage 1 checkpoint, freezes shared transformer
#   and input projections, fine-tunes each domain's MLP head independently using
#   that domain's full training split. Saves per-domain fine-tuned heads to
#   outputs/multi_domain/models/stage2_{domain}_best.pt.
#
# Run via run_multi_domain.py --stage pretrain  or  --stage finetune.
