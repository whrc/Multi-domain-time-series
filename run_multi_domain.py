# Entry point for the multi-domain experiment.
# Accepts --stage {preprocess, pretrain, finetune, predict, evaluate} to run
# any pipeline step in isolation. Intended workflow:
#   1. python run_multi_domain.py --stage preprocess
#   2. python run_multi_domain.py --stage pretrain
#      (inspect Stage 1 val metrics and training plots before proceeding)
#   3. python run_multi_domain.py --stage finetune
#   4. python run_multi_domain.py --stage predict --domain {arctic,amazon,rangeland} --checkpoint {stage1,stage2}
#   5. python run_multi_domain.py --stage evaluate
