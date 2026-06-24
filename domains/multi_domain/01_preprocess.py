# Reads per-domain raw data and produces multi-domain training splits.
# Applies 40/10/50 spatial train/val/test splits for all three domains.
# Subsets Arctic to the grids listed in config.preprocessing.arctic_grids.
# Writes WindowedDataset pkls (train.pkl, val.pkl, test.pkl) per domain
# to outputs/multi_domain/preprocessed/. Does NOT re-run individual domain
# preprocessing — expects individual domain preprocessed data to already exist.
