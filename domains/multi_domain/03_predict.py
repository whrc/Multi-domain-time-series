# Generates predictions for a single domain using a trained multi-domain checkpoint.
# Specify --domain {arctic,amazon,rangeland} and --checkpoint {stage1,stage2}.
# Loads the appropriate model weights (unified for stage1, domain-specific head for stage2),
# runs inference on the domain's held-out test set, and writes predictions to:
#   Arctic:    NetCDF per variable per grid per SSP
#              (outputs/multi_domain/predictions/{domain}_{checkpoint}_{grid}_{ssp}_{variable}.nc)
#   Amazon:    Parquet (outputs/multi_domain/predictions/amazon_{checkpoint}_predictions.parquet)
#   Rangeland: Parquet (outputs/multi_domain/predictions/rangeland_{checkpoint}_predictions.parquet)
