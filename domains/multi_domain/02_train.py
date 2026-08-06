"""
Multi-domain — Step 2: training.

See domains/multi_domain/multi_description.md § "Step 2 — Training".

Two stages controlled by --stage:
  pretrain  — joint mixed-step training across all three domains
  finetune  — freeze shared weights, fine-tune each domain head independently

--flux-only selects the flux-only target-set variant (Arctic: GPP/RECO only;
Rangeland: GPP/RECO/Rm/Rg only; Amazon unaffected — see multi_description.md
§ "Flux-Only Variant"). Outputs for each variant land in separate
pretrained[_fluxonly]/ and finetuned[_fluxonly]/ subfolders.
"""

import argparse
import itertools
import logging
import pickle
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from domains.multi_domain.flux_only import (  # noqa: E402
    AMAZON_NFEATURES,
    DOMAIN_ID_FIELDS,
    DOMAINS,
    RANGELAND_NFEATURES,
    apply_flux_only,
    arctic_stride_seq_len,
    arctic_train_pkl_path,
    checkpoint_path,
    pretrain_shared_dir,
    stage_output_dir,
    variant_ntargets,
    variant_target_names,
)
from domains.multi_domain.model import DomainRoutedModel, MultiDomainModel  # noqa: E402
from shared.dataset import WindowedDataset, records_to_segments  # noqa: E402
from shared.evaluate import per_unit_metrics, predict_and_inverse, stack_by_target  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_pred_vs_true  # noqa: E402
from shared.training import build_warmup_cosine_scheduler, masked_mse_loss, run_lr_finder, set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_pkl(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def domain_stride(domain: str, path: Path, cfg: dict) -> int:
    """Arctic pkls carry their own sidecar stride (its size/stride-labeled train variant, and
    separately for val) — a global cfg.preprocessing.stride does not apply to Arctic. Amazon/
    Rangeland have no such labeling; they use the global config stride (unchanged behavior)."""
    if domain == "arctic":
        stride, _ = arctic_stride_seq_len(path)
        return stride
    return cfg["preprocessing"]["stride"]


def make_loader(records: list[dict], n_targets: int, seq_len: int, stride: int,
                batch_size: int, shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    segs, meta = records_to_segments(records)
    ds = WindowedDataset(segs, meta, n_targets, seq_len, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def build_domain_specs(active_domains: list[str], train_records: dict, ntargets: dict) -> dict:
    """nFeatures/nTargets per domain, restricted to active_domains — Arctic's nFeatures is
    inferred from data (its feature count depends on nStatic, which varies), Amazon/Rangeland
    are fixed known constants."""
    nfeatures = {"amazon": AMAZON_NFEATURES, "rangeland": RANGELAND_NFEATURES}
    specs = {}
    for d in active_domains:
        if d == "arctic":
            nF = train_records["arctic"][0]["data"].shape[1] - ntargets["arctic"]
        else:
            nF = nfeatures[d]
        specs[d] = {"nFeatures": nF, "nTargets": ntargets[d]}
    return specs


def val_loss_per_domain(model: MultiDomainModel, val_loaders: dict,
                        device: torch.device) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for d in val_loaders:
            sse = cnt = 0.0
            for x, y in val_loaders[d]:
                x, y  = x.to(device), y.to(device)
                pred  = model(x, domain=d)
                valid = ~torch.isnan(y)
                if valid.any():
                    sse += ((pred - y)[valid] ** 2).sum().item()
                    cnt += valid.sum().item()
            losses[d] = sse / cnt if cnt > 0 else float("nan")
    return losses


def post_train_plots(model: MultiDomainModel, val_records: dict, scalers: dict,
                     domain_specs: dict, seq_len: int, flux_only: bool,
                     device: torch.device, out_dir_for: Callable[[str], Path]) -> None:
    model.eval()
    target_names_by_domain = variant_target_names(flux_only)
    for d in domain_specs:
        out_dir = out_dir_for(d)
        out_dir.mkdir(parents=True, exist_ok=True)
        domain_model = DomainRoutedModel(model, d)
        n_targets    = domain_specs[d]["nTargets"]
        target_names = target_names_by_domain[d]
        seg_meta, pred_list, obs_list = predict_and_inverse(
            domain_model, val_records[d], n_targets, seq_len, device, scalers[d]
        )
        pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
        plot_pred_vs_true(pred_d, obs_d, save_path=out_dir / f"{d}_scatter.png")
        metrics_df = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, DOMAIN_ID_FIELDS[d])
        plot_metric_boxplot(metrics_df, save_path=out_dir / f"{d}_boxplot.png")
        logger.info("Post-train plots saved: %s", out_dir / f"{d}_*.png")




def run_pretrain(cfg: dict, train_records: dict, val_records: dict, scalers: dict,
                 train_paths: dict, val_paths: dict, flux_only: bool,
                 active_domains: list[str], seed: int | None = None) -> float:
    tcfg  = cfg["training"]
    seq_len  = cfg["model"]["seq_len"]
    batch_sz = tcfg["batch_size"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg["paths"]["models_dir"])
    eval_dir   = Path(cfg["paths"]["evaluation_dir"])

    ntargets = variant_ntargets(flux_only)
    ckpt_path = checkpoint_path(models_dir, "pretrained", None, flux_only, seed, active_domains)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    domain_specs = build_domain_specs(active_domains, train_records, ntargets)
    logger.info("Device=%s | domains=%s | flux_only=%s", device, active_domains, flux_only)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loaders = {d: make_loader(train_records[d], ntargets[d], seq_len,
                                    domain_stride(d, train_paths[d], cfg), batch_sz, True,
                                    generator)
                     for d in active_domains}
    val_loaders   = {d: make_loader(val_records[d],   ntargets[d], seq_len,
                                    domain_stride(d, val_paths[d], cfg), batch_sz, False)
                     for d in active_domains}
    logger.info("Train windows: %s", {d: len(train_loaders[d].dataset) for d in active_domains})

    # Anchor domain for the LR finder and the steps_per_epoch fallback: the domain with the
    # most training windows in the active set is fully exhausted each epoch while the others
    # cycle multiple times — generalizes the original 3-domain design (where Arctic was always
    # the anchor, since it has by far the most data) to subsets that may exclude Arctic (e.g.
    # the {amazon, rangeland} ablation — see ablation_test/ablation_description.md).
    anchor = max(active_domains, key=lambda d: len(train_loaders[d].dataset))

    model = MultiDomainModel(cfg, domain_specs).to(device)

    if tcfg.get("optimized_lr") is not None:
        lr = float(tcfg["optimized_lr"])
        logger.info("Using configured optimized_lr=%.3e", lr)
    else:
        lr_dir = pretrain_shared_dir(eval_dir, flux_only, seed, active_domains)
        lr_dir.mkdir(parents=True, exist_ok=True)
        probe = DomainRoutedModel(model, anchor)
        lr = run_lr_finder(probe, train_loaders[anchor], float(tcfg["initial_lr"]),
                           device, lr_dir / "lr_finder.png")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    scheduler = build_warmup_cosine_scheduler(optimizer, tcfg["pretrain_epochs"], tcfg.get("warmup_epochs", 0))

    steps_per_epoch = tcfg["steps_per_epoch"] or len(train_loaders[anchor])
    domain_iters    = {d: iter(itertools.cycle(train_loaders[d])) for d in active_domains}

    best_val   = float("inf")
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, tcfg["pretrain_epochs"] + 1):
        model.train()
        epoch_sse = {d: 0.0 for d in active_domains}
        epoch_cnt = {d: 0.0 for d in active_domains}

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for d in active_domains:
                x, y  = next(domain_iters[d])
                x, y  = x.to(device), y.to(device)
                pred  = model(x, domain=d)
                loss_d = masked_mse_loss(pred, y)
                total_loss = total_loss + loss_d
                valid = ~torch.isnan(y)
                if valid.any():
                    epoch_sse[d] += ((pred - y)[valid] ** 2).sum().item()
                    epoch_cnt[d] += valid.sum().item()
            (total_loss / len(active_domains)).backward()
            optimizer.step()

        scheduler.step()
        train_losses = {d: epoch_sse[d] / epoch_cnt[d] if epoch_cnt[d] > 0 else float("nan")
                        for d in active_domains}
        train_str = "  ".join(f"{d}={train_losses[d]:.4f}" for d in active_domains)
        logger.info("Epoch %3d  train  %s", epoch, train_str)

        if epoch % tcfg["eval_every_n_epochs"] == 0:
            val_losses = val_loss_per_domain(model, val_loaders, device)
            mean_val   = sum(v for v in val_losses.values() if v == v) / len(active_domains)
            val_str    = "  ".join(f"{d}={v:.4f}" for d, v in val_losses.items())
            logger.info("Epoch %3d  val    %s  mean=%.4f", epoch, val_str, mean_val)
            history.append({
                "epoch": epoch,
                **{f"train_{d}": train_losses[d] for d in active_domains},
                **{f"val_{d}": val_losses[d] for d in active_domains},
                "val_mean": mean_val,
            })

            if mean_val < best_val:
                best_val   = mean_val
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                logger.info("  → checkpoint saved (val=%.4f)", best_val)
            else:
                no_improve += 1
                if no_improve >= tcfg["early_stopping_patience"]:
                    logger.info("Early stopping at epoch %d (best val=%.4f)", epoch, best_val)
                    break

    logger.info("Pretrain stage complete. best_val=%.4f  checkpoint=%s", best_val, ckpt_path)
    hist_dir = pretrain_shared_dir(eval_dir, flux_only, seed, active_domains)
    hist_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).round(4).to_csv(hist_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    post_train_plots(model, val_records, scalers, domain_specs, seq_len, flux_only, device,
                     lambda d: stage_output_dir(eval_dir, "pretrained", d, flux_only, seed, active_domains))
    return lr


def run_finetune(cfg: dict, train_records: dict, val_records: dict, scalers: dict,
                 train_paths: dict, val_paths: dict, flux_only: bool,
                 active_domains: list[str], lr: float | None = None,
                 seed: int | None = None) -> None:
    tcfg  = cfg["training"]
    seq_len  = cfg["model"]["seq_len"]
    batch_sz = tcfg["batch_size"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg["paths"]["models_dir"])
    eval_dir   = Path(cfg["paths"]["evaluation_dir"])

    ntargets = variant_ntargets(flux_only)
    domain_specs = build_domain_specs(active_domains, train_records, ntargets)

    pretrain_ckpt = checkpoint_path(models_dir, "pretrained", None, flux_only, seed, active_domains)
    if not pretrain_ckpt.exists():
        raise FileNotFoundError(f"{pretrain_ckpt} — run --stage pretrain first")

    model = MultiDomainModel(cfg, domain_specs).to(device)
    model.load_state_dict(torch.load(pretrain_ckpt, map_location=device, weights_only=False))

    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.projections.parameters():
        param.requires_grad = False
    logger.info("Shared transformer + projections frozen")

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    for d in active_domains:
        logger.info("Fine-tuning: %s", d)
        n_targets    = ntargets[d]
        train_loader = make_loader(train_records[d], n_targets, seq_len,
                                   domain_stride(d, train_paths[d], cfg), batch_sz, True,
                                   generator)
        val_loader   = make_loader(val_records[d],   n_targets, seq_len,
                                   domain_stride(d, val_paths[d], cfg), batch_sz, False)

        finetune_lr = lr if lr is not None else float(
            tcfg.get("finetune_lr", tcfg.get("initial_lr", 1e-3))
        )
        optimizer = torch.optim.AdamW(
            model.heads[d].parameters(), lr=finetune_lr, weight_decay=tcfg["weight_decay"]
        )
        scheduler = build_warmup_cosine_scheduler(optimizer, tcfg["finetune_epochs"], tcfg.get("warmup_epochs", 0))
        best_val   = float("inf")
        no_improve = 0
        history: list[dict] = []
        ckpt_path  = checkpoint_path(models_dir, "finetuned", d, flux_only, seed, active_domains)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, tcfg["finetune_epochs"] + 1):
            model.train()
            sse = cnt = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred  = model(x, domain=d)
                loss  = masked_mse_loss(pred, y)
                loss.backward()
                optimizer.step()
                valid = ~torch.isnan(y)
                if valid.any():
                    sse += ((pred - y)[valid] ** 2).sum().item()
                    cnt += valid.sum().item()
            scheduler.step()
            train_loss = sse / cnt if cnt > 0 else float("nan")

            if epoch % tcfg["eval_every_n_epochs"] == 0:
                model.eval()
                v_sse = v_cnt = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y  = x.to(device), y.to(device)
                        pred  = model(x, domain=d)
                        valid = ~torch.isnan(y)
                        if valid.any():
                            v_sse += ((pred - y)[valid] ** 2).sum().item()
                            v_cnt += valid.sum().item()
                val_loss = v_sse / v_cnt if v_cnt > 0 else float("nan")
                logger.info("  %s epoch %3d  train=%.4f  val=%.4f", d, epoch, train_loss, val_loss)
                history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    no_improve += 1
                    if no_improve >= tcfg["early_stopping_patience"]:
                        logger.info("  Early stopping for %s at epoch %d", d, epoch)
                        break

        logger.info("Finetune stage %s done. best_val=%.4f  checkpoint=%s", d, best_val, ckpt_path)
        hist_dir = stage_output_dir(eval_dir, "finetuned", d, flux_only, seed, active_domains)
        hist_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).round(4).to_csv(hist_dir / "history.csv", index=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        post_train_plots(model, val_records, scalers, domain_specs, seq_len, flux_only, device,
                         lambda dd, _d=d: stage_output_dir(eval_dir, "finetuned", _d, flux_only, seed, active_domains))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--flux-only", action="store_true",
                        help="Train the flux-only target-set variant: Arctic GPP/RECO only, "
                             "Rangeland GPP/RECO/Rm/Rg only, Amazon unaffected (no flux/pool "
                             "distinction). Reuses the existing full-target pkl/scaler — no "
                             "re-preprocessing. Outputs go to pretrained_fluxonly/ / "
                             "finetuned_fluxonly/ so they never collide with the full-target run.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Training RNG seed (weight init + minibatch shuffle order). Use "
                             "the same seed for --stage pretrain and the matching --stage "
                             "finetune run, so finetune loads that seed's pretrain checkpoint. "
                             "Omit for today's unseeded behavior. Does not affect the (fixed) "
                             "train/val data split.")
    parser.add_argument("--domains", type=str, default=None,
                        help="Ablation only (see ablation_test/ablation_description.md): "
                             "comma-separated subset of domains to pretrain/finetune on, e.g. "
                             "'arctic,amazon'. Omit for today's default (all domains: "
                             f"{','.join(DOMAINS)}). Pass the same value to the matching "
                             "04_evaluate.py call — it determines which checkpoint path is "
                             "written/read.")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    active_domains = args.domains.split(",") if args.domains else list(DOMAINS)
    unknown = set(active_domains) - set(DOMAINS)
    if unknown:
        raise ValueError(f"--domains contains unknown domain(s) {unknown} — must be a subset of {DOMAINS}")

    cfg = load_config("multi_domain")
    train_records, val_records, scalers, train_paths, val_paths = {}, {}, {}, {}, {}
    for d in active_domains:
        pre_dir = Path(cfg["paths"][d]["preprocessed_dir"])
        train_paths[d] = arctic_train_pkl_path(cfg) if d == "arctic" else pre_dir / "train.pkl"
        val_paths[d]   = pre_dir / "val.pkl"
        train_records[d] = load_pkl(train_paths[d])
        val_records[d]   = load_pkl(val_paths[d])
        with Path(cfg["paths"][d]["scaler"]).open("rb") as f:
            scalers[d] = pickle.load(f)
        if args.flux_only:
            orig_scaler = scalers[d]
            train_records[d], scalers[d] = apply_flux_only(d, train_records[d], orig_scaler)
            val_records[d], _ = apply_flux_only(d, val_records[d], orig_scaler)
    logger.info("Loaded train/val records for domains=%s (flux_only=%s)", active_domains, args.flux_only)

    lr: float | None = None
    if args.stage == "pretrain":
        lr = run_pretrain(cfg, train_records, val_records, scalers, train_paths, val_paths,
                          args.flux_only, active_domains, seed=args.seed)
    else:
        run_finetune(cfg, train_records, val_records, scalers, train_paths, val_paths,
                     args.flux_only, active_domains, lr=lr, seed=args.seed)


if __name__ == "__main__":
    main()
