"""
Multi-domain — Step 2: training.

See domains/multi_domain/multi_description.md § "Step 2 — Training".

Two stages controlled by --stage:
  pretrain  — joint mixed-step training across all three domains (Stage 1)
  finetune  — freeze shared weights, fine-tune each domain head independently (Stage 2)
"""

import argparse
import itertools
import logging
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config  # noqa: E402
from domains.multi_domain.model import MultiDomainModel  # noqa: E402
from shared.dataset import WindowedDataset, records_to_segments  # noqa: E402
from shared.evaluate import per_unit_metrics, predict_and_inverse, stack_by_target  # noqa: E402
from shared.plots import plot_metric_boxplot, plot_pred_vs_true  # noqa: E402
from shared.training import build_warmup_cosine_scheduler, masked_mse_loss, run_lr_finder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ["arctic", "amazon", "rangeland"]
DOMAIN_NTARGETS = {"arctic": 4, "amazon": 3, "rangeland": 10}
DOMAIN_TARGET_NAMES = {
    "arctic":    ["ALD", "GPP", "RECO", "VEGC"],
    "amazon":    ["discharge", "active_fire_count", "burned_area"],
    "rangeland": ["GPP", "RECO", "Rm", "Rg", "AGB", "BGB", "AGL", "BGL", "POC", "HOC"],
}
DOMAIN_ID_FIELDS = {
    "arctic":    ["grid", "y", "x", "ssp"],
    "amazon":    ["station_id"],
    "rangeland": ["site", "pft"],
}


def load_pkl(path: Path) -> list[dict]:
    with path.open("rb") as f:
        return pickle.load(f)


def make_loader(records: list[dict], n_targets: int, seq_len: int, stride: int,
                batch_size: int, shuffle: bool) -> DataLoader:
    segs, meta = records_to_segments(records)
    ds = WindowedDataset(segs, meta, n_targets, seq_len, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def val_loss_per_domain(model: MultiDomainModel, val_loaders: dict,
                        device: torch.device) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for d in DOMAINS:
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
                     domain_specs: dict, seq_len: int,
                     device: torch.device, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for d in DOMAINS:
        domain_model = lambda x, _d=d: model(x, domain=_d)  # default-arg capture avoids late-binding
        n_targets    = domain_specs[d]["nTargets"]
        target_names = DOMAIN_TARGET_NAMES[d]
        seg_meta, pred_list, obs_list = predict_and_inverse(
            domain_model, val_records[d], n_targets, seq_len, device, scalers[d]
        )
        pred_d, obs_d = stack_by_target(pred_list, obs_list, target_names)
        plot_pred_vs_true(pred_d, obs_d, save_path=out_dir / f"{d}_scatter.png")
        metrics_df = per_unit_metrics(seg_meta, pred_list, obs_list, target_names, DOMAIN_ID_FIELDS[d])
        plot_metric_boxplot(metrics_df, save_path=out_dir / f"{d}_boxplot.png")
        logger.info("Post-train plots saved: %s", out_dir / f"{d}_*.png")


class _LRProbeModel(nn.Module):
    """Wraps MultiDomainModel for LR finder; routes all batches through the Arctic branch.

    The shared transformer is domain-agnostic, so Arctic batches are a fair proxy for the
    multi-domain training signal. lr_finder.reset() restores the underlying model correctly
    because _base is a registered PyTorch submodule (all parameters are tracked).
    """

    def __init__(self, base: MultiDomainModel) -> None:
        super().__init__()
        self._base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._base(x, domain="arctic")


def run_pretrain(cfg: dict, train_records: dict, val_records: dict, scalers: dict) -> float:
    tcfg  = cfg["training"]
    seq_len  = cfg["model"]["seq_len"]
    stride   = cfg["preprocessing"]["stride"]
    batch_sz = tcfg["batch_size"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg["paths"]["models_dir"])
    eval_dir   = Path(cfg["paths"]["evaluation_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    nF_arctic = train_records["arctic"][0]["data"].shape[1] - 4
    domain_specs = {
        "arctic":    {"nFeatures": nF_arctic, "nTargets": 4},
        "amazon":    {"nFeatures": 14,        "nTargets": 3},
        "rangeland": {"nFeatures": 22,        "nTargets": 10},
    }
    logger.info("Device=%s | arctic nFeatures=%d", device, nF_arctic)

    train_loaders = {d: make_loader(train_records[d], DOMAIN_NTARGETS[d], seq_len, stride, batch_sz, True)
                     for d in DOMAINS}
    val_loaders   = {d: make_loader(val_records[d],   DOMAIN_NTARGETS[d], seq_len, stride, batch_sz, False)
                     for d in DOMAINS}
    logger.info("Train windows: %s", {d: len(train_loaders[d].dataset) for d in DOMAINS})

    model = MultiDomainModel(cfg, domain_specs).to(device)

    if tcfg.get("optimized_lr") is not None:
        lr = float(tcfg["optimized_lr"])
        logger.info("Using configured optimized_lr=%.3e", lr)
    else:
        eval_dir.mkdir(parents=True, exist_ok=True)
        probe = _LRProbeModel(model)
        lr = run_lr_finder(probe, train_loaders["arctic"], float(tcfg["initial_lr"]),
                           device, eval_dir / "lr_finder.png")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    scheduler = build_warmup_cosine_scheduler(optimizer, tcfg["pretrain_epochs"], tcfg.get("warmup_epochs", 0))

    steps_per_epoch = tcfg["steps_per_epoch"] or len(train_loaders["arctic"])
    domain_iters    = {d: iter(itertools.cycle(train_loaders[d])) for d in DOMAINS}

    best_val   = float("inf")
    no_improve = 0
    ckpt_path  = models_dir / "stage1_best.pt"

    for epoch in range(1, tcfg["pretrain_epochs"] + 1):
        model.train()
        epoch_sse = {d: 0.0 for d in DOMAINS}
        epoch_cnt = {d: 0.0 for d in DOMAINS}

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for d in DOMAINS:
                x, y  = next(domain_iters[d])
                x, y  = x.to(device), y.to(device)
                pred  = model(x, domain=d)
                loss_d = masked_mse_loss(pred, y)
                total_loss = total_loss + loss_d
                valid = ~torch.isnan(y)
                if valid.any():
                    epoch_sse[d] += ((pred - y)[valid] ** 2).sum().item()
                    epoch_cnt[d] += valid.sum().item()
            (total_loss / len(DOMAINS)).backward()
            optimizer.step()

        scheduler.step()
        train_str = "  ".join(
            f"{d}={epoch_sse[d]/epoch_cnt[d]:.4f}" if epoch_cnt[d] > 0 else f"{d}=nan"
            for d in DOMAINS
        )
        logger.info("Epoch %3d  train  %s", epoch, train_str)

        if epoch % tcfg["eval_every_n_epochs"] == 0:
            val_losses = val_loss_per_domain(model, val_loaders, device)
            mean_val   = sum(v for v in val_losses.values() if v == v) / len(DOMAINS)
            val_str    = "  ".join(f"{d}={v:.4f}" for d, v in val_losses.items())
            logger.info("Epoch %3d  val    %s  mean=%.4f", epoch, val_str, mean_val)

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

    logger.info("Stage 1 complete. best_val=%.4f  checkpoint=%s", best_val, ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    post_train_plots(model, val_records, scalers, domain_specs, seq_len, device, eval_dir / "stage1")
    return lr


def run_finetune(cfg: dict, train_records: dict, val_records: dict, scalers: dict,
                 lr: float | None = None) -> None:
    tcfg  = cfg["training"]
    seq_len  = cfg["model"]["seq_len"]
    stride   = cfg["preprocessing"]["stride"]
    batch_sz = tcfg["batch_size"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg["paths"]["models_dir"])
    eval_dir   = Path(cfg["paths"]["evaluation_dir"])

    nF_arctic = train_records["arctic"][0]["data"].shape[1] - 4
    domain_specs = {
        "arctic":    {"nFeatures": nF_arctic, "nTargets": 4},
        "amazon":    {"nFeatures": 14,        "nTargets": 3},
        "rangeland": {"nFeatures": 22,        "nTargets": 10},
    }

    stage1_ckpt = models_dir / "stage1_best.pt"
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"{stage1_ckpt} — run --stage pretrain first")

    model = MultiDomainModel(cfg, domain_specs).to(device)
    model.load_state_dict(torch.load(stage1_ckpt, map_location=device, weights_only=False))

    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.projections.parameters():
        param.requires_grad = False
    logger.info("Shared transformer + projections frozen")

    for d in DOMAINS:
        logger.info("Fine-tuning: %s", d)
        n_targets    = DOMAIN_NTARGETS[d]
        train_loader = make_loader(train_records[d], n_targets, seq_len, stride, batch_sz, True)
        val_loader   = make_loader(val_records[d],   n_targets, seq_len, stride, batch_sz, False)

        finetune_lr = lr if lr is not None else float(
            tcfg.get("learning_rate", tcfg.get("initial_lr", 1e-3))
        )
        optimizer = torch.optim.AdamW(
            model.heads[d].parameters(), lr=finetune_lr, weight_decay=tcfg["weight_decay"]
        )
        scheduler = build_warmup_cosine_scheduler(optimizer, tcfg["finetune_epochs"], tcfg.get("warmup_epochs", 0))
        best_val   = float("inf")
        no_improve = 0
        ckpt_path  = models_dir / f"stage2_{d}_best.pt"

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

                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    no_improve += 1
                    if no_improve >= tcfg["early_stopping_patience"]:
                        logger.info("  Early stopping for %s at epoch %d", d, epoch)
                        break

        logger.info("Stage 2 %s done. best_val=%.4f  checkpoint=%s", d, best_val, ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        post_train_plots(model, val_records, scalers, domain_specs, seq_len, device,
                         eval_dir / f"stage2_{d}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    args = parser.parse_args()

    cfg = load_config("multi_domain")
    train_records, val_records, scalers = {}, {}, {}
    for d in DOMAINS:
        pre_dir = Path(cfg["paths"][d]["preprocessed_dir"])
        train_records[d] = load_pkl(pre_dir / "train.pkl")
        val_records[d]   = load_pkl(pre_dir / "val.pkl")
        with Path(cfg["paths"][d]["scaler"]).open("rb") as f:
            scalers[d] = pickle.load(f)
    logger.info("Loaded train/val records for all domains")

    lr: float | None = None
    if args.stage == "pretrain":
        lr = run_pretrain(cfg, train_records, val_records, scalers)
    else:
        run_finetune(cfg, train_records, val_records, scalers, lr=lr)


if __name__ == "__main__":
    main()
