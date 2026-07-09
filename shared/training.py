"""
Shared training utilities — used by every domain's 02_train.py (and future multi-domain).

Provides the masked-MSE loss, an optional learning-rate range test, and a single
training loop that checkpoints on the best validation loss with early stopping. All
domains differ only in their data and the number of features/targets; the optimisation
machinery here is identical, so the multi-domain model can reuse it unchanged.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def masked_mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over valid (non-NaN) target positions, across all targets.

    Returns a graph-connected zero when a batch has no valid positions, so the step
    is a no-op rather than a NaN.
    """
    valid = ~torch.isnan(target)
    if not valid.any():
        return pred.sum() * 0.0
    diff = (pred - target)[valid]
    return (diff ** 2).mean()


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    target_names: list[str],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Masked-MSE on a loader, overall and per target. Targets in the last columns."""
    n = len(target_names)
    sse = np.zeros(n)
    cnt = np.zeros(n)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device, non_blocking=True)).cpu().numpy()
            obs = y.numpy()
            valid = ~np.isnan(obs)
            err = np.where(valid, (pred - obs) ** 2, 0.0)
            sse += err.sum(axis=(0, 1))
            cnt += valid.sum(axis=(0, 1))
    per_target = {
        name: float(sse[i] / cnt[i]) if cnt[i] > 0 else float("nan")
        for i, name in enumerate(target_names)
    }
    overall = float(sse.sum() / cnt.sum()) if cnt.sum() > 0 else float("nan")
    return overall, per_target


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear LR warmup for warmup_epochs, then cosine decay over remaining epochs."""
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_epochs - warmup_epochs)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


def run_lr_finder(
    model: torch.nn.Module,
    train_loader: DataLoader,
    initial_lr: float,
    device: torch.device,
    save_path: Path,
    num_iter: int = 100,
    weight_decay: float = 1e-4,
) -> float:
    """Run a learning-rate range test and return the suggested LR (steepest descent).

    Saves the loss-vs-LR curve to ``save_path`` and restores the model/optimizer state
    so training starts clean. Falls back to ``initial_lr`` if the test is degenerate.
    """
    import matplotlib.pyplot as plt
    from torch_lr_finder import LRFinder

    logger.info("Starting LR range test (%d iterations)...", num_iter)
    t0 = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    lr_finder = LRFinder(model, optimizer, masked_mse_loss, device=device)
    lr_finder.range_test(train_loader, end_lr=1.0, num_iter=num_iter)

    lrs = np.array(lr_finder.history["lr"])
    losses = np.array(lr_finder.history["loss"])
    finite = np.isfinite(losses)
    suggested = initial_lr
    if finite.sum() >= 2:
        lrs, losses = lrs[finite], losses[finite]
        suggested = float(lrs[np.gradient(losses).argmin()])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lrs, losses, color="#0072B2")
    ax.axvline(suggested, color="#D55E00", linestyle="--", label=f"suggested={suggested:.2e}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss")
    ax.set_title("LR range test")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    lr_finder.reset()
    logger.info("LR finder suggested lr=%.3e in %.1fs (curve saved to %s)", suggested, time.time() - t0, save_path)
    return suggested


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    target_names: list[str],
    lr: float,
    device: torch.device,
    checkpoint_path: Path,
    num_features: int,
) -> dict:
    """Train with AdamW + masked MSE, checkpoint on best val loss, stop early.

    Returns a history dict: ``train_loss``, ``val_loss`` (per-epoch lists),
    ``per_target_val`` (target -> per-epoch list), ``best_epoch``, ``best_val_loss``.
    """
    tcfg = cfg["training"]
    num_epochs: int = tcfg["num_epochs"]
    eval_every: int = tcfg["eval_every_n_epochs"]
    patience: int = tcfg["early_stopping_patience"]

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    scheduler = None
    if tcfg["lr_scheduler"] == "cosine":
        scheduler = build_warmup_cosine_scheduler(optimizer, num_epochs, tcfg.get("warmup_epochs", 0))
    elif tcfg["lr_scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.1)

    history: dict = {"train_loss": [], "val_loss": [], "per_target_val": {n: [] for n in target_names}}
    best_val = float("inf")
    best_epoch = -1
    evals_since_improve = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        sse = 0.0
        cnt = 0.0
        epoch_start = time.time()
        data_time = 0.0
        t_prev = time.time()
        for x, y in train_loader:
            data_time += time.time() - t_prev
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)
            loss = masked_mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            valid = ~torch.isnan(y)
            sse += float(((pred - y)[valid] ** 2).sum().item()) if valid.any() else 0.0
            cnt += float(valid.sum().item())
            t_prev = time.time()
        train_loss = sse / cnt if cnt > 0 else float("nan")
        history["train_loss"].append(train_loss)
        epoch_time = time.time() - epoch_start
        logger.info(
            "Epoch %3d/%3d done in %.1fs (data=%.1fs compute=%.1fs) | train=%.4f",
            epoch, num_epochs, epoch_time, data_time, epoch_time - data_time, train_loss,
        )

        if scheduler is not None:
            scheduler.step()

        if epoch % eval_every == 0:
            val_loss, per_target = _evaluate(model, val_loader, target_names, device)
            history["val_loss"].append(val_loss)
            for name in target_names:
                history["per_target_val"][name].append(per_target[name])
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info("Epoch %3d | train=%.4f  val=%.4f  lr=%.2e", epoch, train_loss, val_loss, current_lr)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                evals_since_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "num_features": num_features,
                        "num_targets": len(target_names),
                        "cfg": cfg,
                    },
                    checkpoint_path,
                )
            else:
                evals_since_improve += 1
                if evals_since_improve >= patience:
                    logger.info("Early stopping at epoch %d (best val=%.4f @ epoch %d)", epoch, best_val, best_epoch)
                    break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return history
