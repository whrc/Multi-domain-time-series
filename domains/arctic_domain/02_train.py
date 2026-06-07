# domains/arctic_domain/02_train.py

import importlib.util
import logging
import math
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.config import load_config

DOMAIN = os.environ.get("ARCTIC_DOMAIN", "arctic_domain")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_TARGETS = 4  # ALD, GPP, RECO, VEGC


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(cfg: dict) -> tuple[list, list, int]:
    """Load preprocessed splits and derive num_features from the scaler."""
    base = Path(cfg["paths"]["preprocessed_dir"])
    with open(base / "train.pkl", "rb") as f:
        train_records = pickle.load(f)
    with open(base / "val.pkl", "rb") as f:
        val_records = pickle.load(f)
    with open(cfg["paths"]["scaler"], "rb") as f:
        scaler = pickle.load(f)

    num_features = len(scaler["mean"]) - NUM_TARGETS
    log.info("Loaded %d train / %d val records  num_features=%d",
             len(train_records), len(val_records), num_features)
    return train_records, val_records, num_features


def _load_arctic_dataset_class(script_path: Path):
    """Import ArcticDataset from 01_preprocess.py (digit prefix prevents normal import)."""
    spec = importlib.util.spec_from_file_location("preprocess", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ArcticDataset


def build_loaders(
    train_records: list, val_records: list, cfg: dict, num_features: int
) -> tuple[DataLoader, DataLoader]:
    preprocess_path = Path(__file__).parent / "01_preprocess.py"
    ArcticDataset = _load_arctic_dataset_class(preprocess_path)

    train_ds = ArcticDataset(train_records, cfg, num_features)
    val_ds   = ArcticDataset(val_records,   cfg, num_features)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    log.info("DataLoaders — train batches=%d  val batches=%d  batch_size=%d",
             len(train_loader), len(val_loader), batch_size)
    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, num_features: int, num_targets: int) -> nn.Module:
    arch = cfg["model"]["architecture"]
    if arch == "transformer":
        from models.transformer import TransformerModel
        model = TransformerModel(num_features, num_targets, cfg)
    elif arch == "lstm":
        from models.lstm import LSTMModel
        model = LSTMModel(num_features, num_targets, cfg)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Expected 'transformer' or 'lstm'.")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s  params=%s", arch, f"{n_params:,}")
    return model


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
    return total_loss / len(loader)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    num_features: int,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "num_features": num_features,
            "cfg": cfg,
        },
        path,
    )
    log.info("Checkpoint saved: %s  (val_loss=%.4f)", path, val_loss)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config(DOMAIN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train_records, val_records, num_features = load_data(cfg)
    train_loader, val_loader = build_loaders(train_records, val_records, cfg, num_features)

    model = build_model(cfg, num_features, NUM_TARGETS).to(device)

    t = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t["num_epochs"]
    )

    best_val_loss = math.inf
    patience_counter = 0
    checkpoint_path = Path(cfg["paths"]["best_model"])

    for epoch in range(1, t["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        log.info(
            "Epoch %3d | train=%.4f  val=%.4f  lr=%.2e",
            epoch, train_loss, val_loss, current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss, num_features, cfg)
        else:
            patience_counter += 1
            if patience_counter >= t["early_stopping_patience"]:
                log.info("Early stopping triggered at epoch %d (patience=%d)", epoch, t["early_stopping_patience"])
                break

    log.info("Training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
