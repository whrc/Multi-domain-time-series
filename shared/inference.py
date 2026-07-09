"""
Shared inference helper — used by every domain's 03_predict.py (and future multi-domain).

Runs the causal model densely (stride = 1) over each segment and keeps the prediction
at the *last* position of every window, where the model has seen the most context. The
first ``seq_len - 1`` steps of each segment have no prediction and stay NaN. Predictions
come back grouped per segment so each domain can reshape them to its own output format.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from shared.dataset import WindowedDataset


def predict_last_position(
    model: torch.nn.Module,
    dataset: WindowedDataset,
    device: torch.device,
    batch_size: int = 8192,
    num_workers: int = 4,
) -> list[np.ndarray]:
    """Return one ``(T_seg, num_targets)`` prediction array per segment (NaN where uncovered).

    ``dataset`` must be built with ``stride = 1`` to cover every position, so a dense pass
    touches far more windows than training ever did (e.g. ~140x more than a stride=150
    training set, for seq_len=12 over a ~2400-step series) — this is what makes val/test
    evaluation the slow step, not the forward pass itself. No backward pass here, so
    batch_size can go far higher than training's without memory pressure; num_workers
    overlaps the per-window CPU slicing (shared/dataset.py's __getitem__) with GPU compute
    instead of doing it synchronously on the main process.
    """
    if dataset.stride != 1:
        raise ValueError(f"predict_last_position requires stride=1, got {dataset.stride}")
    seq_len = dataset.seq_len
    num_targets = dataset.num_targets
    preds = [np.full((seg.shape[0], num_targets), np.nan) for seg in dataset.segments]
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model.eval()
    batches = []
    with torch.no_grad():
        for x, _ in loader:
            batches.append(model(x.to(device, non_blocking=True))[:, -1, :].cpu().numpy())
    if not batches:
        # No windows at all (e.g. every segment shorter than seq_len) - nothing to assign,
        # preds is already all-NaN from its initialization above.
        return preds
    all_preds = np.concatenate(batches, axis=0)  # (total_windows, num_targets), dataset.windows order

    # dataset.windows is built segment-major with stride=1 (WindowedDataset's nested
    # comprehension), so each segment's windows are one contiguous run of consecutive start
    # positions - a single vectorized slice assignment per segment reconstructs them, instead
    # of a per-window Python loop (millions of iterations) that dominated wall time on top of
    # the batched forward pass above.
    wi = 0
    for si, seg in enumerate(dataset.segments):
        n = max(0, seg.shape[0] - seq_len + 1)
        if n == 0:
            continue
        preds[si][seq_len - 1:seq_len - 1 + n] = all_preds[wi:wi + n]
        wi += n
    return preds
