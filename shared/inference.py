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
    batch_size: int = 256,
) -> list[np.ndarray]:
    """Return one ``(T_seg, num_targets)`` prediction array per segment (NaN where uncovered).

    ``dataset`` must be built with ``stride = 1`` to cover every position. The DataLoader
    runs unshuffled so window order matches ``dataset.windows``.
    """
    if dataset.stride != 1:
        raise ValueError(f"predict_last_position requires stride=1, got {dataset.stride}")
    seq_len = dataset.seq_len
    num_targets = dataset.num_targets
    preds = [np.full((seg.shape[0], num_targets), np.nan) for seg in dataset.segments]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    wi = 0
    with torch.no_grad():
        for x, _ in loader:
            last = model(x.to(device)).cpu().numpy()[:, -1, :]  # (B, num_targets)
            for b in range(last.shape[0]):
                si, start = dataset.windows[wi]
                preds[si][start + seq_len - 1] = last[b]
                wi += 1
    return preds
