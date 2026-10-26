"""Dense (stride-1) inference keeping each window's last-position prediction; the first
seq_len - 1 steps of each segment stay NaN.
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
    """Return one ``(T_seg, num_targets)`` prediction array per segment (NaN where uncovered)."""
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
        return preds
    all_preds = np.concatenate(batches, axis=0)  # (total_windows, num_targets), dataset.windows order

    # Windows are segment-major with stride 1, so each segment's predictions are one contiguous slice.
    wi = 0
    for si, seg in enumerate(dataset.segments):
        n = max(0, seg.shape[0] - seq_len + 1)
        if n == 0:
            continue
        preds[si][seq_len - 1:seq_len - 1 + n] = all_preds[wi:wi + n]
        wi += n
    return preds
