"""
Shared sliding-window dataset — used by every domain (and future multi-domain work).

A domain's preprocessed records are flattened into a list of contiguous segment
arrays, each shaped ``(T_seg, nFeatures + num_targets)`` with targets in the last
``num_targets`` columns. ``WindowedDataset`` slides a fixed-length window over every
segment and serves ``(input, target)`` pairs. The same class backs both training
(configured stride) and prediction (stride = 1); ``segments``/``seg_meta``/``windows``
are exposed so the inference helper can map predictions back to their source positions.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class WindowedDataset(Dataset):
    def __init__(
        self,
        segments: list[np.ndarray],
        seg_meta: list[Any],
        num_targets: int,
        seq_len: int,
        stride: int,
    ) -> None:
        if len(segments) != len(seg_meta):
            raise ValueError(f"segments ({len(segments)}) and seg_meta ({len(seg_meta)}) length mismatch.")
        self.segments = segments
        self.seg_meta = seg_meta
        self.num_targets = num_targets
        self.seq_len = seq_len
        self.stride = stride
        # Flat index of (segment_index, window_start) over every long-enough segment.
        self.windows: list[tuple[int, int]] = [
            (si, start)
            for si, seg in enumerate(segments)
            if seg.shape[0] >= seq_len
            for start in range(0, seg.shape[0] - seq_len + 1, stride)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        si, start = self.windows[idx]
        window = self.segments[si][start : start + self.seq_len]
        x = torch.from_numpy(window[:, : -self.num_targets]).float()
        y = torch.from_numpy(window[:, -self.num_targets :]).float()
        return x, y


def records_to_segments(records: list[dict]) -> tuple[list[np.ndarray], list[dict]]:
    """Flatten preprocessed records into parallel (segment, metadata) lists.

    Handles both record shapes: Amazon/Rangeland store a ``segments`` list; Arctic
    stores a single ``data`` array (treated as one segment). Each metadata dict carries
    the record's non-array fields plus ``seg_idx`` so callers can reconstruct positions.
    """
    segments: list[np.ndarray] = []
    seg_meta: list[dict] = []
    for r in records:
        seglist = r["segments"] if "segments" in r else [r["data"]]
        base = {k: v for k, v in r.items() if k not in ("segments", "data")}
        for j, seg in enumerate(seglist):
            segments.append(seg)
            seg_meta.append({**base, "seg_idx": j})
    return segments, seg_meta
