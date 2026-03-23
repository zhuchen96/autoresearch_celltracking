"""
Fixed data/runtime utilities for mitosis heatmap training.

This file is the stable infrastructure layer:
- frame path discovery
- TIFF loading
- frame cache
- point CSV loading
- train/val split
- normalization
- Gaussian target generation
- base dataset (no augmentation)
- fixed evaluation metric

Do not modify during autoresearch experiments.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FMT = "t%03d.tif"
DEFAULT_PATCH = 256
DEFAULT_SIGMA = 5.0
DEFAULT_VAL_FRAC = 0.2
DEFAULT_SEED = 0


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def build_paths(folder: str, fmt: str = DEFAULT_FMT, n_frames: Optional[int] = None) -> List[str]:
    if n_frames is None:
        paths = []
        for i in range(10000):
            p = os.path.join(folder, fmt % i)
            if os.path.exists(p):
                paths.append(p)
            else:
                if len(paths) > 0:
                    break
        if not paths:
            raise FileNotFoundError(f"No files found like {os.path.join(folder, fmt % 0)}")
        return paths
    else:
        paths = [os.path.join(folder, fmt % i) for i in range(n_frames)]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} files, e.g. {missing[0]}")
        return paths


class FrameCache:
    """Tiny LRU-ish cache for TIFF reads."""
    def __init__(self, max_items: int = 16):
        self.max_items = max_items
        self._store: Dict[int, np.ndarray] = {}
        self._order: List[int] = []

    def get(self, t: int):
        if t in self._store:
            self._order.remove(t)
            self._order.append(t)
            return self._store[t]
        return None

    def put(self, t: int, arr: np.ndarray):
        if t in self._store:
            self._order.remove(t)
        self._store[t] = arr
        self._order.append(t)
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._store.pop(old, None)


def read_frame(paths: List[str], t: int, cache: Optional[FrameCache] = None) -> np.ndarray:
    t = int(max(0, min(len(paths) - 1, int(t))))

    if cache is not None:
        arr = cache.get(t)
        if arr is not None:
            return arr

    arr = tifffile.imread(paths[t])
    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D frame, got shape {arr.shape} for {paths[t]}")

    if cache is not None:
        cache.put(t, arr)
    return arr


# ---------------------------------------------------------------------------
# Target / normalization
# ---------------------------------------------------------------------------

def gaussian2d(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def robust_norm_patch(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32)
    lo, hi = np.percentile(p, [1, 99])
    return np.clip((p - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Labels / split
# ---------------------------------------------------------------------------

@dataclass
class PointSample:
    t: int
    y: int
    x: int
    label: int  # 1=pos, 0=neg


def load_points(pos_csv: str, neg_csv: str) -> List[PointSample]:
    pos = pd.read_csv(pos_csv)
    neg = pd.read_csv(neg_csv)

    for c in ["t", "y", "x"]:
        if c not in pos.columns or c not in neg.columns:
            raise ValueError("CSV must have columns t,y,x")

    samples: List[PointSample] = []
    for row in pos[["t", "y", "x"]].itertuples(index=False):
        samples.append(PointSample(int(round(row.t)), int(round(row.y)), int(round(row.x)), 1))
    for row in neg[["t", "y", "x"]].itertuples(index=False):
        samples.append(PointSample(int(round(row.t)), int(round(row.y)), int(round(row.x)), 0))
    return samples


def split_train_val(samples: List[PointSample], val_frac: float = DEFAULT_VAL_FRAC, seed: int = DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(samples) * val_frac)))
    val_idx = set(idx[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BaseMitosisPointDataset(Dataset):
    """
    Base dataset with no augmentation.
    train.py owns augmentation so it can be optimized experimentally.
    """

    def __init__(
        self,
        paths: List[str],
        samples: List[PointSample],
        patch: int = DEFAULT_PATCH,
        sigma: float = DEFAULT_SIGMA,
        temporal: bool = True,
        jitter: int = 0,
        seed: int = DEFAULT_SEED,
        cache_frames: bool = True,
    ):
        self.paths = paths
        self.samples = samples
        self.patch = int(patch)
        self.sigma = float(sigma)
        self.temporal = bool(temporal)
        self.jitter = int(jitter)
        self.rng = np.random.default_rng(seed)
        self.cache = FrameCache(max_items=32) if cache_frames else None

        f0 = read_frame(self.paths, 0, cache=self.cache)
        self.H, self.W = int(f0.shape[0]), int(f0.shape[1])

        if self.patch > self.H or self.patch > self.W:
            raise ValueError(f"Patch size {self.patch} is larger than frame size {(self.H, self.W)}")

    def __len__(self):
        return len(self.samples)

    def _extract_patch(self, frame: np.ndarray, cy: int, cx: int) -> Tuple[np.ndarray, int, int]:
        r = self.patch // 2
        y0 = int(np.clip(cy - r, 0, self.H - self.patch))
        x0 = int(np.clip(cx - r, 0, self.W - self.patch))
        p = frame[y0:y0 + self.patch, x0:x0 + self.patch]
        return p, y0, x0

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        jy = int(self.rng.integers(-self.jitter, self.jitter + 1)) if self.jitter > 0 else 0
        jx = int(self.rng.integers(-self.jitter, self.jitter + 1)) if self.jitter > 0 else 0
        cy = int(np.clip(s.y + jy, 0, self.H - 1))
        cx = int(np.clip(s.x + jx, 0, self.W - 1))

        if self.temporal:
            t0 = max(0, min(len(self.paths) - 1, s.t - 1))
            t1 = max(0, min(len(self.paths) - 1, s.t))
            t2 = max(0, min(len(self.paths) - 1, s.t + 1))

            I0 = read_frame(self.paths, t0, cache=self.cache)
            I1 = read_frame(self.paths, t1, cache=self.cache)
            I2 = read_frame(self.paths, t2, cache=self.cache)

            p0, y0, x0 = self._extract_patch(I0, cy, cx)
            p1, _, _ = self._extract_patch(I1, cy, cx)
            p2, _, _ = self._extract_patch(I2, cy, cx)

            x = np.stack([robust_norm_patch(p0), robust_norm_patch(p1), robust_norm_patch(p2)], axis=0)
        else:
            I1 = read_frame(self.paths, s.t, cache=self.cache)
            p1, y0, x0 = self._extract_patch(I1, cy, cx)
            x = robust_norm_patch(p1)[None, ...]

        y = np.zeros((1, self.patch, self.patch), dtype=np.float32)
        if s.label == 1:
            py = cy - y0
            px = cx - x0
            y[0] = gaussian2d(self.patch, self.patch, py, px, self.sigma)

        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def make_base_datasets(
    data_dir: str,
    pos_csv: str,
    neg_csv: str,
    fmt: str = DEFAULT_FMT,
    n_frames: Optional[int] = None,
    patch: int = DEFAULT_PATCH,
    sigma: float = DEFAULT_SIGMA,
    temporal: bool = True,
    jitter: int = 8,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = DEFAULT_SEED,
):
    paths = build_paths(data_dir, fmt=fmt, n_frames=n_frames)
    all_samples = load_points(pos_csv, neg_csv)
    train_samples, val_samples = split_train_val(all_samples, val_frac=val_frac, seed=seed)

    train_ds = BaseMitosisPointDataset(
        paths=paths,
        samples=train_samples,
        patch=patch,
        sigma=sigma,
        temporal=temporal,
        jitter=jitter,
        seed=seed,
        cache_frames=True,
    )
    val_ds = BaseMitosisPointDataset(
        paths=paths,
        samples=val_samples,
        patch=patch,
        sigma=sigma,
        temporal=temporal,
        jitter=0,
        seed=seed + 1,
        cache_frames=True,
    )

    meta = {
        "paths": paths,
        "all_samples": all_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "num_frames": len(paths),
        "frame_shape": (train_ds.H, train_ds.W),
    }
    return train_ds, val_ds, meta


# ---------------------------------------------------------------------------
# Fixed evaluation metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_fixed_metric(model, loader, device):
    """
    Fixed validation metric for autoresearch experiments.

    This metric must not be changed during experiments.
    We use BCEWithLogitsLoss on the validation heatmaps, regardless of the
    training loss used in train.py.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")
