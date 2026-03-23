"""
Mitosis heatmap training script. Single-GPU, single-file style.

prepare.py contains the fixed data infrastructure and fixed evaluation metric.
This file is the experiment file:
- augmentation
- model
- optimizer
- hyperparameters
- training loop
- logging
"""

import os
import time
import copy
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from prepare import BaseMitosisPointDataset, make_base_datasets, evaluate_fixed_metric


# ---------------------------------------------------------------------------
# Augmentation (experiment-owned)
# ---------------------------------------------------------------------------

def aug_patch(x, y, rng):
    """
    x: (C,H,W), y: (1,H,W)
    Everything here is fair game for autoresearch.
    """
    if rng.random() < HFLIP_PROB:
        x = x[:, :, ::-1].copy()
        y = y[:, :, ::-1].copy()

    if rng.random() < VFLIP_PROB:
        x = x[:, ::-1, :].copy()
        y = y[:, ::-1, :].copy()

    if ROT90:
        k = int(rng.integers(0, 4))
        if k:
            x = np.rot90(x, k, axes=(1, 2)).copy()
            y = np.rot90(y, k, axes=(1, 2)).copy()

    if rng.random() < INTENSITY_PROB:
        scale = float(rng.uniform(INTENSITY_SCALE_MIN, INTENSITY_SCALE_MAX))
        shift = float(rng.uniform(INTENSITY_SHIFT_MIN, INTENSITY_SHIFT_MAX))
        x = np.clip(x * scale + shift, 0.0, 1.0)

    if rng.random() < GAMMA_PROB:
        gamma = float(rng.uniform(GAMMA_MIN, GAMMA_MAX))
        x = np.clip(x, 0.0, 1.0) ** gamma

    if rng.random() < NOISE_PROB and NOISE_STD > 0:
        noise = rng.normal(0.0, NOISE_STD, size=x.shape).astype(np.float32)
        x = np.clip(x + noise, 0.0, 1.0)

    return x.astype(np.float32), y.astype(np.float32)


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, seed=0, use_augmentation=True):
        self.base_dataset = base_dataset
        self.rng = np.random.default_rng(seed)
        self.use_augmentation = bool(use_augmentation)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        if not self.use_augmentation:
            return x, y
        x_np, y_np = x.numpy(), y.numpy()
        x_np, y_np = aug_patch(x_np, y_np, self.rng)
        return torch.from_numpy(x_np), torch.from_numpy(y_np)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class UNetConfig:
    in_ch: int = 3
    base: int = 32


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=16):
        super().__init__()
        g = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bott = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bott(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, loss_fn, optimizer, scaler, device, amp):
    model.train()
    losses = []
    n_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and amp)):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))
        n_samples += x.size(0)

    return float(np.mean(losses)) if losses else float("nan"), n_samples


# ---------------------------------------------------------------------------
# Hyperparameters (edit directly)
# ---------------------------------------------------------------------------

# Data
DATA_DIR = "./002_nuc_data"
POS_CSV = "./002_labels/pos.csv"
NEG_CSV = "./002_labels/neg.csv"
OUT_DIR = "./output"

FMT = "t%03d.tif"
N_FRAMES = None

# Task
PATCH = 256
SIGMA = 5.0
TEMPORAL = True
JITTER = 8
VAL_FRAC = 0.2
SEED = 0
POS_OVERSAMPLE = 3

# Optimization
BATCH_SIZE = 8
TRAIN_SECONDS = 300  # 5-minute wall clock budget (excluding startup/eval)
LR = 2e-3
WEIGHT_DECAY = 1e-5
AMP = True
NUM_WORKERS = 0

# Training loss choice (free to optimize)
USE_FOCAL = True
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# Model size
BASE = 64

# Augmentation
USE_AUGMENTATION = True
HFLIP_PROB = 0.5
VFLIP_PROB = 0.5
ROT90 = True
INTENSITY_PROB = 0.8
INTENSITY_SCALE_MIN = 0.8
INTENSITY_SCALE_MAX = 1.2
INTENSITY_SHIFT_MIN = -0.05
INTENSITY_SHIFT_MAX = 0.05
GAMMA_PROB = 0.5
GAMMA_MIN = 0.8
GAMMA_MAX = 1.2
NOISE_PROB = 0.5
NOISE_STD = 0.02

# Optional init
INIT_CKPT = None
RESET_HEAD = False


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

train_base, val_base, meta = make_base_datasets(
    data_dir=DATA_DIR,
    pos_csv=POS_CSV,
    neg_csv=NEG_CSV,
    fmt=FMT,
    n_frames=N_FRAMES,
    patch=PATCH,
    sigma=SIGMA,
    temporal=TEMPORAL,
    jitter=JITTER,
    val_frac=VAL_FRAC,
    seed=SEED,
)

train_samples = meta["train_samples"]
val_samples = meta["val_samples"]

if POS_OVERSAMPLE > 1:
    pos = [s for s in train_samples if s.label == 1]
    neg = [s for s in train_samples if s.label == 0]
    oversampled_train = (pos * POS_OVERSAMPLE) + neg

    train_base = BaseMitosisPointDataset(
        paths=meta["paths"],
        samples=oversampled_train,
        patch=PATCH,
        sigma=SIGMA,
        temporal=TEMPORAL,
        jitter=JITTER,
        seed=SEED,
        cache_frames=True,
    )
    train_samples = oversampled_train

train_ds = AugmentedDataset(train_base, seed=SEED + 123, use_augmentation=USE_AUGMENTATION)
val_ds = AugmentedDataset(val_base, seed=SEED + 456, use_augmentation=False)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

config = UNetConfig(in_ch=(3 if TEMPORAL else 1), base=BASE)
model = UNetSmall(in_ch=config.in_ch, base=config.base).to(device)

if INIT_CKPT is not None:
    ckpt = torch.load(INIT_CKPT, map_location="cpu")
    state = ckpt["model"]
    if RESET_HEAD:
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[init_ckpt] loaded from {INIT_CKPT}")
    print("  missing:", missing)
    print("  unexpected:", unexpected)

train_loss_fn = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA) if USE_FOCAL else nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=LR * 0.01)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and AMP))

num_params = count_parameters(model)
peak_vram_mb = 0.0

print(f"Device: {device}")
print(f"Model config: {asdict(config)}")
print(f"Frame shape: {meta['frame_shape']}")
print(f"Num frames: {meta['num_frames']}")
print(f"All samples: {len(meta['all_samples'])}")
print(f"Train samples: {len(train_samples)}")
print(f"Val samples: {len(val_samples)}")
print(f"Num params: {num_params:,}")
print("---")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

final_val_metric = float("nan")
best_val_metric = float("inf")
best_epoch = 0
epochs_completed = 0

t_start_training = time.time()

epoch = 0
while True:
    elapsed = time.time() - t_start_training
    if elapsed >= TRAIN_SECONDS:
        break

    epoch += 1
    t0 = time.time()

    train_loss, seen_samples = train_epoch(
        model, train_loader, train_loss_fn, optimizer, scaler, device, AMP
    )
    val_metric = evaluate_fixed_metric(model, val_loader, device)
    final_val_metric = val_metric
    epochs_completed = epoch

    if device.type == "cuda":
        peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)

    dt = time.time() - t0
    samples_per_sec = seen_samples / max(dt, 1e-6)
    elapsed_total = time.time() - t_start_training

    torch.save(
        {"epoch": epoch, "model": model.state_dict()},
        os.path.join(OUT_DIR, "last.pt"),
    )

    scheduler.step()

    if val_metric < best_val_metric:
        best_val_metric = val_metric
        best_epoch = epoch
        torch.save(
            {"epoch": epoch, "model": copy.deepcopy(model.state_dict())},
            os.path.join(OUT_DIR, "best.pt"),
        )

    print(
        f"epoch {epoch:03d} | "
        f"train_loss: {train_loss:.6f} | "
        f"val_metric: {val_metric:.6f} | "
        f"elapsed: {elapsed_total:.1f}s/{TRAIN_SECONDS}s | "
        f"dt: {dt:.1f}s | "
        f"samples/sec: {samples_per_sec:.1f}"
    )

training_seconds = time.time() - t_start_training
total_seconds = time.time() - t_start


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

print("---")
print(f"final_val_metric:  {final_val_metric:.6f}")
print(f"epochs_completed:  {epochs_completed}")
print(f"training_seconds:  {training_seconds:.1f}")
print(f"total_seconds:     {total_seconds:.1f}")
print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
print(f"num_params_M:      {num_params / 1e6:.3f}")
print(f"batch_size:        {BATCH_SIZE}")
print(f"patch:             {PATCH}")
print(f"temporal:          {TEMPORAL}")
print(f"augmentation:      {USE_AUGMENTATION}")