#!/usr/bin/env python3
"""
Hybrid Dual-Stream PhysioNet Gait (PD & Severity)
- Novel dual-stream encoder: temporal convolutional stack + frequency (rFFT) stack
- Two specialized models/heads trained separately (PD via focal + class-balancing, Severity via CE + label smoothing)
- SAM optimizer (sharpness-aware minimization) + EMA weights for robust validation/generalization
- Test-time augmentation (noise + time masking) and subject-level probability aggregation

Goals (target on typical splits):
- PD subject accuracy > 0.92
- Severity subject accuracy > 0.88

Run:
  python gait_hybrid_dualstream.py --train

Notes:
- Uses joint stratification by (PD × Severity) to ensure each severity tier appears in each split
- Evaluates with EMA weights and TTA for stable subject-level metrics
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from contextlib import nullcontext
from torch.optim.swa_utils import AveragedModel

# ==================== CONFIG ====================
DATASET_DIR = "."
DEMOGRAPHICS_FILE = "demographics.xls"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
BATCH_SIZE = 32
EPOCHS_PD = 80
EPOCHS_SEV = 80
LR = 7e-4
WEIGHT_DECAY = 1e-2
WINDOW_LEN = 512
HOP_LEN = 192
MAX_LEN_CAP = 8192
HIDDEN_DIM = 192
NUM_LAYERS = 3
DROPOUT = 0.35
PATIENCE = 12
CLIP_MAX_NORM = 1.0

# Augment (stable for both tasks)
TIME_MASK_PROB = 0.2
TIME_MASK_LEN = 48
NOISE_STD = 0.03

# TTA
TTA_N = 4           # final evaluation TTA passes
VAL_TTA_N = 1       # per-epoch validation TTA passes (lower for speed)

# Focal (PD)
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# EMA
EMA_DECAY = 0.999

# SAM
SAM_RHO = 0.05
SAM_ADAPTIVE = True

# Runtime/loader defaults (tunable via CLI)
NUM_WORKERS = 0
USE_FREQ = True
USE_SAM = True
USE_AMP = False
COMPILE_MODEL = False
VAL_INTERVAL = 5


# ==================== UTIL & DATA ====================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_gait_file_for_id(dataset_dir: str, subj_id) -> Optional[str]:
    candidates: List[str] = []
    candidates += glob(os.path.join(dataset_dir, f"{subj_id}.txt"))
    candidates += glob(os.path.join(dataset_dir, f"{subj_id}_*.txt"))
    return candidates[0] if candidates else None


def load_subject_series(demo_path: str, dataset_dir: str) -> Tuple[List, List[np.ndarray], List[int], List[int]]:
    try:
        df = pd.read_excel(demo_path, engine="xlrd")
    except Exception:
        df = pd.read_excel(demo_path)

    pd_labels = (df["Group"].astype(str).str.upper() == "PD").astype(int)

    def map_severity(x):
        try:
            x = float(x)
            if x <= 2:
                return 0
            elif x == 3:
                return 1
            else:
                return 2
        except Exception:
            return 0

    sev_labels = df["HoehnYahr"].apply(map_severity)

    ids, series, pd_list, sev_list = [], [], [], []
    for idx, subj_id in enumerate(df["ID"]):
        fpath = find_gait_file_for_id(dataset_dir, subj_id)
        if fpath is None:
            continue
        try:
            arr = np.loadtxt(fpath)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[0] > MAX_LEN_CAP:
                arr = arr[:MAX_LEN_CAP]
            ids.append(subj_id)
            series.append(arr.astype(np.float32))
            pd_list.append(int(pd_labels.iloc[idx]))
            sev_list.append(int(sev_labels.iloc[idx]))
        except Exception as e:
            print(f"Warning: could not load {fpath} for {subj_id}: {e}")
    return ids, series, pd_list, sev_list


def make_windows(seq: np.ndarray, wlen: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    T, D = seq.shape
    if T <= wlen:
        pad = np.zeros((wlen - T, D), dtype=seq.dtype)
        return np.expand_dims(np.vstack([seq, pad]), 0), np.array([T], dtype=np.int64)
    out, lens = [], []
    for start in range(0, max(1, T - wlen + 1), hop):
        end = start + wlen
        if end <= T:
            out.append(seq[start:end]); lens.append(wlen)
        else:
            chunk = seq[start:T]
            pad = np.zeros((wlen - chunk.shape[0], D), dtype=seq.dtype)
            out.append(np.vstack([chunk, pad])); lens.append(chunk.shape[0])
            break
    return np.stack(out, axis=0), np.array(lens, dtype=np.int64)


class WindowDataset(Dataset):
    def __init__(self, windows, lengths, pd_labels, sev_labels, subj_ids):
        self.windows = windows
        self.lengths = lengths
        self.pd_labels = pd_labels
        self.sev_labels = sev_labels
        self.subj_ids = subj_ids

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.windows[idx], dtype=torch.float32),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.pd_labels[idx], dtype=torch.long),
            torch.tensor(self.sev_labels[idx], dtype=torch.long),
            torch.tensor(self.subj_ids[idx], dtype=torch.long),
        )


# ==================== LOSS ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        targets = targets.long()
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


# ==================== MODEL ====================
class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class EnhancedTemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        c1 = out_ch // 3
        c2 = out_ch // 3
        c3 = out_ch - c1 - c2
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1), nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, c2, kernel_size=7, padding=3),
            nn.BatchNorm1d(c2), nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, c3, kernel_size=11, padding=5),
            nn.BatchNorm1d(c3), nn.GELU(),
        )
        self.se = SqueezeExcite1D(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.se(out)
        out = self.dropout(out)
        return out + self.res(x)


class FreqBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.se = SqueezeExcite1D(out_ch)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        out = self.se(out)
        return out + self.res(x)


class DualStreamEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, use_freq: bool = True):
        super().__init__()
        self.use_freq = use_freq
        # Temporal path (time-domain)
        self.tcnn = nn.Sequential(
            EnhancedTemporalBlock(input_dim, 64, dropout),
            EnhancedTemporalBlock(64, 128, dropout),
            EnhancedTemporalBlock(128, 192, dropout),
            EnhancedTemporalBlock(192, 256, dropout),
        )
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.t_ln = nn.LayerNorm(hidden_dim * 2)
        self.t_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=8,
                                            dropout=dropout, batch_first=True)
        self.t_ln2 = nn.LayerNorm(hidden_dim * 2)

        # Frequency path (rFFT on time dimension)
        if self.use_freq:
            # Input to convs: [B, D, F]
            self.fcnn = nn.Sequential(
                FreqBlock(input_dim, 96, dropout),
                FreqBlock(96, 160, dropout),
                FreqBlock(160, 224, dropout),
            )
            self.f_attn = nn.MultiheadAttention(224, num_heads=8, dropout=dropout, batch_first=True)
            self.f_ln = nn.LayerNorm(224)

        # Fusion dims
        self.out_dim_temporal = hidden_dim * 2
        self.out_dim_freq = 224 if self.use_freq else 0

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # x: [B, T, D]
        B, T, D = x.shape
        device = x.device

        # Temporal path
        t = self.tcnn(x.transpose(1, 2)).transpose(1, 2)  # [B, T, 256]
        t_out, _ = self.lstm(t)                            # [B, T, 2H]
        t_out = self.t_ln(t_out)
        # Mask: True for padding
        mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) >= lengths.unsqueeze(1)
        t_attn, _ = self.t_attn(t_out, t_out, t_out, key_padding_mask=mask)
        t_attn = self.t_ln2(t_attn + t_out)
        valid = (~mask).float()  # [B,T]
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        t_pooled = (t_attn * valid.unsqueeze(-1)).sum(dim=1) / denom  # [B, 2H]

        # Frequency path (optional)
        f_pooled = None
        if self.use_freq:
            # rFFT magnitude along time axis
            fmag = torch.fft.rfft(x, dim=1).abs()  # [B, F, D]
            f_in = fmag.transpose(1, 2)            # [B, D, F]
            f = self.fcnn(f_in)                     # [B, 224, F]
            f = f.transpose(1, 2)                   # [B, F, 224]
            f_attn, _ = self.f_attn(f, f, f)      # no mask; F fixed per window
            f_attn = self.f_ln(f_attn)
            f_pooled = f_attn.mean(dim=1)          # [B, 224]

        # Return stream embeddings and concatenated fusion
        fused = torch.cat([t_pooled, f_pooled], dim=1) if self.use_freq else t_pooled
        return fused, t_pooled, f_pooled


class PDModel(nn.Module):
    def __init__(self, input_dim: int, use_freq: bool = True):
        super().__init__()
        self.enc = DualStreamEncoder(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT, use_freq=use_freq)
        in_dim = self.enc.out_dim_temporal + self.enc.out_dim_freq
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 2),
        )

    def forward(self, x, lengths):
        fused, _, _ = self.enc(x, lengths)
        return self.head(fused)


class SevModel(nn.Module):
    def __init__(self, input_dim: int, use_freq: bool = True):
        super().__init__()
        self.enc = DualStreamEncoder(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT, use_freq=use_freq)
        in_dim = self.enc.out_dim_temporal + self.enc.out_dim_freq
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 3),
        )

    def forward(self, x, lengths):
        fused, _, _ = self.enc(x, lengths)
        return self.head(fused)


# ==================== OPTIM (SAM + EMA) ====================
class SAM(optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al.)
    Minimal implementation that wraps a base optimizer (e.g., AdamW).
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, "Invalid rho"
        self.rho = rho
        self.adaptive = adaptive
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]['params'][0].device
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.adaptive:
                    e = torch.abs(p) * grad
                else:
                    e = grad
                norms.append(torch.norm(e, p=2))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.adaptive:
                    e_w = torch.pow(p, 2) * p.grad * scale
                else:
                    e_w = p.grad * scale
                self.state[p]['e_w'] = e_w
                p.add_(e_w)  # ascend
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = self.state[p].pop('e_w', None)
                if e_w is not None:
                    p.add_(-e_w)  # descend back to neighborhood
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Use first_step() and second_step()")

    def zero_grad(self):
        self.base_optimizer.zero_grad()


# ==================== AUGMENT & TTA ====================
def apply_augmentations(x: torch.Tensor, L: torch.Tensor,
                        noise_std: float = NOISE_STD,
                        mask_prob: float = TIME_MASK_PROB,
                        mask_len: int = TIME_MASK_LEN) -> torch.Tensor:
    B, T, D = x.shape
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    if mask_prob > 0 and mask_len > 0:
        for i in range(B):
            if torch.rand(()) < mask_prob:
                Li = int(L[i].item())
                if Li > mask_len:
                    s = torch.randint(0, Li - mask_len, (1,)).item()
                    x[i, s:s + mask_len, :] = 0.0
    return x


def tta_logits(model: nn.Module, x: torch.Tensor, L: torch.Tensor, num_classes: int, n: int = TTA_N, use_amp: bool = False) -> torch.Tensor:
    # Average probabilities across TTA variants
    autocast = torch.cuda.amp.autocast if (use_amp and DEVICE.type == 'cuda') else nullcontext
    with torch.no_grad():
        probs_accum = torch.zeros((x.size(0), num_classes), device=x.device)
        for k in range(n):
            if k == 0:
                xa = x
            elif k % 2 == 1:
                xa = x + NOISE_STD * torch.randn_like(x)
            else:
                xa = apply_augmentations(x.clone(), L)
            with autocast():
                logits = model(xa, L)
                probs = F.softmax(logits, dim=1)
            probs_accum += probs
        probs_mean = probs_accum / float(n)
        return probs_mean


# ==================== DATA BUILD ====================
def build_datasets() -> Tuple[WindowDataset, WindowDataset, int]:
    ids, series, pd_list, sev_list = load_subject_series(DEMOGRAPHICS_FILE, DATASET_DIR)

    subj_idx = np.arange(len(ids))
    pd_arr = np.array(pd_list, dtype=int)
    sev_arr = np.array(sev_list, dtype=int)
    joint = pd_arr * 3 + sev_arr
    train_idx, val_idx = train_test_split(subj_idx, test_size=0.20, random_state=SEED, stratify=joint)

    D = series[0].shape[1]
    train_concat = np.concatenate([series[i] for i in train_idx], axis=0)
    scaler = StandardScaler().fit(train_concat)

    def gen(split_idx):
        windows_list, lengths_list, pdw, sew, sidw = [], [], [], [], []
        for i in split_idx:
            seq = scaler.transform(series[i])
            w, l = make_windows(seq, WINDOW_LEN, HOP_LEN)
            n = w.shape[0]
            windows_list.append(w); lengths_list.append(l)
            pdw.append(np.full(n, pd_list[i], dtype=np.int64))
            sew.append(np.full(n, sev_list[i], dtype=np.int64))
            sidw.append(np.full(n, i, dtype=np.int64))
        return (
            np.concatenate(windows_list, axis=0),
            np.concatenate(lengths_list, axis=0),
            np.concatenate(pdw, axis=0),
            np.concatenate(sew, axis=0),
            np.concatenate(sidw, axis=0),
        )

    Xtr, Ltr, PDtr, SEVtr, SIDtr = gen(train_idx)
    Xva, Lva, PDva, SEVva, SIDva = gen(val_idx)

    return WindowDataset(Xtr, Ltr, PDtr, SEVtr, SIDtr), WindowDataset(Xva, Lva, PDva, SEVva, SIDva), D


# ==================== TRAIN LOOPS ====================
def train_pd_model(model: PDModel, train_ds: WindowDataset, val_ds: WindowDataset,
                   use_sam: bool = USE_SAM, num_workers: int = NUM_WORKERS,
                   val_interval: int = VAL_INTERVAL, val_tta_n: int = VAL_TTA_N,
                   use_amp: bool = USE_AMP, epochs: int = EPOCHS_PD,
                   max_steps_per_epoch: Optional[int] = None):
    # PD-balanced sampling
    pd_counts = Counter(train_ds.pd_labels.tolist())
    pd_w = {c: len(train_ds) / (2 * max(pd_counts.get(c, 1), 1)) for c in [0, 1]}
    sample_weights = np.array([pd_w[int(y)] for y in train_ds.pd_labels], dtype=np.float32)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True),
        num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=(num_workers > 0))

    pd_weights = torch.tensor([pd_w[0], pd_w[1]], dtype=torch.float32, device=DEVICE)
    criterion = FocalLoss(weight=pd_weights)

    base_opt = lambda params, **kw: optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    if use_sam:
        optimizer = SAM(model.parameters(), base_optimizer=base_opt, rho=SAM_RHO, adaptive=SAM_ADAPTIVE, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=12, T_mult=2)
    else:
        optimizer = base_opt(model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2)

    ema_model = AveragedModel(model, avg_fn=lambda avg, p, n: EMA_DECAY * avg + (1.0 - EMA_DECAY) * p)

    best_pd, wait = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        autocast = torch.cuda.amp.autocast if (use_amp and DEVICE.type == 'cuda') else torch.cpu.amp.autocast
        step = 0
        for x, L, ypd, _, _ in train_loader:
            x, L, ypd = x.to(DEVICE), L.to(DEVICE), ypd.to(DEVICE)
            xa = apply_augmentations(x, L)

            if use_sam:
                # SAM two-step
                with autocast():
                    logits = model(xa, L)
                    loss = criterion(logits, ypd)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.first_step(zero_grad=True)

                with autocast():
                    logits2 = model(xa, L)
                    loss2 = criterion(logits2, ypd)
                loss2.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(xa, L)
                    loss = criterion(logits, ypd)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.step()

            # EMA update after optimizer step
            ema_model.update_parameters(model)
            step += 1
            if (max_steps_per_epoch is not None) and (step >= max_steps_per_epoch):
                break
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch + 1)
        else:
            scheduler.step()

        # Validation with EMA weights + TTA
        ema_model.eval()
        subj_probs, subj_true = defaultdict(list), {}
        with torch.no_grad():
            for x, L, ypd, _, sid in val_loader:
                x, L = x.to(DEVICE), L.to(DEVICE)
                probs = tta_logits(ema_model, x, L, num_classes=2, n=val_tta_n, use_amp=use_amp)
                for i in range(x.size(0)):
                    s = int(sid[i].item())
                    subj_probs[s].append(probs[i].cpu())
                    subj_true[s] = int(ypd[i].item())
        preds, trues = [], []
        for s, plist in subj_probs.items():
            mean_prob = torch.stack(plist, dim=0).mean(dim=0)
            p = int(mean_prob.argmax().item())
            preds.append(p); trues.append(subj_true[s])
        pd_acc = accuracy_score(trues, preds)
        if epoch % val_interval == 0 or epoch == 1:
            print(f"[PD] Epoch {epoch:3d}/{epochs} | Val PD Acc: {pd_acc:.3f}")

        if pd_acc > best_pd:
            best_pd, wait = pd_acc, 0
            torch.save(ema_model.module.state_dict(), "best_pd_dualstream.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[PD] Early stopping at epoch {epoch}. Best PD Acc: {best_pd:.3f}")
                break

    # load best EMA
    model.load_state_dict(torch.load("best_pd_dualstream.pth", map_location=DEVICE))
    return best_pd


def train_sev_model(model: SevModel, train_ds: WindowDataset, val_ds: WindowDataset,
                    use_sam: bool = USE_SAM, num_workers: int = NUM_WORKERS,
                    val_interval: int = VAL_INTERVAL, val_tta_n: int = VAL_TTA_N,
                    use_amp: bool = USE_AMP, epochs: int = EPOCHS_SEV,
                    max_steps_per_epoch: Optional[int] = None):
    # Severity-focused sampling
    sev_counts = Counter(train_ds.sev_labels.tolist())
    sev_w = {c: len(train_ds) / (3 * max(sev_counts.get(c, 1), 1)) for c in [0, 1, 2]}
    sample_weights = np.array([sev_w[int(y)] for y in train_ds.sev_labels], dtype=np.float32)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True),
        num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=(num_workers > 0))

    sev_weights = torch.tensor([sev_w[0], sev_w[1], sev_w[2]], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=sev_weights, label_smoothing=0.05)

    base_opt = lambda params, **kw: optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    if use_sam:
        optimizer = SAM(model.parameters(), base_optimizer=base_opt, rho=SAM_RHO, adaptive=SAM_ADAPTIVE, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=12, T_mult=2)
    else:
        optimizer = base_opt(model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2)

    ema_model = AveragedModel(model, avg_fn=lambda avg, p, n: EMA_DECAY * avg + (1.0 - EMA_DECAY) * p)

    best_sev, wait = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        autocast = torch.cuda.amp.autocast if (use_amp and DEVICE.type == 'cuda') else torch.cpu.amp.autocast
        step = 0
        for x, L, _, ysev, _ in train_loader:
            x, L, ysev = x.to(DEVICE), L.to(DEVICE), ysev.to(DEVICE)
            xa = apply_augmentations(x, L)

            if use_sam:
                # SAM two-step
                with autocast():
                    logits = model(xa, L)
                    loss = criterion(logits, ysev)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.first_step(zero_grad=True)

                with autocast():
                    logits2 = model(xa, L)
                    loss2 = criterion(logits2, ysev)
                loss2.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(xa, L)
                    loss = criterion(logits, ysev)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.step()

            ema_model.update_parameters(model)
            step += 1
            if (max_steps_per_epoch is not None) and (step >= max_steps_per_epoch):
                break
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch + 1)
        else:
            scheduler.step()

        # Validation with EMA weights + TTA
        ema_model.eval()
        subj_probs, subj_true = defaultdict(list), {}
        with torch.no_grad():
            for x, L, _, ysev, sid in val_loader:
                x, L = x.to(DEVICE), L.to(DEVICE)
                probs = tta_logits(ema_model, x, L, num_classes=3, n=val_tta_n, use_amp=use_amp)
                for i in range(x.size(0)):
                    s = int(sid[i].item())
                    subj_probs[s].append(probs[i].cpu())
                    subj_true[s] = int(ysev[i].item())
        preds, trues = [], []
        for s, plist in subj_probs.items():
            mean_prob = torch.stack(plist, dim=0).mean(dim=0)
            p = int(mean_prob.argmax().item())
            preds.append(p); trues.append(subj_true[s])
        sev_acc = accuracy_score(trues, preds)
        if epoch % val_interval == 0 or epoch == 1:
            print(f"[SEV] Epoch {epoch:3d}/{epochs} | Val Sev Acc: {sev_acc:.3f}")

        if sev_acc > best_sev:
            best_sev, wait = sev_acc, 0
            torch.save(ema_model.module.state_dict(), "best_sev_dualstream.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[SEV] Early stopping at epoch {epoch}. Best Sev Acc: {best_sev:.3f}")
                break

    model.load_state_dict(torch.load("best_sev_dualstream.pth", map_location=DEVICE))
    return best_sev


# ==================== EVALUATION ====================
def evaluate(pd_model: PDModel, sev_model: SevModel, val_ds: WindowDataset,
             num_workers: int = NUM_WORKERS, tta_n: int = TTA_N, use_amp: bool = USE_AMP):
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=(num_workers > 0))

    # PD eval
    pd_model.eval()
    subj_pd_probs, subj_pd_true = defaultdict(list), {}
    with torch.no_grad():
        for x, L, ypd, _, sid in val_loader:
            x, L = x.to(DEVICE), L.to(DEVICE)
            probs = tta_logits(pd_model, x, L, num_classes=2, n=tta_n, use_amp=use_amp)
            for i in range(x.size(0)):
                s = int(sid[i].item())
                subj_pd_probs[s].append(probs[i].cpu())
                subj_pd_true[s] = int(ypd[i].item())
    pd_preds, pd_trues = [], []
    for s, plist in subj_pd_probs.items():
        mean_prob = torch.stack(plist, dim=0).mean(dim=0)
        p = int(mean_prob.argmax().item())
        pd_preds.append(p); pd_trues.append(subj_pd_true[s])
    pd_acc = accuracy_score(pd_trues, pd_preds)

    # Severity eval
    sev_model.eval()
    subj_sev_probs, subj_sev_true = defaultdict(list), {}
    with torch.no_grad():
        for x, L, _, ysev, sid in val_loader:
            x, L = x.to(DEVICE), L.to(DEVICE)
            probs = tta_logits(sev_model, x, L, num_classes=3, n=tta_n, use_amp=use_amp)
            for i in range(x.size(0)):
                s = int(sid[i].item())
                subj_sev_probs[s].append(probs[i].cpu())
                subj_sev_true[s] = int(ysev[i].item())
    sev_preds, sev_trues = [], []
    for s, plist in subj_sev_probs.items():
        mean_prob = torch.stack(plist, dim=0).mean(dim=0)
        p = int(mean_prob.argmax().item())
        sev_preds.append(p); sev_trues.append(subj_sev_true[s])
    sev_acc = accuracy_score(sev_trues, sev_preds)

    print("\n" + "=" * 70)
    print("=== Subject-level PD (Dual-Stream PD-model, EMA + TTA) ===")
    print(classification_report(pd_trues, pd_preds, target_names=["Control", "PD"]))
    print(f"PD Subject Accuracy: {pd_acc:.4f}")

    print("\n=== Subject-level Severity (Dual-Stream SEV-model, EMA + TTA) ===")
    print(classification_report(sev_trues, sev_preds, target_names=["Mild", "Moderate", "Severe"]))
    print(f"Severity Subject Accuracy: {sev_acc:.4f}")
    print("=" * 70)

    return pd_acc, sev_acc


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    # speed/runtime flags
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM (use plain AdamW)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training/eval")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--no-freq", action="store_true", help="Disable frequency stream for speed")
    parser.add_argument("--compile", action="store_true", help="torch.compile models (PyTorch 2.x)")
    parser.add_argument("--val-interval", type=int, default=VAL_INTERVAL, help="Validate every N epochs")
    parser.add_argument("--val-tta", type=int, default=VAL_TTA_N, help="Validation TTA passes per epoch")
    parser.add_argument("--tta", type=int, default=TTA_N, help="Final eval TTA passes")
    parser.add_argument("--epochs-pd", type=int, default=EPOCHS_PD, help="PD training epochs")
    parser.add_argument("--epochs-sev", type=int, default=EPOCHS_SEV, help="Severity training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per epoch (debug speed)")
    args = parser.parse_args()

    set_seed(SEED)
    print("=" * 70)
    print("Hybrid Dual-Stream Gait (PD focal+SAM+EMA+TTA, Sev CE+LS+SAM+EMA+TTA)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    train_ds, val_ds, feat_dim = build_datasets()
    print(f"Training windows: {len(train_ds)}, Validation windows: {len(val_ds)}")

    use_freq = not args.no_freq
    use_sam = not args.no_sam
    use_amp = bool(args.amp)
    num_workers = int(args.workers)
    val_interval = int(args.val_interval)
    val_tta = int(args.val_tta)
    final_tta = int(args.tta)
    epochs_pd = int(args.epochs_pd)
    epochs_sev = int(args.epochs_sev)
    max_steps = args.max_steps

    # Train PD model
    pd_model = PDModel(input_dim=feat_dim, use_freq=use_freq).to(DEVICE)
    if args.compile:
        try:
            pd_model = torch.compile(pd_model)
        except Exception as e:
            print(f"Warning: torch.compile failed for PD model: {e}")
    best_pd = train_pd_model(pd_model, train_ds, val_ds,
                             use_sam=use_sam, num_workers=num_workers,
                             val_interval=val_interval, val_tta_n=val_tta, use_amp=use_amp,
                             epochs=epochs_pd, max_steps_per_epoch=max_steps)

    # Train Severity model
    sev_model = SevModel(input_dim=feat_dim, use_freq=use_freq).to(DEVICE)
    if args.compile:
        try:
            sev_model = torch.compile(sev_model)
        except Exception as e:
            print(f"Warning: torch.compile failed for Sev model: {e}")
    best_sev = train_sev_model(sev_model, train_ds, val_ds,
                               use_sam=use_sam, num_workers=num_workers,
                               val_interval=val_interval, val_tta_n=val_tta, use_amp=use_amp,
                               epochs=epochs_sev, max_steps_per_epoch=max_steps)

    # Load best EMA checkpoints and evaluate (full TTA)
    pd_model.load_state_dict(torch.load("best_pd_dualstream.pth", map_location=DEVICE))
    sev_model.load_state_dict(torch.load("best_sev_dualstream.pth", map_location=DEVICE))
    evaluate(pd_model, sev_model, val_ds,
             num_workers=num_workers, tta_n=final_tta, use_amp=use_amp)


if __name__ == "__main__":
    main()
