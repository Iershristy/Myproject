#!/usr/bin/env python3
"""
PhysioNet gaitpdb - Dual-Stream, Two-Head, K-Fold Training
- Dual-stream encoder (temporal + rFFT frequency) with attention and SE
- Two specialized heads/models: PD (focal), Severity (CE + label smoothing)
- EMA + optional SAM + AMP + TTA
- K-fold training with subject-level stratification on (PD × HY)
- Aggregated (OOF) subject-level metrics across folds; per-fold checkpoints

Usage:
  python gait_pdb_kfold.py --data-dir /path/to/gaitpdb --kfolds 5 --train \
    --amp --workers 4 --val-interval 5 --val-tta 1 --tta 4

Fast debug:
  python gait_pdb_kfold.py --data-dir . --kfolds 3 --train \
    --epochs-pd 1 --epochs-sev 1 --max-steps 2 --no-freq --no-sam --amp --workers 2 --val-interval 1 --val-tta 1 --tta 1
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel
from contextlib import nullcontext

# Reuse core hyperparams from the hybrid script (tuned for accuracy)
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

# Augment
TIME_MASK_PROB = 0.2
TIME_MASK_LEN = 48
NOISE_STD = 0.03

# TTA
TTA_N = 4
VAL_TTA_N = 1

# Focal (PD)
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# EMA
EMA_DECAY = 0.999

# SAM
SAM_RHO = 0.05
SAM_ADAPTIVE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_demographics_file(data_dir: str) -> Optional[str]:
    # Try common patterns in gaitpdb
    candidates = []
    candidates += glob(os.path.join(data_dir, "demographics.xls"))
    candidates += glob(os.path.join(data_dir, "demographics.xlsx"))
    candidates += glob(os.path.join(data_dir, "**/demographics.xls"), recursive=True)
    candidates += glob(os.path.join(data_dir, "**/demographics.xlsx"), recursive=True)
    return candidates[0] if candidates else None


def read_demographics(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    else:
        try:
            df = pd.read_excel(path, engine="xlrd")
        except Exception:
            df = pd.read_excel(path)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    # Expected
    id_col = None
    for key in ["id", "subject", "subject_id", "subjid", "participant", "patientid"]:
        if key in cols:
            id_col = cols[key]
            break
    if id_col is None:
        raise ValueError("Could not find subject ID column in demographics.")

    if "group" in cols:
        group_col = cols["group"]
        pd_series = (df[group_col].astype(str).str.upper() == "PD").astype(int)
    else:
        # Fallback: try 'diagnosis' style fields
        diag_col = None
        for key in ["diagnosis", "label", "class"]:
            if key in cols:
                diag_col = cols[key]
                break
        if diag_col is None:
            raise ValueError("Could not find PD/Control group column.")
        pd_series = (df[diag_col].astype(str).str.upper().str.contains("PD")).astype(int)

    # HY severity (mild/moderate/severe mapping)
    hy_col = None
    for key in ["hoehnyahr", "hy", "hoehn_yahr", "hoehn-yahr", "stage", "severity"]:
        if key in cols:
            hy_col = cols[key]
            break
    if hy_col is None:
        # default to zeros if missing
        hy_series = pd.Series([0] * len(df))
    else:
        def map_hy(x):
            try:
                v = float(x)
                if v <= 2: return 0
                elif v == 3: return 1
                else: return 2
            except Exception:
                return 0
        hy_series = df[hy_col].apply(map_hy)

    out = pd.DataFrame({
        "ID": df[id_col],
        "PD": pd_series.astype(int),
        "SEV": hy_series.astype(int),
    })
    return out


def find_gait_file_for_id(data_dir: str, subj_id) -> Optional[str]:
    candidates: List[str] = []
    candidates += glob(os.path.join(data_dir, f"{subj_id}.txt"))
    candidates += glob(os.path.join(data_dir, f"{subj_id}_*.txt"))
    candidates += glob(os.path.join(data_dir, "**", f"{subj_id}.txt"), recursive=True)
    candidates += glob(os.path.join(data_dir, "**", f"{subj_id}_*.txt"), recursive=True)
    return candidates[0] if candidates else None


def load_subject_series(demo_df: pd.DataFrame, data_dir: str) -> Tuple[List, List[np.ndarray], List[int], List[int]]:
    ids, series, pd_list, sev_list = [], [], [], []
    for idx, subj_id in enumerate(demo_df["ID"].tolist()):
        fpath = find_gait_file_for_id(data_dir, subj_id)
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
            pd_list.append(int(demo_df["PD"].iloc[idx]))
            sev_list.append(int(demo_df["SEV"].iloc[idx]))
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
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.windows[idx], dtype=torch.float32),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.pd_labels[idx], dtype=torch.long),
            torch.tensor(self.sev_labels[idx], dtype=torch.long),
            torch.tensor(self.subj_ids[idx], dtype=torch.long),
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=None):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.weight = weight
    def forward(self, inputs, targets):
        targets = targets.long()
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1), nn.Sigmoid(),
        )
    def forward(self, x): return x * self.se(x)


class EnhancedTemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        c1 = out_ch // 3; c2 = out_ch // 3; c3 = out_ch - c1 - c2
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=3, padding=1), nn.BatchNorm1d(c1), nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, c2, kernel_size=7, padding=3), nn.BatchNorm1d(c2), nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, c3, kernel_size=11, padding=5), nn.BatchNorm1d(c3), nn.GELU(),
        )
        self.se = SqueezeExcite1D(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        b1 = self.branch1(x); b2 = self.branch2(x); b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.se(out)
        out = self.dropout(out)
        return out + self.res(x)


class FreqBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2), nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2), nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.se = SqueezeExcite1D(out_ch)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.net(x); out = self.se(out); return out + self.res(x)


class DualStreamEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, use_freq: bool = True):
        super().__init__()
        self.use_freq = use_freq
        self.tcnn = nn.Sequential(
            EnhancedTemporalBlock(input_dim, 64, dropout),
            EnhancedTemporalBlock(64, 128, dropout),
            EnhancedTemporalBlock(128, 192, dropout),
            EnhancedTemporalBlock(192, 256, dropout),
        )
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.t_ln = nn.LayerNorm(hidden_dim * 2)
        self.t_attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True)
        self.t_ln2 = nn.LayerNorm(hidden_dim * 2)
        if self.use_freq:
            self.fcnn = nn.Sequential(
                FreqBlock(input_dim, 96, dropout), FreqBlock(96, 160, dropout), FreqBlock(160, 224, dropout),
            )
            self.f_attn = nn.MultiheadAttention(224, num_heads=8, dropout=dropout, batch_first=True)
            self.f_ln = nn.LayerNorm(224)
        self.out_dim_temporal = hidden_dim * 2
        self.out_dim_freq = 224 if self.use_freq else 0
    def forward(self, x, lengths):
        B, T, D = x.shape
        device = x.device
        t = self.tcnn(x.transpose(1, 2)).transpose(1, 2)
        t_out, _ = self.lstm(t)
        t_out = self.t_ln(t_out)
        mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) >= lengths.unsqueeze(1)
        t_attn, _ = self.t_attn(t_out, t_out, t_out, key_padding_mask=mask)
        t_attn = self.t_ln2(t_attn + t_out)
        valid = (~mask).float(); denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        t_pooled = (t_attn * valid.unsqueeze(-1)).sum(dim=1) / denom
        f_pooled = None
        if self.use_freq:
            fmag = torch.fft.rfft(x, dim=1).abs()
            f_in = fmag.transpose(1, 2)
            f = self.fcnn(f_in).transpose(1, 2)
            f_attn, _ = self.f_attn(f, f, f)
            f_attn = self.f_ln(f_attn)
            f_pooled = f_attn.mean(dim=1)
        fused = torch.cat([t_pooled, f_pooled], dim=1) if self.use_freq else t_pooled
        return fused


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
        return self.head(self.enc(x, lengths))


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
        return self.head(self.enc(x, lengths))


class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0
        self.rho = rho; self.adaptive = adaptive
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]['params'][0].device
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                e = torch.abs(p) * grad if self.adaptive else grad
                norms.append(torch.norm(e, p=2))
        if not norms:
            return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(norms), p=2)
    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm(); scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) * p.grad * scale) if self.adaptive else (p.grad * scale)
                self.state[p]['e_w'] = e_w
                p.add_(e_w)
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                e_w = self.state[p].pop('e_w', None)
                if e_w is not None:
                    p.add_(-e_w)
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    def step(self, closure=None):
        raise NotImplementedError()
    def zero_grad(self): self.base_optimizer.zero_grad()


def apply_augmentations(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    if NOISE_STD > 0:
        x = x + NOISE_STD * torch.randn_like(x)
    if TIME_MASK_PROB > 0 and TIME_MASK_LEN > 0:
        for i in range(B):
            if torch.rand(()) < TIME_MASK_PROB:
                Li = int(L[i].item())
                if Li > TIME_MASK_LEN:
                    s = torch.randint(0, Li - TIME_MASK_LEN, (1,)).item()
                    x[i, s:s+TIME_MASK_LEN, :] = 0.0
    return x


def tta_logits(model: nn.Module, x: torch.Tensor, L: torch.Tensor, num_classes: int, n: int, use_amp: bool) -> torch.Tensor:
    with torch.no_grad():
        autocast = torch.cuda.amp.autocast if (use_amp and DEVICE.type == 'cuda') else nullcontext
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
        return probs_accum / float(n)


def build_windows_for_split(series: List[np.ndarray], idxs: List[int], labels_pd: List[int], labels_sev: List[int]) -> Tuple:
    D = series[0].shape[1]
    scaler = StandardScaler().fit(np.concatenate([series[i] for i in idxs], axis=0))
    windows_list, lengths_list, pdw, sew, sidw = [], [], [], [], []
    for i in idxs:
        seq = scaler.transform(series[i])
        w, l = make_windows(seq, WINDOW_LEN, HOP_LEN)
        n = w.shape[0]
        windows_list.append(w); lengths_list.append(l)
        pdw.append(np.full(n, labels_pd[i], dtype=np.int64))
        sew.append(np.full(n, labels_sev[i], dtype=np.int64))
        sidw.append(np.full(n, i, dtype=np.int64))
    return (
        np.concatenate(windows_list, axis=0),
        np.concatenate(lengths_list, axis=0),
        np.concatenate(pdw, axis=0),
        np.concatenate(sew, axis=0),
        np.concatenate(sidw, axis=0),
        D,
    )


def train_one_task(model: nn.Module, train_ds: WindowDataset, val_ds: WindowDataset,
                   num_classes: int, is_pd: bool,
                   use_sam: bool, use_amp: bool, num_workers: int,
                   val_interval: int, val_tta_n: int,
                   epochs: int, max_steps_per_epoch: Optional[int],
                   ckpt_path: str) -> float:
    # Sampling weights
    if is_pd:
        counts = Counter(train_ds.pd_labels.tolist()); C = 2
        weights_map = {c: len(train_ds) / (C * max(counts.get(c, 1), 1)) for c in range(C)}
        sample_weights = np.array([weights_map[int(y)] for y in train_ds.pd_labels], dtype=np.float32)
        criterion = FocalLoss(weight=torch.tensor([weights_map[0], weights_map[1]], dtype=torch.float32, device=DEVICE))
    else:
        counts = Counter(train_ds.sev_labels.tolist()); C = 3
        weights_map = {c: len(train_ds) / (C * max(counts.get(c, 1), 1)) for c in range(C)}
        sample_weights = np.array([weights_map[int(y)] for y in train_ds.sev_labels], dtype=np.float32)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([weights_map[0], weights_map[1], weights_map[2]], dtype=torch.float32, device=DEVICE), label_smoothing=0.05)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True),
        num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'), persistent_workers=(num_workers > 0))

    base_opt = lambda params, **kw: optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    if use_sam:
        optimizer = SAM(model.parameters(), base_optimizer=base_opt, rho=SAM_RHO, adaptive=SAM_ADAPTIVE, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.base_optimizer, T_0=12, T_mult=2)
    else:
        optimizer = base_opt(model.parameters())
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=12, T_mult=2)

    ema_model = AveragedModel(model, avg_fn=lambda avg, p, n: EMA_DECAY * avg + (1.0 - EMA_DECAY) * p)

    best, wait = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        autocast = torch.cuda.amp.autocast if (use_amp and DEVICE.type == 'cuda') else nullcontext
        step = 0
        for x, L, ypd, ysev, _ in train_loader:
            x, L = x.to(DEVICE), L.to(DEVICE)
            y = (ypd if is_pd else ysev).to(DEVICE)
            xa = apply_augmentations(x, L)
            if use_sam:
                with autocast():
                    logits = model(xa, L)
                    loss = criterion(logits, y)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.first_step(zero_grad=True)
                with autocast():
                    logits2 = model(xa, L)
                    loss2 = criterion(logits2, y)
                loss2.backward(); nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(xa, L)
                    loss = criterion(logits, y)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
                optimizer.step()
            ema_model.update_parameters(model)
            step += 1
            if (max_steps_per_epoch is not None) and (step >= max_steps_per_epoch):
                break
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step(epoch + 1)
        else:
            scheduler.step()

        # val subject-level (EMA + TTA)
        ema_model.eval()
        subj_probs, subj_true = defaultdict(list), {}
        with torch.no_grad():
            for x, L, ypd, ysev, sid in val_loader:
                x, L = x.to(DEVICE), L.to(DEVICE)
                y = (ypd if is_pd else ysev)
                probs = tta_logits(ema_model, x, L, num_classes=num_classes, n=VAL_TTA_N, use_amp=use_amp)
                for i in range(x.size(0)):
                    s = int(sid[i].item())
                    subj_probs[s].append(probs[i].cpu())
                    subj_true[s] = int(y[i].item())
        preds, trues = [], []
        for s, plist in subj_probs.items():
            mean_prob = torch.stack(plist, dim=0).mean(dim=0)
            p = int(mean_prob.argmax().item())
            preds.append(p); trues.append(subj_true[s])
        acc = accuracy_score(trues, preds)
        if epoch % val_interval == 0 or epoch == 1:
            print(f"[{'PD' if is_pd else 'SEV'}] Epoch {epoch:3d}/{epochs} | Val Acc: {acc:.3f}")
        if acc > best:
            best, wait = acc, 0
            torch.save(ema_model.module.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[{'PD' if is_pd else 'SEV'}] Early stopping at epoch {epoch}. Best Acc: {best:.3f}")
                break
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return best


def evaluate_fold(pd_model: nn.Module, sev_model: nn.Module, val_ds: WindowDataset, use_amp: bool, tta_n: int) -> Tuple[float, float, Dict, Dict]:
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    # PD
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
    # SEV
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
    print("\n" + "="*70)
    print("=== Fold Subject-level PD ===")
    print(classification_report(pd_trues, pd_preds, target_names=["Control", "PD"]))
    print(f"PD Subject Accuracy: {pd_acc:.4f}")
    print("\n=== Fold Subject-level Severity ===")
    print(classification_report(sev_trues, sev_preds, target_names=["Mild", "Moderate", "Severe"]))
    print(f"Severity Subject Accuracy: {sev_acc:.4f}")
    print("="*70)
    return pd_acc, sev_acc, {"trues": pd_trues, "preds": pd_preds}, {"trues": sev_trues, "preds": sev_preds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True, help="Path to PhysioNet gaitpdb root")
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--train", action="store_true")
    # speed/accuracy flags
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-freq", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--val-tta", type=int, default=VAL_TTA_N)
    p.add_argument("--tta", type=int, default=TTA_N)
    p.add_argument("--epochs-pd", type=int, default=EPOCHS_PD)
    p.add_argument("--epochs-sev", type=int, default=EPOCHS_SEV)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    set_seed(SEED)
    print("="*70)
    print("gaitpdb K-Fold: Dual-Stream + Two-Head (EMA+TTA [+SAM])")
    print("="*70)
    print(f"Device: {DEVICE}")

    demo_path = find_demographics_file(args.data_dir)
    if demo_path is None:
        raise FileNotFoundError("Could not find demographics .xls/.xlsx in data-dir")
    demo_df = read_demographics(demo_path)

    ids, series, pd_list, sev_list = load_subject_series(demo_df, args.data_dir)
    if len(ids) == 0:
        raise RuntimeError("No subjects matched ID->txt files. Check naming.")

    subj_idx = np.arange(len(ids))
    pd_arr = np.array(pd_list, dtype=int)
    sev_arr = np.array(sev_list, dtype=int)
    joint = pd_arr * 3 + sev_arr

    kf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=SEED)

    all_pd_trues, all_pd_preds = [], []
    all_sev_trues, all_sev_preds = [], []

    use_freq = not args.no_freq
    use_sam = not args.no_sam
    use_amp = bool(args.amp)

    for fold, (train_idx, val_idx) in enumerate(kf.split(subj_idx, joint), 1):
        print(f"\n##### Fold {fold}/{args.kfolds} ####")
        Xtr, Ltr, PDtr, SEVtr, SIDtr, D = build_windows_for_split(series, train_idx.tolist(), pd_list, sev_list)
        Xva, Lva, PDva, SEVva, SIDva, _ = build_windows_for_split(series, val_idx.tolist(), pd_list, sev_list)
        train_ds = WindowDataset(Xtr, Ltr, PDtr, SEVtr, SIDtr)
        val_ds = WindowDataset(Xva, Lva, PDva, SEVva, SIDva)

        pd_model = PDModel(input_dim=D, use_freq=use_freq).to(DEVICE)
        sev_model = SevModel(input_dim=D, use_freq=use_freq).to(DEVICE)
        if args.compile:
            try:
                pd_model = torch.compile(pd_model)
                sev_model = torch.compile(sev_model)
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

        if args.train:
            best_pd = train_one_task(
                pd_model, train_ds, val_ds,
                num_classes=2, is_pd=True,
                use_sam=use_sam, use_amp=use_amp, num_workers=args.workers,
                val_interval=args.val_interval, val_tta_n=args.val_tta,
                epochs=args.epochs_pd, max_steps_per_epoch=args.max_steps,
                ckpt_path=f"best_pd_fold{fold}.pth",
            )
            best_sev = train_one_task(
                sev_model, train_ds, val_ds,
                num_classes=3, is_pd=False,
                use_sam=use_sam, use_amp=use_amp, num_workers=args.workers,
                val_interval=args.val_interval, val_tta_n=args.val_tta,
                epochs=args.epochs_sev, max_steps_per_epoch=args.max_steps,
                ckpt_path=f"best_sev_fold{fold}.pth",
            )
        else:
            pd_model.load_state_dict(torch.load(f"best_pd_fold{fold}.pth", map_location=DEVICE))
            sev_model.load_state_dict(torch.load(f"best_sev_fold{fold}.pth", map_location=DEVICE))

        pd_acc, sev_acc, pd_dict, sev_dict = evaluate_fold(pd_model, sev_model, val_ds, use_amp=use_amp, tta_n=args.tta)
        all_pd_trues.extend(pd_dict["trues"])
        all_pd_preds.extend(pd_dict["preds"])
        all_sev_trues.extend(sev_dict["trues"])
        all_sev_preds.extend(sev_dict["preds"])

    # Aggregate OOF metrics
    print("\n" + "#"*70)
    print("=== OOF Subject-level PD (across folds) ===")
    print(classification_report(all_pd_trues, all_pd_preds, target_names=["Control", "PD"]))
    print(f"PD OOF Accuracy: {accuracy_score(all_pd_trues, all_pd_preds):.4f}")
    print("\n=== OOF Subject-level Severity (across folds) ===")
    print(classification_report(all_sev_trues, all_sev_preds, target_names=["Mild", "Moderate", "Severe"]))
    print(f"Severity OOF Accuracy: {accuracy_score(all_sev_trues, all_sev_preds):.4f}")
    print("#"*70)


if __name__ == "__main__":
    main()
