#!/usr/bin/env python3
"""
PhysioNet gaitpdb - Windowed Training with Subject Aggregation

Key ideas to boost accuracy:
- Segment each subject's time series into overlapping windows => more training samples
- Temporal Conv1D frontend + BiLSTM + masked attention pooling
- Class weights for PD and severity; AdamW + OneCycleLR + grad clipping
- Subject-level validation: aggregate window predictions per subject (often higher accuracy)
- Optional ordinal severity loss (CORAL). Default: CrossEntropy.

Usage:
    python gait_windowed.py --train
"""

import argparse
import os
from glob import glob
import pickle
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ------------------------
# CONFIG
# ------------------------
DATASET_DIR = "."
DEMOGRAPHICS_FILE = "demographics.xls"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-2
WINDOW_LEN = 256
HOP_LEN = 64
MAX_LEN_CAP = 4096      # cap extremely long sequences (safety)
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
PATIENCE = 15
CLIP_MAX_NORM = 1.0
RANDOM_SEED = 42
USE_ORDINAL_SEVERITY = False  # set True to try CORAL ordinal loss
USE_AGE_COVARIATE = False     # concatenate normalized age to pooled features
USE_AGE_ADVERSARIAL = False   # adversarially remove age with GRL
LAMBDA_AGE = 0.1              # strength of adversarial age loss
TIME_MASK_PROB = 0.0          # training aug: probability to apply time mask per sample
TIME_MASK_LEN = 16            # training aug: time mask length
NOISE_STD = 0.0               # training aug: gaussian noise std added to inputs


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_gait_file_for_id(dataset_dir: str, subj_id) -> Optional[str]:
    candidates: List[str] = []
    candidates += glob(os.path.join(dataset_dir, f"{subj_id}.txt"))
    candidates += glob(os.path.join(dataset_dir, f"{subj_id}_*.txt"))
    return candidates[0] if candidates else None


def load_subject_series(demo_path: str, dataset_dir: str) -> Tuple[List, List[np.ndarray], List[int], List[int], List[float]]:
    # Load demographics
    try:
        df = pd.read_excel(demo_path, engine="xlrd")
    except Exception:
        df = pd.read_excel(demo_path)

    pd_labels_all = (df["Group"].astype(str).str.upper() == "PD").astype(int)

    def map_sev(x):
        try:
            x = float(x)
            if x <= 2: return 0
            elif x == 3: return 1
            else: return 2
        except Exception:
            return 0
    sev_labels_all = df["HoehnYahr"].apply(map_sev)

    ids: List = []
    series: List[np.ndarray] = []
    pd_list: List[int] = []
    sev_list: List[int] = []
    ages: List[float] = []

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
            pd_list.append(int(pd_labels_all.iloc[idx]))
            sev_list.append(int(sev_labels_all.iloc[idx]))
            # age may be in 'Age' or similar column; default to NaN -> impute later
            age_val = None
            for col in ["Age", "age", "AGE"]:
                if col in df.columns:
                    try:
                        age_val = float(df.loc[idx, col])
                    except Exception:
                        age_val = None
                    break
            ages.append(np.nan if age_val is None else age_val)
        except Exception as e:
            print(f"Warning: could not load {fpath} for {subj_id}: {e}")

    return ids, series, pd_list, sev_list, ages


def make_windows(seq: np.ndarray, wlen: int, hop: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (windows [Nw, wlen, D], lengths [Nw])"""
    T, D = seq.shape
    if T <= wlen:
        pad = np.zeros((wlen - T, D), dtype=seq.dtype)
        return np.expand_dims(np.vstack([seq, pad]), 0), np.array([T], dtype=np.int64)
    out = []
    lens = []
    for start in range(0, max(1, T - wlen + 1), hop):
        end = start + wlen
        if end <= T:
            out.append(seq[start:end])
            lens.append(wlen)
        else:
            chunk = seq[start:T]
            pad = np.zeros((wlen - chunk.shape[0], D), dtype=seq.dtype)
            out.append(np.vstack([chunk, pad]))
            lens.append(chunk.shape[0])
            break
    return np.stack(out, axis=0), np.array(lens, dtype=np.int64)


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, lengths: np.ndarray, pd_labels: np.ndarray, sev_labels: np.ndarray, subj_ids: np.ndarray, ages: Optional[np.ndarray] = None, age_mean: Optional[float] = None, age_std: Optional[float] = None):
        self.windows = windows
        self.lengths = lengths
        self.pd_labels = pd_labels
        self.sev_labels = sev_labels
        self.subj_ids = subj_ids
        self.ages = ages  # per-window age
        self.age_mean = age_mean
        self.age_std = age_std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        L = torch.tensor(self.lengths[idx], dtype=torch.long)
        y_pd = torch.tensor(self.pd_labels[idx], dtype=torch.long)
        y_sev = torch.tensor(self.sev_labels[idx], dtype=torch.long)
        sid = torch.tensor(self.subj_ids[idx], dtype=torch.long)
        if self.ages is not None:
            a = float(self.ages[idx])
            if self.age_mean is not None and self.age_std is not None and self.age_std > 0:
                a = (a - self.age_mean) / self.age_std
            age_t = torch.tensor([a], dtype=torch.float32)
        else:
            age_t = torch.tensor([0.0], dtype=torch.float32)
        return x, L, y_pd, y_sev, sid, age_t


class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 9, d: int = 1, p: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if p is None:
            p = (k - 1) // 2 * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=p, dilation=d),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, k, padding=p, dilation=d),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.res(x)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class WindowModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT, num_severity: int = 3, ordinal: bool = False, use_age_covariate: bool = False, use_age_adversarial: bool = False):
        super().__init__()
        # Temporal conv frontend over features
        self.tcnn = nn.Sequential(
            TemporalConvBlock(input_dim, 64, k=9, d=1, dropout=dropout),
            TemporalConvBlock(64, 128, k=9, d=2, dropout=dropout),
        )
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.relu = nn.ReLU()
        self.use_age_covariate = use_age_covariate
        fc_in = hidden_dim * 2 + (1 if use_age_covariate else 0)
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pd_head = nn.Linear(128, 2)
        self.ordinal = ordinal
        if ordinal:
            # CORAL: K=3 -> K-1=2 ordinal logits thresholds
            self.sev_head = nn.Linear(128, 2)
        else:
            self.sev_head = nn.Linear(128, num_severity)
        # Adversarial age head (regress normalized age)
        self.use_age_adversarial = use_age_adversarial
        if use_age_adversarial:
            self.age_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, age_cov: Optional[torch.Tensor] = None, lambda_age: float = 0.0):
        # x: (B,T,D)
        b, t, d = x.shape
        h = self.tcnn(x.transpose(1, 2))  # (B, C, T)
        h = h.transpose(1, 2)             # (B, T, C)
        out, _ = self.lstm(h)              # (B, T, 2H)
        B, T, _ = out.shape
        device = out.device
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = (idx < lengths.unsqueeze(1)).float()
        logits = self.attn(out).squeeze(-1)
        logits = logits.masked_fill(mask <= 0, float('-inf'))
        attn = torch.softmax(logits, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), out).squeeze(1)
        pooled = self.bn(pooled)
        pooled = self.relu(pooled)
        if self.use_age_covariate and age_cov is not None:
            z_in = torch.cat([pooled, age_cov], dim=1)
        else:
            z_in = pooled
        z = self.fc(z_in)
        pd_logits = self.pd_head(z)
        sev_logits = self.sev_head(z)
        age_pred = None
        if self.use_age_adversarial:
            z_rev = GradReverse.apply(z, lambda_age)
            age_pred = self.age_head(z_rev)
        return pd_logits, sev_logits, age_pred


def coral_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ordinal loss for K=3 -> logits shape [B,2]; targets in {0,1,2}
    target ordinals:
      y>=1 -> t1=1 else 0
      y>=2 -> t2=1 else 0
    """
    B = targets.shape[0]
    t1 = (targets >= 1).float()
    t2 = (targets >= 2).float()
    T = torch.stack([t1, t2], dim=1)
    return nn.functional.binary_cross_entropy_with_logits(logits, T, reduction='mean')


def build_datasets() -> Tuple[WindowDataset, WindowDataset, int, np.ndarray, dict]:
    ids, series, pd_list, sev_list, ages = load_subject_series(DEMOGRAPHICS_FILE, DATASET_DIR)
    # subject split by PD label
    subj_idx = np.arange(len(ids))
    pd_arr = np.array(pd_list, dtype=int)
    train_idx, val_idx = train_test_split(subj_idx, test_size=0.2, random_state=RANDOM_SEED, stratify=pd_arr)

    # Fit scaler on training subjects only
    # First, stack all training series to compute scaler
    D = series[0].shape[1]
    train_concat = np.concatenate([s for i, s in enumerate(series) if i in set(train_idx)], axis=0)
    scaler = StandardScaler().fit(train_concat)

    # Create windows
    # Age stats on training subjects
    age_vals_train = []
    for i in train_idx:
        age_vals_train.append(ages[i] if ages[i] == ages[i] else np.nan)  # keep NaN
    age_vals_train = np.array(age_vals_train, dtype=np.float32)
    age_mean = float(np.nanmean(age_vals_train)) if np.any(~np.isnan(age_vals_train)) else 0.0
    age_std = float(np.nanstd(age_vals_train) + 1e-6) if np.any(~np.isnan(age_vals_train)) else 1.0

    def gen(split_idx):
        windows_list = []
        lengths_list = []
        pdw = []
        sew = []
        sidw = []
        agew = []
        for i in split_idx:
            seq = series[i]
            seq = scaler.transform(seq)
            w, l = make_windows(seq, WINDOW_LEN, HOP_LEN)
            n = w.shape[0]
            windows_list.append(w)
            lengths_list.append(l)
            pdw.append(np.full(n, pd_list[i], dtype=np.int64))
            sew.append(np.full(n, sev_list[i], dtype=np.int64))
            sidw.append(np.full(n, i, dtype=np.int64))
            # replicate subject age to all windows
            subj_age = ages[i]
            if subj_age != subj_age:  # NaN
                subj_age = age_mean
            agew.append(np.full((n,), subj_age, dtype=np.float32))
        windows = np.concatenate(windows_list, axis=0)
        lengths = np.concatenate(lengths_list, axis=0)
        pdw = np.concatenate(pdw, axis=0)
        sew = np.concatenate(sew, axis=0)
        sidw = np.concatenate(sidw, axis=0)
        agew_arr = np.concatenate(agew, axis=0)
        return windows, lengths, pdw, sew, sidw, agew_arr

    Xtr, Ltr, PDtr, SEVtr, SIDtr, AGEtr = gen(train_idx)
    Xva, Lva, PDva, SEVva, SIDva, AGEva = gen(val_idx)
    train_ds = WindowDataset(Xtr, Ltr, PDtr, SEVtr, SIDtr, AGEtr, age_mean, age_std)
    val_ds = WindowDataset(Xva, Lva, PDva, SEVva, SIDva, AGEva, age_mean, age_std)
    return train_ds, val_ds, D, np.array(ids), {"age_mean": age_mean, "age_std": age_std}


def train_loop(model: nn.Module,
               train_ds: WindowDataset,
               val_ds: WindowDataset,
               use_ordinal: bool,
               use_age_covariate: bool,
               use_age_adversarial: bool,
               lambda_age: float,
               agg: str = "mean",
               time_mask_prob: float = 0.0,
               time_mask_len: int = 0,
               noise_std: float = 0.0,
               tune_threshold: bool = False) -> None:
    # Sampler to balance PD windows
    counts = Counter(train_ds.pd_labels.tolist())
    total = len(train_ds)
    class_weight = {c: total / (2 * counts.get(c, 1)) for c in [0, 1]}
    sample_weights = np.array([class_weight[int(y)] for y in train_ds.pd_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Severity weights
    sev_counts = Counter(train_ds.sev_labels.tolist())
    for c in range(3):
        if sev_counts.get(c, 0) == 0:
            sev_counts[c] = 1
    sev_weights = torch.tensor([len(train_ds) / (3 * sev_counts[c]) for c in range(3)], dtype=torch.float32, device=DEVICE)

    pd_loss = nn.CrossEntropyLoss(weight=torch.tensor([class_weight[0], class_weight[1]], dtype=torch.float32, device=DEVICE))
    sev_ce = nn.CrossEntropyLoss(weight=sev_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    best_val_pd_acc = -1.0
    wait = 0

    def apply_train_augment(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)
        if time_mask_prob > 0 and time_mask_len > 0:
            B, T, D = x.shape
            for i in range(B):
                if torch.rand(()) < time_mask_prob:
                    Li = int(L[i].item())
                    if Li > 0:
                        start_max = max(1, Li - time_mask_len)
                        s = int(torch.randint(0, start_max, (1,)).item())
                        e = min(Li, s + time_mask_len)
                        x[i, s:e, :] = 0.0
        return x

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_pd_acc = 0.0
        tr_sev_acc = 0.0
        tr_n = 0
        for x, L, ypd, ysev, sid, agev in train_loader:
            x = x.to(DEVICE); L = L.to(DEVICE); ypd = ypd.to(DEVICE); ysev = ysev.to(DEVICE); agev = agev.to(DEVICE)
            x = apply_train_augment(x, L)
            optimizer.zero_grad(set_to_none=True)
            pd_logits, sev_logits, age_pred = model(x, L, agev if use_age_covariate else None, lambda_age if use_age_adversarial else 0.0)
            if use_ordinal:
                sev_loss = coral_loss(sev_logits, ysev)
                sev_pred = (torch.sigmoid(sev_logits) > 0.5).sum(dim=1)  # back to {0,1,2}
            else:
                sev_loss = sev_ce(sev_logits, ysev)
                sev_pred = sev_logits.argmax(dim=1)
            loss = pd_loss(pd_logits, ypd) + sev_loss
            if use_age_adversarial and age_pred is not None:
                age_loss = nn.functional.mse_loss(age_pred.squeeze(-1), agev.squeeze(-1))
                loss = loss + age_loss  # GRL already reverses gradient; adding positive loss is correct
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
            optimizer.step()
            scheduler.step()

            tr_loss += float(loss.item()) * x.size(0)
            tr_pd_acc += int((pd_logits.argmax(dim=1) == ypd).sum().item())
            tr_sev_acc += int((sev_pred == ysev).sum().item())
            tr_n += int(x.size(0))

        tr_loss /= max(tr_n, 1)
        tr_pd = tr_pd_acc / max(tr_n, 1)
        tr_sev = tr_sev_acc / max(tr_n, 1)

        # Validation: window-level and subject-level
        model.eval()
        va_loss = 0.0
        va_n = 0
        pd_win_correct = 0
        sev_win_correct = 0
        subj_pd_logits = defaultdict(list)
        subj_pd_true = {}
        subj_pd_votes = defaultdict(list)
        subj_sev_logits = defaultdict(list)
        subj_sev_true = {}
        with torch.no_grad():
            for x, L, ypd, ysev, sid, agev in val_loader:
                x = x.to(DEVICE); L = L.to(DEVICE); ypd = ypd.to(DEVICE); ysev = ysev.to(DEVICE); agev = agev.to(DEVICE)
                pd_logits, sev_logits, age_pred = model(x, L, agev if use_age_covariate else None, lambda_age if use_age_adversarial else 0.0)
                if use_ordinal:
                    sev_loss = coral_loss(sev_logits, ysev)
                    sev_pred = (torch.sigmoid(sev_logits) > 0.5).sum(dim=1)
                    sev_logits_flat = torch.nn.functional.pad(sev_logits, (0,1), value=0.0)  # not directly comparable; keep pred only
                else:
                    sev_loss = sev_ce(sev_logits, ysev)
                    sev_pred = sev_logits.argmax(dim=1)
                loss = pd_loss(pd_logits, ypd) + sev_loss
                if use_age_adversarial and age_pred is not None:
                    age_loss = nn.functional.mse_loss(age_pred.squeeze(-1), agev.squeeze(-1))
                    loss = loss + age_loss
                va_loss += float(loss.item()) * x.size(0)
                va_n += int(x.size(0))
                pd_win_correct += int((pd_logits.argmax(dim=1) == ypd).sum().item())
                sev_win_correct += int((sev_pred == ysev).sum().item())
                # aggregate per subject
                for i in range(x.size(0)):
                    s = int(sid[i].cpu().item())
                    subj_pd_logits[s].append(pd_logits[i].cpu())
                    subj_pd_votes[s].append(int(pd_logits[i].argmax().item()))
                    subj_pd_true[s] = int(ypd[i].cpu().item())
                    if not use_ordinal:
                        subj_sev_logits[s].append(sev_logits[i].cpu())
                        subj_sev_true[s] = int(ysev[i].cpu().item())

        va_loss /= max(va_n, 1)
        pd_win_acc = pd_win_correct / max(va_n, 1)
        sev_win_acc = sev_win_correct / max(va_n, 1)

        # Subject-level metrics
        pd_subj_pred = []
        pd_subj_true = []
        if agg == "vote":
            for s, votes in subj_pd_votes.items():
                # majority vote, tie-breaker by mean logits
                counts = Counter(votes)
                if counts[0] == counts[1]:
                    mean_logits = torch.stack(subj_pd_logits[s], dim=0).mean(dim=0)
                    pred = int(mean_logits.argmax().item())
                else:
                    pred = 0 if counts[0] > counts[1] else 1
                pd_subj_pred.append(pred)
                pd_subj_true.append(subj_pd_true[s])
        else:
            for s, logits_list in subj_pd_logits.items():
                mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
                pd_subj_pred.append(int(mean_logits.argmax().item()))
                pd_subj_true.append(subj_pd_true[s])
        pd_subj_acc = accuracy_score(pd_subj_true, pd_subj_pred) if len(pd_subj_true) else 0.0

        if not use_ordinal and len(subj_sev_logits):
            sev_subj_pred = []
            sev_subj_true_list = []
            for s, logits_list in subj_sev_logits.items():
                mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
                sev_subj_pred.append(int(mean_logits.argmax().item()))
                sev_subj_true_list.append(subj_sev_true[s])
            sev_subj_acc = accuracy_score(sev_subj_true_list, sev_subj_pred)
        else:
            sev_subj_acc = sev_win_acc  # fallback

        print(f"Epoch {epoch}/{EPOCHS} -- TrainLoss: {tr_loss:.4f}, Train PD Acc: {tr_pd:.3f}, Train Sev Acc: {tr_sev:.3f} "
              f"| ValLoss: {va_loss:.4f}, Val PD(win) Acc: {pd_win_acc:.3f}, Val PD(subj) Acc: {pd_subj_acc:.3f}, Val Sev(subj) Acc: {sev_subj_acc:.3f}")

        if tune_threshold and len(subj_pd_logits):
            # threshold tuning on subject-level mean probability
            proba = []
            for s, logits_list in subj_pd_logits.items():
                mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
                p = torch.softmax(mean_logits, dim=0)[1].item()
                proba.append(p)
            ytrue = np.array(pd_subj_true)
            proba = np.array(proba)
            best_acc, best_th = 0.0, 0.5
            for th in np.linspace(0.3, 0.7, 41):
                ypred = (proba >= th).astype(int)
                acc = accuracy_score(ytrue, ypred)
                if acc > best_acc:
                    best_acc, best_th = acc, th
            print(f"  tuned threshold -> PD(subj) Acc: {best_acc:.3f} @ thr={best_th:.2f}")

        # Early stopping on subject-level PD accuracy
        if pd_subj_acc > best_val_pd_acc:
            best_val_pd_acc = pd_subj_acc
            wait = 0
            torch.save(model.state_dict(), "best_gait_windowed.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping. Best PD(subj) Acc: {best_val_pd_acc:.3f}")
                break

    # Final subject-level report
    # Reload best
    model.load_state_dict(torch.load("best_gait_windowed.pth", map_location=DEVICE))
    model.eval()
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    subj_pd_logits = defaultdict(list)
    subj_pd_true = {}
    subj_sev_logits = defaultdict(list)
    subj_sev_true = {}
    with torch.no_grad():
        for x, L, ypd, ysev, sid, agev in val_loader:
            x = x.to(DEVICE); L = L.to(DEVICE); ypd = ypd.to(DEVICE); ysev = ysev.to(DEVICE); agev = agev.to(DEVICE)
            pd_logits, sev_logits, _ = model(x, L, agev if use_age_covariate else None, lambda_age if use_age_adversarial else 0.0)
            for i in range(x.size(0)):
                s = int(sid[i].cpu().item())
                subj_pd_logits[s].append(pd_logits[i].cpu())
                subj_pd_true[s] = int(ypd[i].cpu().item())
                subj_sev_logits[s].append(sev_logits[i].cpu())
                subj_sev_true[s] = int(ysev[i].cpu().item())

    pd_subj_pred, pd_subj_true_list = [], []
    for s, logits_list in subj_pd_logits.items():
        mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        pd_subj_pred.append(int(mean_logits.argmax().item()))
        pd_subj_true_list.append(subj_pd_true[s])
    print("\n=== Final Subject-level PD ===")
    print(classification_report(pd_subj_true_list, pd_subj_pred, target_names=["Control", "PD"]))
    print("PD Subject Accuracy:", accuracy_score(pd_subj_true_list, pd_subj_pred))

    sev_subj_pred, sev_subj_true_list = [], []
    for s, logits_list in subj_sev_logits.items():
        mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
        sev_subj_pred.append(int(mean_logits.argmax().item()))
        sev_subj_true_list.append(subj_sev_true[s])
    print("\n=== Final Subject-level Severity ===")
    print(classification_report(sev_subj_true_list, sev_subj_pred, target_names=["Mild", "Moderate", "Severe"]))
    print("Severity Subject Accuracy:", accuracy_score(sev_subj_true_list, sev_subj_pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--ordinal", action="store_true", help="Use ordinal (CORAL) severity loss")
    parser.add_argument("--age_covariate", action="store_true", help="Concatenate normalized age to pooled features")
    parser.add_argument("--adversarial_age", action="store_true", help="Use adversarial GRL to remove age information")
    parser.add_argument("--lambda_age", type=float, default=LAMBDA_AGE, help="Adversarial age loss strength")
    args = parser.parse_args()

    set_seed()
    train_ds, val_ds, feat_dim, subj_ids, age_stats = build_datasets()
    model = WindowModel(input_dim=feat_dim,
                        ordinal=(args.ordinal or USE_ORDINAL_SEVERITY),
                        use_age_covariate=(args.age_covariate or USE_AGE_COVARIATE),
                        use_age_adversarial=(args.adversarial_age or USE_AGE_ADVERSARIAL)).to(DEVICE)
    if args.train:
        train_loop(model, train_ds, val_ds,
                   use_ordinal=(args.ordinal or USE_ORDINAL_SEVERITY),
                   use_age_covariate=(args.age_covariate or USE_AGE_COVARIATE),
                   use_age_adversarial=(args.adversarial_age or USE_AGE_ADVERSARIAL),
                   lambda_age=args.lambda_age)


if __name__ == "__main__":
    main()

