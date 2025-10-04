#!/usr/bin/env python3
"""
Hybrid PhysioNet Gait Analysis (PD + Severity)

Novel hybrid approach combining:
- Enhanced multi-branch temporal encoder (from gait11-style CNN+BiLSTM+MHA)
- Two specialized heads/models (from gait17) for PD and Severity
- Self-supervised-ish supervised contrastive pretraining + prototype-based classification
- Fine-tuning per task with focal (PD) and class-weighted CE (Severity)
- Test-time augmentation (TTA) and robust subject-level aggregation

Goal (dataset-dependent): reach strong PD (>92%) and Severity (>88%) subject accuracy.
This code provides a principled path; tune epochs and capacity if needed.

Run examples:
  # Full pipeline: pretrain encoder, fine-tune PD and Severity, evaluate
  python gait_hybrid.py --train

  # Individual stages
  python gait_hybrid.py --stage pretrain
  python gait_hybrid.py --stage pd
  python gait_hybrid.py --stage sev
  python gait_hybrid.py --stage eval --tta 8

Notes:
- Uses joint stratification by (PD x Severity) to ensure classes in both splits
- Uses strict masked pooling and LayerNorm to avoid BN shape issues post-RNN
- Prototypes are learnable cosine classifiers with temperature and regularization
"""

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

# ==================== CONFIG ====================
DATASET_DIR = "."
DEMOGRAPHICS_FILE = "demographics.xls"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds and reproducibility
SEED = 42

# Windowing
WINDOW_LEN = 512
HOP_LEN = 160
MAX_LEN_CAP = 8192

# Model
HIDDEN_DIM = 192
NUM_LAYERS = 3
DROPOUT = 0.35
ATTN_HEADS = 8

# Optimization
BATCH_SIZE_PRE = 64
BATCH_SIZE_FT = 32
LR_PRETRAIN = 1e-3
LR_FINETUNE = 6e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 12
CLIP_MAX_NORM = 1.0

# Epochs (tune up for best performance on your GPU)
EPOCHS_PRETRAIN = 60
EPOCHS_PD = 80
EPOCHS_SEV = 80

# Augmentations (light but effective)
TIME_MASK_PROB = 0.2
TIME_MASK_LEN = 48
NOISE_STD = 0.03
MIXUP_ALPHA = 0.0  # disabled for stability on severity

# Focal (PD)
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# SupCon / Prototype scaling
SUPCON_TEMP_PD = 0.25
SUPCON_TEMP_SEV = 0.25
PROTOTYPE_TEMP = 0.10
LAMBDA_SUPCON_PD = 0.4
LAMBDA_SUPCON_SEV = 0.6
LAMBDA_CE_PROTO = 0.3
LAMBDA_CENTER = 0.2
LAMBDA_PROTO_ORTHO = 0.01

# Eval
VAL_SIZE = 0.20  # joint stratification split
DEFAULT_TTA = 8


# ==================== UTILS ====================
def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_gait_file_for_id(dataset_dir: str, subj_id) -> Optional[str]:
    candidates: List[str] = []
    candidates += glob(os.path.join(dataset_dir, f"{subj_id}.txt"))
    candidates += glob(os.path.join(dataset_dir, f"{subj_id}_*.txt"))
    return candidates[0] if candidates else None


def load_subject_series(demo_path: str, dataset_dir: str):
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


def make_windows(seq: np.ndarray, wlen: int, hop: int):
    T, D = seq.shape
    if T <= wlen:
        pad = np.zeros((wlen - T, D), dtype=seq.dtype)
        return np.expand_dims(np.vstack([seq, pad]), 0), np.array([T], dtype=np.int64)
    out, lens = [], []
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


# ==================== LOSSES ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        targets = targets.long()
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss with multi-positive labels.
    Computes separate losses for PD and Severity, combines them.
    """
    def __init__(self, temp_pd: float, temp_sev: float,
                 lambda_pd: float = LAMBDA_SUPCON_PD,
                 lambda_sev: float = LAMBDA_SUPCON_SEV):
        super().__init__()
        self.temp_pd = temp_pd
        self.temp_sev = temp_sev
        self.lambda_pd = lambda_pd
        self.lambda_sev = lambda_sev

    @staticmethod
    def _supcon_from_mask(z: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
        # z: [B, d] normalized
        # mask: [B, B] bool, True marks positives (exclude self separately)
        B = z.size(0)
        sim = torch.matmul(z, z.t()) / max(1e-6, temperature)  # [B,B]
        logits = sim - torch.diag(torch.diag(sim))  # zero out self-sim for numerical stability
        # For stability, mask self later
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits) * (~torch.eye(B, dtype=torch.bool, device=z.device))
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # Positive log-prob
        pos_mask = mask & (~torch.eye(B, dtype=torch.bool, device=z.device))
        pos_counts = pos_mask.sum(dim=1)  # [B]
        # Avoid div by zero: only include anchors with >=1 positive
        valid = pos_counts > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        loss = -(log_prob * pos_mask.float()).sum(dim=1)
        loss = loss[valid] / pos_counts[valid].float()
        return loss.mean()

    def forward(self, z: torch.Tensor, pd_labels: torch.Tensor, sev_labels: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings
        z = F.normalize(z, dim=1)
        pd_mask = (pd_labels.unsqueeze(1) == pd_labels.unsqueeze(0))
        sev_mask = (sev_labels.unsqueeze(1) == sev_labels.unsqueeze(0))
        loss_pd = self._supcon_from_mask(z, pd_mask, self.temp_pd)
        loss_sev = self._supcon_from_mask(z, sev_mask, self.temp_sev)
        return self.lambda_pd * loss_pd + self.lambda_sev * loss_sev


# ==================== MODEL ====================
class EnhancedTemporalBlock(nn.Module):
    """Multi-branch temporal block with Squeeze-and-Excite and residual.
    Branches at kernel sizes 3/7/11 to capture multi-scale gait patterns.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        ch1 = out_ch // 3
        ch2 = out_ch // 3
        ch3 = out_ch - ch1 - ch2
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, ch1, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, ch2, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch2),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, ch3, kernel_size=11, padding=5),
            nn.BatchNorm1d(ch3),
            nn.ReLU(inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, out_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch // 4, out_ch, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = out * self.se(out)
        out = self.dropout(out)
        return out + self.res(x)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
                 attn_heads: int = ATTN_HEADS):
        super().__init__()
        self.tcnn = nn.Sequential(
            EnhancedTemporalBlock(input_dim, 64, dropout),
            EnhancedTemporalBlock(64, 128, dropout),
            EnhancedTemporalBlock(128, 192, dropout),
            EnhancedTemporalBlock(192, 256, dropout),
        )
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        self.attn = nn.MultiheadAttention(hidden_dim * 2, num_heads=attn_heads,
                                          dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, lengths):
        # x: [B,T,D]
        B, T, D = x.shape
        h = self.tcnn(x.transpose(1, 2)).transpose(1, 2)  # [B,T,256]
        out, _ = self.lstm(h)                              # [B,T,2H]
        out = self.ln1(out)
        device = out.device
        mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = (mask >= lengths.unsqueeze(1))             # [B,T] True for padding
        attn_out, _ = self.attn(out, out, out, key_padding_mask=mask)
        attn_out = self.ln2(attn_out + out)
        valid = (~mask).float()
        den = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (attn_out * valid.unsqueeze(-1)).sum(dim=1) / den   # [B,2H]
        return pooled


class CosinePrototypeHead(nn.Module):
    """Learnable class prototypes with cosine similarity classifier."""
    def __init__(self, in_dim: int, num_classes: int, temperature: float = PROTOTYPE_TEMP):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.temperature = temperature

    def forward(self, x):
        # x: [B, d], cosine logits: [B, C]
        x_n = F.normalize(x, dim=1)
        p_n = F.normalize(self.prototypes, dim=1)
        logits = torch.matmul(x_n, p_n.t()) / max(1e-6, self.temperature)
        return logits

    def center_loss(self, x, y):
        # Encourage embeddings to be close to their class prototype
        x_n = F.normalize(x, dim=1)
        p_n = F.normalize(self.prototypes, dim=1)
        d = 1.0 - torch.matmul(x_n, p_n.t())  # cosine distance
        return F.mse_loss(d[torch.arange(x.size(0), device=x.device), y], torch.zeros_like(y, dtype=torch.float, device=x.device))

    def orthogonal_penalty(self):
        # Encourage prototypes to be orthogonal
        p_n = F.normalize(self.prototypes, dim=1)
        gram = torch.matmul(p_n, p_n.t())
        I = torch.eye(gram.size(0), device=gram.device)
        off_diag = gram - I
        return (off_diag ** 2).mean()


class PDHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)


class SevHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)


class PDModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.enc = Encoder(input_dim)
        self.head = PDHead(2 * HIDDEN_DIM)
    def forward(self, x, lengths):
        return self.head(self.enc(x, lengths))


class SevModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.enc = Encoder(input_dim)
        self.head = SevHead(2 * HIDDEN_DIM)
    def forward(self, x, lengths):
        return self.head(self.enc(x, lengths))


class ModelEma:
    """Exponential Moving Average for model parameters (non-decoupled)."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = self._clone_model(model)
        self.decay = decay
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        import copy
        ema = copy.deepcopy(model)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))


# ==================== AUGMENTATION ====================
def apply_augmentations(x, L, noise_std=NOISE_STD, mask_prob=TIME_MASK_PROB, mask_len=TIME_MASK_LEN):
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


def tta_batch(model: nn.Module, x: torch.Tensor, L: torch.Tensor, num: int = DEFAULT_TTA):
    # returns averaged logits over TTA variants
    model.eval()
    logits_accum = None
    with torch.no_grad():
        for _ in range(max(1, num)):
            xa = apply_augmentations(x.clone(), L)
            out = model(xa, L)
            if logits_accum is None:
                logits_accum = out
            else:
                logits_accum = logits_accum + out
    return logits_accum / float(max(1, num))


# ==================== DATA BUILD ====================
def build_datasets():
    ids, series, pd_list, sev_list = load_subject_series(DEMOGRAPHICS_FILE, DATASET_DIR)

    subj_idx = np.arange(len(ids))
    pd_arr = np.array(pd_list, dtype=int)
    sev_arr = np.array(sev_list, dtype=int)
    joint = pd_arr * 3 + sev_arr
    train_idx, val_idx = train_test_split(subj_idx, test_size=VAL_SIZE, random_state=SEED, stratify=joint)

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


# ==================== PRETRAIN ENCODER ====================
def pretrain_encoder(train_ds: WindowDataset, val_ds: WindowDataset, feat_dim: int, save_path: str = "best_encoder.pth"):
    enc = Encoder(feat_dim).to(DEVICE)
    # Prototypes for PD and Severity
    proto_pd = CosinePrototypeHead(2 * HIDDEN_DIM, 2).to(DEVICE)
    proto_sev = CosinePrototypeHead(2 * HIDDEN_DIM, 3).to(DEVICE)

    params = list(enc.parameters()) + list(proto_pd.parameters()) + list(proto_sev.parameters())
    opt = optim.AdamW(params, lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=12, T_mult=2)

    supcon = SupConLoss(SUPCON_TEMP_PD, SUPCON_TEMP_SEV)

    # Weighted samplers based on severity (to surface 'Moderate')
    sev_counts = Counter(train_ds.sev_labels.tolist())
    sev_w = {c: len(train_ds) / (3 * max(sev_counts.get(c, 1), 1)) for c in [0, 1, 2]}
    sample_weights = np.array([sev_w[int(y)] for y in train_ds.sev_labels], dtype=np.float32)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_PRE,
                              sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_PRE, shuffle=False)

    ema = ModelEma(enc, decay=0.998)

    best_val_obj = float('inf')
    wait = 0

    for epoch in range(1, EPOCHS_PRETRAIN + 1):
        enc.train(); proto_pd.train(); proto_sev.train()
        total_loss = 0.0; n_ex = 0
        for x, L, ypd, ysev, sid in train_loader:
            x = x.to(DEVICE); L = L.to(DEVICE)
            ypd = ypd.to(DEVICE); ysev = ysev.to(DEVICE)
            x = apply_augmentations(x, L)
            opt.zero_grad(set_to_none=True)
            z = enc(x, L)
            loss_sup = supcon(z, ypd, ysev)
            # Prototype CE
            log_pd = proto_pd(z)
            log_sev = proto_sev(z)
            # Class weights
            pd_counts = Counter(train_ds.pd_labels.tolist())
            pd_w = torch.tensor([len(train_ds) / (2 * max(pd_counts.get(c, 1), 1)) for c in [0, 1]], dtype=torch.float32, device=DEVICE)
            sev_w_t = torch.tensor([sev_w[c] for c in [0, 1, 2]], dtype=torch.float32, device=DEVICE)
            ce_pd = F.cross_entropy(log_pd, ypd, weight=pd_w)
            ce_sev = F.cross_entropy(log_sev, ysev, weight=sev_w_t)
            # Prototype regularizers
            center = proto_pd.center_loss(z, ypd) + proto_sev.center_loss(z, ysev)
            ortho = proto_pd.orthogonal_penalty() + proto_sev.orthogonal_penalty()
            loss = loss_sup + LAMBDA_CE_PROTO * (ce_pd + ce_sev) + LAMBDA_CENTER * center + LAMBDA_PROTO_ORTHO * ortho
            loss.backward()
            nn.utils.clip_grad_norm_(params, CLIP_MAX_NORM)
            opt.step()
            ema.update(enc)
            total_loss += loss.item() * x.size(0)
            n_ex += x.size(0)
        sched.step()
        total_loss /= max(1, n_ex)

        # Validation proxy: CE on prototypes + center/ortho using EMA encoder
        enc.eval(); proto_pd.eval(); proto_sev.eval()
        with torch.no_grad():
            v_loss = 0.0; v_n = 0
            for x, L, ypd, ysev, sid in val_loader:
                x = x.to(DEVICE); L = L.to(DEVICE)
                ypd = ypd.to(DEVICE); ysev = ysev.to(DEVICE)
                z = ema.ema(x, L)
                lp = proto_pd(z); ls = proto_sev(z)
                pd_counts = Counter(train_ds.pd_labels.tolist())
                pd_w = torch.tensor([len(train_ds) / (2 * max(pd_counts.get(c, 1), 1)) for c in [0, 1]], dtype=torch.float32, device=DEVICE)
                sev_w_t = torch.tensor([sev_w[c] for c in [0, 1, 2]], dtype=torch.float32, device=DEVICE)
                ce_pd_v = F.cross_entropy(lp, ypd, weight=pd_w)
                ce_sev_v = F.cross_entropy(ls, ysev, weight=sev_w_t)
                center_v = proto_pd.center_loss(z, ypd) + proto_sev.center_loss(z, ysev)
                ortho_v = proto_pd.orthogonal_penalty() + proto_sev.orthogonal_penalty()
                l = (ce_pd_v + ce_sev_v) + 0.5 * center_v + 0.01 * ortho_v
                v_loss += l.item() * x.size(0)
                v_n += x.size(0)
            v_loss /= max(1, v_n)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[PRE] Epoch {epoch:3d}/{EPOCHS_PRETRAIN} | TrainLoss: {total_loss:.4f} | ValProxy: {v_loss:.4f}")

        if v_loss < best_val_obj:
            best_val_obj = v_loss; wait = 0
            # Save EMA encoder weights
            torch.save(ema.ema.state_dict(), save_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[PRE] Early stopping at epoch {epoch}. Best Proxy: {best_val_obj:.4f}")
                break


# ==================== FINE-TUNE PD ====================
def train_pd_model(train_ds: WindowDataset, val_ds: WindowDataset, feat_dim: int, enc_ckpt: str = "best_encoder.pth"):
    model = PDModel(feat_dim).to(DEVICE)
    # Load encoder weights
    try:
        state = torch.load(enc_ckpt, map_location=DEVICE)
        model.enc.load_state_dict(state, strict=False)
        print("[PD] Loaded pretrained encoder")
    except Exception as e:
        print(f"[PD] Warning: could not load encoder weights: {e}")

    # PD-focused sampling
    pd_counts = Counter(train_ds.pd_labels.tolist())
    pd_w = {c: len(train_ds) / (2 * max(pd_counts.get(c, 1), 1)) for c in [0, 1]}
    sample_weights = np.array([pd_w[int(y)] for y in train_ds.pd_labels], dtype=np.float32)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_FT,
                              sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_FT, shuffle=False)

    pd_weights = torch.tensor([pd_w[0], pd_w[1]], dtype=torch.float32, device=DEVICE)
    criterion = FocalLoss(weight=pd_weights)

    # Differential LR: smaller for encoder, larger for head
    opt = optim.AdamW([
        {"params": model.enc.parameters(), "lr": LR_FINETUNE * 0.5},
        {"params": model.head.parameters(), "lr": LR_FINETUNE},
    ], lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=12, T_mult=2)

    best_pd, wait = 0.0, 0

    for epoch in range(1, EPOCHS_PD + 1):
        model.train()
        for x, L, ypd, _, sid in train_loader:
            x, L, ypd = x.to(DEVICE), L.to(DEVICE), ypd.to(DEVICE)
            x = apply_augmentations(x, L)
            opt.zero_grad(set_to_none=True)
            logits = model(x, L)
            loss = criterion(logits, ypd)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
            opt.step()
        sched.step()

        # val subject-level majority vote with TTA
        model.eval()
        subj_logits, subj_true = defaultdict(list), {}
        with torch.no_grad():
            for x, L, ypd, _, sid in val_loader:
                x, L = x.to(DEVICE), L.to(DEVICE)
                logits = tta_batch(model, x, L, num=DEFAULT_TTA)
                for i in range(x.size(0)):
                    s = int(sid[i].item())
                    subj_logits[s].append(logits[i].cpu())
                    subj_true[s] = int(ypd[i].item())
        preds, trues = [], []
        for s, logs in subj_logits.items():
            votes = [int(z.argmax().item()) for z in logs]
            c = Counter(votes)
            if c[0] == c[1]:
                mean_logits = torch.stack(logs, dim=0).mean(dim=0)
                p = int(mean_logits.argmax().item())
            else:
                p = 0 if c[0] > c[1] else 1
            preds.append(p); trues.append(subj_true[s])
        pd_acc = accuracy_score(trues, preds)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[PD] Epoch {epoch:3d}/{EPOCHS_PD} | Val PD Acc: {pd_acc:.3f}")

        if pd_acc > best_pd:
            best_pd = pd_acc; wait = 0
            torch.save(model.state_dict(), "best_pd_model.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[PD] Early stopping at epoch {epoch}. Best PD Acc: {best_pd:.3f}")
                break


# ==================== FINE-TUNE SEVERITY ====================
def train_sev_model(train_ds: WindowDataset, val_ds: WindowDataset, feat_dim: int, enc_ckpt: str = "best_encoder.pth"):
    model = SevModel(feat_dim).to(DEVICE)
    try:
        state = torch.load(enc_ckpt, map_location=DEVICE)
        model.enc.load_state_dict(state, strict=False)
        print("[SEV] Loaded pretrained encoder")
    except Exception as e:
        print(f"[SEV] Warning: could not load encoder weights: {e}")

    # Severity-focused sampling (boost 'Moderate')
    sev_counts = Counter(train_ds.sev_labels.tolist())
    sev_w = {c: len(train_ds) / (3 * max(sev_counts.get(c, 1), 1)) for c in [0, 1, 2]}
    sample_weights = np.array([sev_w[int(y)] for y in train_ds.sev_labels], dtype=np.float32)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_FT,
                              sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_FT, shuffle=False)

    sev_weights = torch.tensor([sev_w[0], sev_w[1], sev_w[2]], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=sev_weights)

    opt = optim.AdamW([
        {"params": model.enc.parameters(), "lr": LR_FINETUNE * 0.5},
        {"params": model.head.parameters(), "lr": LR_FINETUNE},
    ], lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=12, T_mult=2)

    best_sev, wait = 0.0, 0

    for epoch in range(1, EPOCHS_SEV + 1):
        model.train()
        for x, L, _, ysev, sid in train_loader:
            x, L, ysev = x.to(DEVICE), L.to(DEVICE), ysev.to(DEVICE)
            x = apply_augmentations(x, L)
            opt.zero_grad(set_to_none=True)
            logits = model(x, L)
            loss = criterion(logits, ysev)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM)
            opt.step()
        sched.step()

        # val subject-level severity vote with TTA
        model.eval()
        subj_logs, subj_true = defaultdict(list), {}
        with torch.no_grad():
            for x, L, _, ysev, sid in val_loader:
                x, L = x.to(DEVICE), L.to(DEVICE)
                logits = tta_batch(model, x, L, num=DEFAULT_TTA)
                for i in range(x.size(0)):
                    s = int(sid[i].item())
                    subj_logs[s].append(logits[i].cpu())
                    subj_true[s] = int(ysev[i].item())
        preds, trues = [], []
        for s, logs in subj_logs.items():
            classes = [int(z.argmax().item()) for z in logs]
            p = Counter(classes).most_common(1)[0][0]
            preds.append(p); trues.append(subj_true[s])
        sev_acc = accuracy_score(trues, preds)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[SEV] Epoch {epoch:3d}/{EPOCHS_SEV} | Val Sev Acc: {sev_acc:.3f}")

        if sev_acc > best_sev:
            best_sev = sev_acc; wait = 0
            torch.save(model.state_dict(), "best_sev_model.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"[SEV] Early stopping at epoch {epoch}. Best Sev Acc: {best_sev:.3f}")
                break


# ==================== EVALUATION ====================
def evaluate(val_ds: WindowDataset, feat_dim: int, tta: int = DEFAULT_TTA):
    pd_model = PDModel(feat_dim).to(DEVICE)
    sev_model = SevModel(feat_dim).to(DEVICE)
    pd_model.load_state_dict(torch.load("best_pd_model.pth", map_location=DEVICE))
    sev_model.load_state_dict(torch.load("best_sev_model.pth", map_location=DEVICE))

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_FT, shuffle=False)

    # PD
    pd_model.eval()
    subj_pd_logits, subj_pd_true = defaultdict(list), {}
    with torch.no_grad():
        for x, L, ypd, _, sid in val_loader:
            x, L = x.to(DEVICE), L.to(DEVICE)
            logits = tta_batch(pd_model, x, L, num=tta)
            for i in range(x.size(0)):
                s = int(sid[i].item())
                subj_pd_logits[s].append(logits[i].cpu())
                subj_pd_true[s] = int(ypd[i].item())
    pd_preds, pd_trues = [], []
    for s, logs in subj_pd_logits.items():
        votes = [int(z.argmax().item()) for z in logs]
        c = Counter(votes)
        if c[0] == c[1]:
            mean_logits = torch.stack(logs, dim=0).mean(dim=0)
            p = int(mean_logits.argmax().item())
        else:
            p = 0 if c[0] > c[1] else 1
        pd_preds.append(p); pd_trues.append(subj_pd_true[s])
    pd_acc = accuracy_score(pd_trues, pd_preds)

    # Severity
    sev_model.eval()
    subj_sev_logs, subj_sev_true = defaultdict(list), {}
    with torch.no_grad():
        for x, L, _, ysev, sid in val_loader:
            x, L = x.to(DEVICE), L.to(DEVICE)
            logits = tta_batch(sev_model, x, L, num=tta)
            for i in range(x.size(0)):
                s = int(sid[i].item())
                subj_sev_logs[s].append(logits[i].cpu())
                subj_sev_true[s] = int(ysev[i].item())
    sev_preds, sev_trues = [], []
    for s, logs in subj_sev_logs.items():
        classes = [int(z.argmax().item()) for z in logs]
        p = Counter(classes).most_common(1)[0][0]
        sev_preds.append(p); sev_trues.append(subj_sev_true[s])
    sev_acc = accuracy_score(sev_trues, sev_preds)

    print("\n" + "=" * 70)
    print("=== Subject-level PD (PD-model) ===")
    print(classification_report(pd_trues, pd_preds, target_names=["Control", "PD"]))
    print(f"PD Subject Accuracy: {pd_acc:.4f}")

    print("\n=== Subject-level Severity (SEV-model) ===")
    print(classification_report(sev_trues, sev_preds, target_names=["Mild", "Moderate", "Severe"]))
    print(f"Severity Subject Accuracy: {sev_acc:.4f}")
    print("=" * 70)
    return pd_acc, sev_acc


# ==================== ORCHESTRATION ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run full pipeline: pretrain, PD, SEV, eval")
    parser.add_argument("--stage", type=str, default=None, choices=["pretrain", "pd", "sev", "eval"], help="Run a single stage")
    parser.add_argument("--tta", type=int, default=DEFAULT_TTA, help="Number of TTA samples during eval")
    args = parser.parse_args()

    set_seed()
    print("=" * 70)
    print("Hybrid Gait Analysis: Enhanced Encoder + SupCon+Prototype pretrain + Specialized Heads + TTA")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    train_ds, val_ds, feat_dim = build_datasets()
    print(f"Training windows: {len(train_ds)}, Validation windows: {len(val_ds)}")

    if args.train or args.stage == "pretrain":
        pretrain_encoder(train_ds, val_ds, feat_dim)

    if args.train or args.stage == "pd":
        train_pd_model(train_ds, val_ds, feat_dim)

    if args.train or args.stage == "sev":
        train_sev_model(train_ds, val_ds, feat_dim)

    if args.train or args.stage == "eval":
        evaluate(val_ds, feat_dim, tta=args.tta)


if __name__ == "__main__":
    main()
