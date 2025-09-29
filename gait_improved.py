#!/usr/bin/env python3
"""
Gait in Parkinson's Disease - Improved Training Script (PhysioNet gaitpdb)

Improvements over baseline:
- Length-aware masked attention pooling (avoids taking last padded timestep)
- Class weights for BOTH PD and severity heads
- AdamW optimizer + OneCycleLR scheduler + gradient clipping
- Early stopping based on validation PD accuracy with best checkpoint

Usage:
    python gait_improved.py --train

Assumptions:
- Demographics Excel file: DEMOGRAPHICS_FILE (.xls in gaitpdb)
- Gait time-series per subject: TXT files named "ID.txt" or "ID_*.txt"
"""

import argparse
import os
from glob import glob
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ------------------------
# CONFIG
# ------------------------
DATASET_DIR = "."  # directory where gait txt files live
DEMOGRAPHICS_FILE = "demographics.xls"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-2
MAX_LEN = 300         # max frames/time-steps to pad/truncate to
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
PATIENCE = 15
CLIP_MAX_NORM = 1.0
RANDOM_SEED = 42


# ------------------------
# Utilities
# ------------------------
def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_gait_file_for_id(dataset_dir: str, subj_id) -> str | None:
    # Support either "ID.txt" or "ID_*.txt"
    candidates: list[str] = []
    candidates += glob(os.path.join(dataset_dir, f"{subj_id}.txt"))
    candidates += glob(os.path.join(dataset_dir, f"{subj_id}_*.txt"))
    return candidates[0] if candidates else None


# ------------------------
# Dataset class
# ------------------------
class GaitDataset(Dataset):
    def __init__(self, features: np.ndarray, pd_labels: np.ndarray, sev_labels: np.ndarray, lengths: np.ndarray):
        """
        features: (N, T, D) sequences padded to T
        lengths:  (N,) true sequence lengths (<= T)
        """
        self.features = features
        self.pd_labels = pd_labels
        self.sev_labels = sev_labels
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y_pd = torch.tensor(self.pd_labels[idx], dtype=torch.long)
        y_sev = torch.tensor(self.sev_labels[idx], dtype=torch.long)
        L = torch.tensor(self.lengths[idx], dtype=torch.long)
        return x, y_pd, y_sev, L


# ------------------------
# Model with masked attention pooling
# ------------------------
class GaitModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT, num_severity: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.relu = nn.ReLU()
        self.fc_common = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pd_head = nn.Linear(128, 2)
        self.sev_head = nn.Linear(128, num_severity)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (B, T, D), lengths: (B,)
        out, _ = self.lstm(x)  # (B, T, 2H)
        B, T, _ = out.shape
        device = out.device
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = (t_idx < lengths.unsqueeze(1)).float()  # (B, T)

        logits = self.attn(out).squeeze(-1)            # (B, T)
        logits = logits.masked_fill(mask <= 0, float('-inf'))
        attn = torch.softmax(logits, dim=1)            # (B, T)
        pooled = torch.bmm(attn.unsqueeze(1), out).squeeze(1)  # (B, 2H)

        pooled = self.bn(pooled)
        pooled = self.relu(pooled)
        common = self.fc_common(pooled)
        return self.pd_head(common), self.sev_head(common)


# ------------------------
# Data loading and preprocessing
# ------------------------
def load_and_align_data(demo_path: str, dataset_dir: str, max_len: int = MAX_LEN):
    """
    Returns:
      ids: list of subject ids used
      features: numpy array shape (N, max_len, D)
      pd_labels: numpy array shape (N,)
      sev_labels: numpy array shape (N,)
      feature_dim: D
      lengths: numpy array shape (N,) true sequence lengths (<= max_len)
    """
    # Try xlrd (for .xls); fallback to openpyxl (for .xlsx)
    try:
        df = pd.read_excel(demo_path, engine="xlrd")
    except Exception:
        df = pd.read_excel(demo_path)  # let pandas choose engine

    # Map PD label
    pd_labels_all = (df["Group"].astype(str).str.upper() == "PD").astype(int)

    # Map severity from HoehnYahr to 3 classes
    def map_sev(x):
        try:
            x = float(x)
            if x <= 2:
                return 0  # Mild
            elif x == 3:
                return 1  # Moderate
            else:
                return 2  # Severe
        except Exception:
            return 0

    sev_labels_all = df["HoehnYahr"].apply(map_sev)

    valid_ids, features_list, pd_list, sev_list, lengths = [], [], [], [], []
    for idx, subj_id in enumerate(df["ID"]):
        fpath = find_gait_file_for_id(dataset_dir, subj_id)
        if fpath is None:
            continue
        try:
            arr = np.loadtxt(fpath)
            if arr.ndim == 1:
                arr = arr[:, None]
            L = min(arr.shape[0], max_len)  # true length
            if arr.shape[0] > max_len:
                arr = arr[:max_len, :]
            elif arr.shape[0] < max_len:
                pad = np.zeros((max_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.vstack([arr, pad])
            features_list.append(arr)
            lengths.append(L)
            pd_list.append(int(pd_labels_all.iloc[idx]))
            sev_list.append(int(sev_labels_all.iloc[idx]))
            valid_ids.append(subj_id)
        except Exception as e:
            print(f"Warning: could not load {fpath} for {subj_id}: {e}")

    if len(features_list) == 0:
        raise RuntimeError("No gait files found. Check DATASET_DIR and filenames (ID.txt or ID_*.txt).")

    # Ensure consistent feature dimension
    D = features_list[0].shape[1]
    for arr in features_list:
        if arr.shape[1] != D:
            raise RuntimeError("Inconsistent feature dimension across gait files. Inspect your data files.")

    features = np.stack(features_list, axis=0)  # (N, T, D)
    pd_labels = np.array(pd_list, dtype=int)
    sev_labels = np.array(sev_list, dtype=int)
    lengths = np.array(lengths, dtype=int)
    return valid_ids, features, pd_labels, sev_labels, D, lengths


# ------------------------
# Fit scaler on TRAINING set only
# ------------------------
def fit_scaler_on_train(features_train: np.ndarray) -> StandardScaler:
    N, T, D = features_train.shape
    flat = features_train.reshape(-1, D)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, T, D = features.shape
    flat = features.reshape(-1, D)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(N, T, D)


# ------------------------
# Training & evaluation
# ------------------------
def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          pd_loss_fn: nn.Module,
          sev_loss_fn: nn.Module,
          optimizer: optim.Optimizer,
          scheduler: optim.lr_scheduler._LRScheduler,
          device: torch.device,
          epochs: int = EPOCHS,
          patience: int = PATIENCE) -> None:
    best_val_pd_acc = -1.0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pd_correct = 0
        sev_correct = 0
        total = 0

        for X, y_pd, y_sev, L in train_loader:
            X = X.to(device); y_pd = y_pd.to(device); y_sev = y_sev.to(device); L = L.to(device)
            optimizer.zero_grad(set_to_none=True)
            pd_logits, sev_logits = model(X, L)
            loss = pd_loss_fn(pd_logits, y_pd) + sev_loss_fn(sev_logits, y_sev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_MAX_NORM)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += float(loss.item()) * X.size(0)
            pd_preds = pd_logits.argmax(dim=1)
            sev_preds = sev_logits.argmax(dim=1)
            pd_correct += int((pd_preds == y_pd).sum().item())
            sev_correct += int((sev_preds == y_sev).sum().item())
            total += int(X.size(0))

        train_loss = running_loss / max(total, 1)
        train_pd_acc = pd_correct / max(total, 1)
        train_sev_acc = sev_correct / max(total, 1)

        # validation
        model.eval()
        val_loss = 0.0
        val_pd_correct = 0
        val_sev_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y_pd, y_sev, L in val_loader:
                X = X.to(device); y_pd = y_pd.to(device); y_sev = y_sev.to(device); L = L.to(device)
                pd_logits, sev_logits = model(X, L)
                loss = pd_loss_fn(pd_logits, y_pd) + sev_loss_fn(sev_logits, y_sev)
                val_loss += float(loss.item()) * X.size(0)
                val_pd_correct += int((pd_logits.argmax(dim=1) == y_pd).sum().item())
                val_sev_correct += int((sev_logits.argmax(dim=1) == y_sev).sum().item())
                val_total += int(X.size(0))

        val_loss = val_loss / max(val_total, 1)
        val_pd_acc = val_pd_correct / max(val_total, 1)
        val_sev_acc = val_sev_correct / max(val_total, 1)

        print(f"Epoch {epoch}/{epochs} -- TrainLoss: {train_loss:.4f}, Train PD Acc: {train_pd_acc:.3f}, Train Sev Acc: {train_sev_acc:.3f} "
              f"| ValLoss: {val_loss:.4f}, Val PD Acc: {val_pd_acc:.3f}, Val Sev Acc: {val_sev_acc:.3f}")

        # Early stopping & checkpoint (based on PD accuracy)
        if val_pd_acc > best_val_pd_acc:
            best_val_pd_acc = val_pd_acc
            wait = 0
            torch.save(model.state_dict(), "best_gait_model.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping (no PD accuracy improvement for {patience} epochs). Best PD Acc: {best_val_pd_acc:.3f}")
                break


@torch.no_grad()
def final_evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    all_pd: list[int] = []
    all_pd_preds: list[int] = []
    all_sev: list[int] = []
    all_sev_preds: list[int] = []
    for X, y_pd, y_sev, L in loader:
        X = X.to(device); L = L.to(device)
        pd_logits, sev_logits = model(X, L)
        all_pd.extend(y_pd.numpy().tolist())
        all_pd_preds.extend(pd_logits.argmax(dim=1).cpu().numpy().tolist())
        all_sev.extend(y_sev.numpy().tolist())
        all_sev_preds.extend(sev_logits.argmax(dim=1).cpu().numpy().tolist())

    print("\n=== Final Evaluation on Validation Set ===")
    print("PD Detection Report:")
    print(classification_report(all_pd, all_pd_preds, target_names=["Control", "PD"]))
    print("Severity Classification Report:")
    print(classification_report(all_sev, all_sev_preds, target_names=["Mild", "Moderate", "Severe"]))
    print("PD Accuracy:", accuracy_score(all_pd, all_pd_preds))
    print("Severity Accuracy:", accuracy_score(all_sev, all_sev_preds))


# ------------------------
# Main
# ------------------------
def main(args: argparse.Namespace) -> None:
    set_seed()
    ids, features, pd_labels, sev_labels, feat_dim, lengths = load_and_align_data(DEMOGRAPHICS_FILE, DATASET_DIR, max_len=MAX_LEN)
    print(f"Found {len(ids)} subjects with gait data. Feature dim = {feat_dim}")

    # subject-level split (stratify by PD)
    subj_indices = np.arange(len(ids))
    train_idx, val_idx = train_test_split(subj_indices, test_size=0.2, random_state=RANDOM_SEED, stratify=pd_labels)
    X_train = features[train_idx]
    X_val = features[val_idx]
    L_train = lengths[train_idx]
    L_val = lengths[val_idx]
    pd_train = pd_labels[train_idx]
    pd_val = pd_labels[val_idx]
    sev_train = sev_labels[train_idx]
    sev_val = sev_labels[val_idx]

    # fit scaler on training data only
    scaler = fit_scaler_on_train(X_train)
    X_train = apply_scaler(X_train, scaler)
    X_val = apply_scaler(X_val, scaler)
    with open("gait_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Datasets & loaders
    train_dataset = GaitDataset(X_train, pd_train, sev_train, L_train)
    val_dataset = GaitDataset(X_val, pd_val, sev_val, L_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Class weights
    # Severity weights (3 classes)
    sev_counts = Counter(sev_train.tolist())
    for c in range(3):
        if sev_counts.get(c, 0) == 0:
            sev_counts[c] = 1
    sev_weights = [len(sev_train) / (3 * sev_counts[c]) for c in range(3)]
    sev_weights = torch.tensor(sev_weights, dtype=torch.float32).to(DEVICE)
    print("Severity class weights:", sev_weights.cpu().numpy())

    # PD weights (2 classes)
    pd_counts = Counter(pd_train.tolist())
    for c in [0, 1]:
        if pd_counts.get(c, 0) == 0:
            pd_counts[c] = 1
    pd_weights = [len(pd_train) / (2 * pd_counts[0]), len(pd_train) / (2 * pd_counts[1])]
    pd_weights = torch.tensor(pd_weights, dtype=torch.float32).to(DEVICE)
    print("PD class weights:", pd_weights.cpu().numpy())

    # model, losses, optimizer, scheduler
    model = GaitModel(input_dim=feat_dim).to(DEVICE)
    pd_loss_fn = nn.CrossEntropyLoss(weight=pd_weights)
    sev_loss_fn = nn.CrossEntropyLoss(weight=sev_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    if args.train:
        train(model, train_loader, val_loader, pd_loss_fn, sev_loss_fn, optimizer, scheduler, DEVICE, epochs=EPOCHS, patience=PATIENCE)
        # load best and evaluate
        model.load_state_dict(torch.load("best_gait_model.pth", map_location=DEVICE))
        final_evaluate(model, val_loader, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    args = parser.parse_args()
    main(args)

