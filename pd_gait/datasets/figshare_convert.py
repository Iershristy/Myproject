"""
Figshare PD 3D kinematics converter to repository format.

Assumptions:
- Raw per-trial kinematics in CSV (e.g., exported from TRC) with columns like <joint>_x,<joint>_y,<joint>_z
- A metadata CSV with columns: trial_id,pd_label,severity_label (optional),subject_id (optional)

Outputs under --data_dir:
- skeleton/{train,val}/{sample_id}.npy -> [T,J,3] with (x,y,conf)
- silhouette/{train,val}/{sample_id}/frame_XXXX.png (synthetic silhouette from 2D projection)
- labels_{train,val}.csv

Use --views N to synthesize N camera yaws for multi-view training.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
import json
import math

import numpy as np
import pandas as pd
import cv2


DEFAULT_JOINTS = [
    "head","neck","r_shoulder","r_elbow","r_wrist",
    "l_shoulder","l_elbow","l_wrist","pelvis","r_hip",
    "r_knee","r_ankle","l_hip","l_knee","l_ankle","spine","thorax"
]


def load_trial_csv(path: Path, joint_names: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    T = len(df)
    J = len(joint_names)
    arr = np.zeros((T, J, 3), dtype=np.float32)
    # Flexible matching: allow names like <joint>_X or <joint>.X
    for j, name in enumerate(joint_names):
        for k, axis in enumerate(['x', 'y', 'z']):
            candidates = [f"{name}_{axis}", f"{name}.{axis}", f"{name.upper()}_{axis.upper()}", f"{name}_{axis.upper()}"]
            col = None
            for c in candidates:
                if c in df.columns:
                    col = c
                    break
            if col is not None:
                arr[:, j, k] = df[col].values.astype(np.float32)
    return arr


def project_points(points3d: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0, f: float = 1000.0, cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    T, J, _ = points3d.shape
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    Ry = np.array([[ math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float32)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch),  math.cos(pitch)]], dtype=np.float32)
    R = Rx @ Ry
    pts = points3d.reshape(-1, 3) @ R.T
    pts[:, 2] = pts[:, 2] + 2000.0
    x = f * (pts[:, 0] / (pts[:, 2] + 1e-6)) + cx
    y = f * (pts[:, 1] / (pts[:, 2] + 1e-6)) + cy
    uv = np.stack([x, y], axis=-1).reshape(T, J, 2)
    return uv.astype(np.float32)


def write_skeleton_npy(dst: Path, uv: np.ndarray) -> None:
    T, J, _ = uv.shape
    arr = np.zeros((T, J, 3), dtype=np.float32)
    arr[..., :2] = uv
    arr[..., 2] = 1.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)


def synthesize_silhouettes(dst_dir: Path, uv: np.ndarray, img_hw=(128, 88)) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    T, J, _ = uv.shape
    for t in range(T):
        img = np.zeros(img_hw, dtype=np.uint8)
        pairs = [(5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
        for a, b in pairs:
            if a < J and b < J:
                ax, ay = uv[t, a]
                bx, by = uv[t, b]
                ax = int(np.clip(ax / 5, 0, img_hw[1] - 1))
                ay = int(np.clip(ay / 5, 0, img_hw[0] - 1))
                bx = int(np.clip(bx / 5, 0, img_hw[1] - 1))
                by = int(np.clip(by / 5, 0, img_hw[0] - 1))
                cv2.line(img, (ax, ay), (bx, by), 255, thickness=6)
        cv2.imwrite(str(dst_dir / f'frame_{t:04d}.png'), img)


def split_train_val(subjects: List[str], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    subs = np.array(subjects)
    perm = rng.permutation(len(subs))
    nv = int(len(subs) * val_ratio)
    val = subs[perm[:nv]].tolist()
    train = subs[perm[nv:]].tolist()
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description='Figshare PD 3D converter')
    parser.add_argument('--raw_dir', type=Path, required=True, help='Directory with raw trial CSVs')
    parser.add_argument('--metadata_csv', type=Path, required=True, help='CSV with trial_id,pd_label,severity_label,subject_id')
    parser.add_argument('--data_dir', type=Path, default=Path('data'))
    parser.add_argument('--joint_spec', type=Path, default=None, help='JSON list of joint base names')
    parser.add_argument('--views', type=int, default=8)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    joint_names = json.loads(args.joint_spec.read_text()) if args.joint_spec else DEFAULT_JOINTS

    meta = pd.read_csv(args.metadata_csv)
    needed_cols = ['trial_id', 'pd_label']
    for c in needed_cols:
        assert c in meta.columns, f"Missing column in metadata: {c}"
    if 'severity_label' not in meta.columns:
        meta['severity_label'] = -1
    if 'subject_id' not in meta.columns:
        meta['subject_id'] = meta['trial_id']

    # subject-wise split
    subs = meta['subject_id'].astype(str).tolist()
    train_subs, val_subs = split_train_val(subs, val_ratio=args.val_ratio, seed=42)
    train_set = set(train_subs)

    rows_train: List[Dict[str, Any]] = []
    rows_val: List[Dict[str, Any]] = []

    for _, row in meta.iterrows():
        trial_id = str(row['trial_id'])
        pd_label = int(row['pd_label'])
        severity_label = int(row['severity_label']) if not pd.isna(row['severity_label']) else -1
        subject_id = str(row['subject_id'])
        split = 'train' if subject_id in train_set else 'val'

        trial_csv = args.raw_dir / f"{trial_id}.csv"
        if not trial_csv.exists():
            # try nested locations
            cands = list(args.raw_dir.rglob(f"{trial_id}.csv"))
            if len(cands) == 0:
                print(f"Warning: missing trial csv {trial_id}")
                continue
            trial_csv = cands[0]

        pts3d = load_trial_csv(trial_csv, joint_names)
        for v in range(args.views):
            yaw = (360.0 / args.views) * v
            uv = project_points(pts3d, yaw_deg=yaw, pitch_deg=0.0)
            sample_id = f"{trial_id}_v{v}"
            write_skeleton_npy(data_dir / 'skeleton' / split / f'{sample_id}.npy', uv)
            synthesize_silhouettes(data_dir / 'silhouette' / split / sample_id)
            row_o = {
                'sample_id': sample_id,
                'pd_label': pd_label,
                'severity_label': severity_label,
                'view_id': v,
            }
            if split == 'train':
                rows_train.append(row_o)
            else:
                rows_val.append(row_o)

    pd.DataFrame(rows_train).to_csv(data_dir / 'labels_train.csv', index=False)
    pd.DataFrame(rows_val).to_csv(data_dir / 'labels_val.csv', index=False)
    print('Done. Train samples:', len(rows_train), 'Val samples:', len(rows_val))


if __name__ == '__main__':
    main()

