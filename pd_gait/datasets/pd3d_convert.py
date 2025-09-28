"""
Converter for PD 3D kinematics/kinetics dataset into repository format.

Input (raw_dir):
  - Trials containing 3D joint trajectories (e.g., CSV with columns per marker)
  - Metadata CSV with columns: sample_id,pd_label,severity_label,view_id (if view)

Output (data_dir):
  - skeleton/{train,val}/{sample_id}.npy -> [T, J, C] with C=3 (x,y,conf)
  - silhouette/{train,val}/{sample_id}/frame_XXXX.png (optional if source has videos)
  - imu/{train,val}/{sample_id}.npy -> [T, S] if available
  - labels_{train,val}.csv

Notes:
  - For 3D motion capture, we project to a synthetic camera to obtain 2D joints
    and silhouettes for multi-view training.
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
import json
import math

import numpy as np
import pandas as pd
import cv2


def load_trial_csv(path: Path, joint_names: List[str]) -> np.ndarray:
	"""Load a trial CSV with columns like joint_x, joint_y, joint_z per joint.
	Returns array [T, J, 3]. Missing joints filled with 0.
	"""
	df = pd.read_csv(path)
	T = len(df)
	J = len(joint_names)
	arr = np.zeros((T, J, 3), dtype=np.float32)
	for j, name in enumerate(joint_names):
		for k, axis in enumerate(['x', 'y', 'z']):
			col = f'{name}_{axis}'
			if col in df.columns:
				arr[:, j, k] = df[col].values.astype(np.float32)
	return arr


def project_points(points3d: np.ndarray, yaw_deg: float = 0.0, pitch_deg: float = 0.0, f: float = 1000.0, cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
	"""Simple pinhole projection of [T,J,3] to [T,J,2]."""
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
	# shift Z to be positive
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
		# draw limbs as thick lines between a few typical connections
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


def split_train_val(ids: List[str], val_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
	n = len(ids)
	perm = np.random.permutation(n)
	nv = int(n * val_ratio)
	val_ids = [ids[i] for i in perm[:nv]]
	train_ids = [ids[i] for i in perm[nv:]]
	return train_ids, val_ids


def main() -> None:
	parser = argparse.ArgumentParser(description='Convert PD 3D kinematics to repo format')
	parser.add_argument('--raw_dir', type=Path, default=Path('data/raw/pd3d'))
	parser.add_argument('--data_dir', type=Path, default=Path('data'))
	parser.add_argument('--joint_spec', type=Path, default=None, help='JSON list of joint base names')
	parser.add_argument('--metadata_csv', type=Path, default=None, help='CSV with sample_id,pd_label,severity_label,view_id')
	parser.add_argument('--glob', type=str, default='*.csv', help='Pattern for trial files')
	parser.add_argument('--views', type=int, default=4, help='Number of synthetic camera yaws to render')
	args = parser.parse_args()

	raw_dir = args.raw_dir
	data_dir = args.data_dir
	data_dir.mkdir(parents=True, exist_ok=True)

	if args.joint_spec is None:
		joint_names = [
			"head","neck","r_shoulder","r_elbow","r_wrist",
			"l_shoulder","l_elbow","l_wrist","pelvis","r_hip",
			"r_knee","r_ankle","l_hip","l_knee","l_ankle","spine","thorax"
		]
	else:
		joint_names = json.loads(Path(args.joint_spec).read_text())

	trial_files = sorted(list(raw_dir.rglob(args.glob)))
	if len(trial_files) == 0:
		print('No raw trial files found under', raw_dir)
		return

	metadata_rows = []
	for trial in trial_files:
		points3d = load_trial_csv(trial, joint_names)
		T, J, _ = points3d.shape
		base_id = trial.stem
		pd_label = 1 if 'pd' in base_id.lower() else 0
		severity_label = 0
		for v in range(args.views):
			yaw = (360.0 / args.views) * v
			uv = project_points(points3d, yaw_deg=yaw, pitch_deg=0.0)
			sample_id = f"{base_id}_v{v}"
			write_skeleton_npy(data_dir / 'skeleton' / 'train' / f'{sample_id}.npy', uv)
			synthesize_silhouettes(data_dir / 'silhouette' / 'train' / sample_id)
			metadata_rows.append({
				'sample_id': sample_id,
				'pd_label': pd_label,
				'severity_label': severity_label,
				'view_id': v,
			})

	# Simple split by sample ids
	ids = [r['sample_id'] for r in metadata_rows]
	train_ids, val_ids = split_train_val(ids, val_ratio=0.2)
	train_rows = [r for r in metadata_rows if r['sample_id'] in train_ids]
	val_rows = [r for r in metadata_rows if r['sample_id'] in val_ids]

	import pandas as pd
	pd.DataFrame(train_rows).to_csv(data_dir / 'labels_train.csv', index=False)
	pd.DataFrame(val_rows).to_csv(data_dir / 'labels_val.csv', index=False)
	print('Converted trials:', len(trial_files), 'views per trial:', args.views)
	print('Wrote labels:', data_dir / 'labels_train.csv', data_dir / 'labels_val.csv')


if __name__ == '__main__':
	main()

