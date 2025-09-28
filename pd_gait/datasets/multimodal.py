from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import glob
import random

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


def _uniform_sample_indices(length: int, num: int) -> List[int]:
	if length <= 0:
		return [0] * num
	if num <= 1:
		return [0]
	return list(np.linspace(0, max(0, length - 1), num=num).astype(int))


def _load_skeleton_npy(path: Path) -> Optional[np.ndarray]:
	if not path.exists():
		return None
	arr = np.load(str(path))
	return arr


def _load_imu_npy(path: Path) -> Optional[np.ndarray]:
	if not path.exists():
		return None
	arr = np.load(str(path))
	return arr


def _load_silhouette_frames(dir_path: Path) -> Optional[List[Path]]:
	if not dir_path.exists():
		return None
	frames = sorted(glob.glob(str(dir_path / "*.png")))
	if len(frames) == 0:
		frames = sorted(glob.glob(str(dir_path / "*.jpg")))
	return [Path(f) for f in frames]


def _read_gray_resize(path: Path, size_hw: Tuple[int, int]) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if img is None:
		img = np.zeros(size_hw, dtype=np.uint8)
\timg = cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA)
	img = img.astype(np.float32) / 255.0
	return img


class MultimodalGaitDataset(Dataset):
	def __init__(self, cfg: Dict[str, Any], split: str = "train"):
		self.cfg = cfg
		self.split = split
		self.root = Path(cfg["data"]["root"])

		self.num_frames = int(cfg["data"]["sequence"]["num_frames"])
		self.image_size = tuple(cfg["data"]["silhouette"]["image_size"])
		self.modalities = cfg["data"]["modalities"]
		self.dummy = bool(cfg["data"].get("dummy_data", False))

		self.occl = cfg["data"].get("occlusion", {})
		self.joint_drop_prob = float(self.occl.get("joint_drop_prob", 0.0))
		self.frame_drop_prob = float(self.occl.get("frame_drop_prob", 0.0))
		self.silhouette_erase_prob = float(self.occl.get("silhouette_erase_prob", 0.0))

		paths = cfg["data"][split]
		self.skeleton_dir = Path(paths["skeleton_dir"])
		self.silhouette_dir = Path(paths["silhouette_dir"])
		self.imu_dir = Path(paths["imu_dir"])
		labels_csv = Path(paths["labels_csv"])

		if self.dummy:
			self.samples = pd.DataFrame({
				"sample_id": [f"dummy_{i}" for i in range(64)],
				"pd_label": np.random.randint(0, 2, size=64),
				"severity_label": np.random.randint(0, 3, size=64),
				"view_id": np.random.randint(0, 8, size=64),
			})
		else:
			self.samples = pd.read_csv(labels_csv)

	def __len__(self) -> int:
		return len(self.samples)

	def _skeleton_part(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
		path = self.skeleton_dir / f"{sample_id}.npy"
		arr = _load_skeleton_npy(path)

		if arr is None:
			t = self.num_frames
			j = int(self.cfg["data"]["skeleton"]["joints"])
			c = 3 if self.cfg["data"]["skeleton"].get("use_confidence", True) else 2
			arr = np.zeros((t, j, c), dtype=np.float32)
			mask = np.zeros((t, j), dtype=np.float32)
		else:
			t, j, c = arr.shape
			idx = _uniform_sample_indices(t, self.num_frames)
			arr = arr[idx]
			mask = np.ones((self.num_frames, j), dtype=np.float32)

		midhip_idx = 11
		if arr.shape[1] > midhip_idx and arr.shape[2] >= 2:
			center = arr[:, midhip_idx:midhip_idx + 1, :2]
			arr[:, :, :2] = arr[:, :, :2] - center

		if self.split == "train":
			if self.joint_drop_prob > 0:
				drop = np.random.rand(*mask.shape) < self.joint_drop_prob
				mask[drop] = 0.0
			if self.frame_drop_prob > 0:
				fdrop = (np.random.rand(mask.shape[0]) < self.frame_drop_prob).astype(np.float32)
				mask[fdrop > 0, :] = 0.0

		return torch.from_numpy(arr).float(), torch.from_numpy(mask).float()

	def _silhouette_part(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
		frames = _load_silhouette_frames(self.silhouette_dir / sample_id)
		if frames is None or len(frames) == 0:
			imgs = np.zeros((self.num_frames, 1, self.image_size[0], self.image_size[1]), dtype=np.float32)
			mask = np.zeros((self.num_frames,), dtype=np.float32)
			return torch.from_numpy(imgs), torch.from_numpy(mask)

		idx = _uniform_sample_indices(len(frames), self.num_frames)
		imgs = []
		for i in idx:
			gray = _read_gray_resize(frames[i], self.image_size)
			imgs.append(gray[None, ...])
		imgs = np.stack(imgs, axis=0)
		mask = np.ones((self.num_frames,), dtype=np.float32)

		if self.split == "train" and self.silhouette_erase_prob > 0:
			t, _, h, w = imgs.shape
			for ti in range(t):
				if random.random() < self.silhouette_erase_prob:
					rh = max(4, int(h * 0.1))
					rw = max(4, int(w * 0.1))
					y0 = random.randint(0, max(1, h - rh))
					x0 = random.randint(0, max(1, w - rw))
					imgs[ti, :, y0:y0 + rh, x0:x0 + rw] = 0.0

		return torch.from_numpy(imgs).float(), torch.from_numpy(mask).float()

	def _imu_part(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
		path = self.imu_dir / f"{sample_id}.npy"
		arr = _load_imu_npy(path)
		sensors = int(self.cfg["data"]["imu"]["sensors"])

		if arr is None:
			t = self.num_frames
			arr = np.zeros((t, sensors), dtype=np.float32)
			mask = np.zeros((t,), dtype=np.float32)
		else:
			t = arr.shape[0]
			if arr.shape[1] != sensors:
				out = np.zeros((t, sensors), dtype=np.float32)
				s = min(sensors, arr.shape[1])
				out[:, :s] = arr[:, :s]
				arr = out
			idx = _uniform_sample_indices(t, self.num_frames)
			arr = arr[idx]
			mask = np.ones((self.num_frames,), dtype=np.float32)

		return torch.from_numpy(arr).float(), torch.from_numpy(mask).float()

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		row = self.samples.iloc[idx]
		sample_id = str(row["sample_id"])
		pd_label = int(row["pd_label"])
		severity_label = int(row.get("severity_label", -1))
		view_id = int(row.get("view_id", 0))

		out: Dict[str, Any] = {
			"sample_id": sample_id,
			"pd_label": pd_label,
			"severity_label": severity_label,
			"view_id": view_id,
		}

		if "skeleton" in self.modalities:
			skel, skel_mask = self._skeleton_part(sample_id)
			out["skeleton"] = skel
			out["skeleton_mask"] = skel_mask

		if "silhouette" in self.modalities:
			sil, sil_mask = self._silhouette_part(sample_id)
			out["silhouette"] = sil
			out["silhouette_mask"] = sil_mask

		if "imu" in self.modalities:
			imu, imu_mask = self._imu_part(sample_id)
			out["imu"] = imu
			out["imu_mask"] = imu_mask

		return out


def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
	train_ds = MultimodalGaitDataset(cfg, split="train")
	val_ds = MultimodalGaitDataset(cfg, split="val")
	bs = int(cfg["train"]["batch_size"])
	nw = int(cfg["data"]["num_workers"])
	train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
	return train_loader, val_loader

