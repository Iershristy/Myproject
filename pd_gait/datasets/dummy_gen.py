from pathlib import Path
import numpy as np
import cv2
import pandas as pd


def make_dummy(root: Path, num_train: int = 32, num_val: int = 16, t: int = 60, j: int = 17, s: int = 6, img_hw=(128, 88)) -> None:
	root.mkdir(parents=True, exist_ok=True)
	for split, n in [("train", num_train), ("val", num_val)]:
		skel_dir = root / "skeleton" / split
		sil_dir = root / "silhouette" / split
		imu_dir = root / "imu" / split
		skel_dir.mkdir(parents=True, exist_ok=True)
		sil_dir.mkdir(parents=True, exist_ok=True)
		imu_dir.mkdir(parents=True, exist_ok=True)

		rows = []
		for i in range(n):
			sid = f"dummy_{split}_{i}"
			# skeleton: simple sinusoidal trajectories with noise
			time = np.linspace(0, 2 * np.pi, t)
			skel = np.zeros((t, j, 3), dtype=np.float32)
			skel[:, :, 0] = (np.sin(time)[:, None] * 5.0) + np.random.randn(t, j) * 0.05
			skel[:, :, 1] = (np.cos(time)[:, None] * 2.0) + np.random.randn(t, j) * 0.05
			skel[:, :, 2] = 1.0
			np.save(skel_dir / f"{sid}.npy", skel)

			# silhouette: simple moving blob
			sdir = sil_dir / sid
			sdir.mkdir(parents=True, exist_ok=True)
			for ti in range(t):
				img = np.zeros(img_hw, dtype=np.uint8)
				y = int((np.sin(time[ti]) * 0.3 + 0.5) * (img_hw[0] - 30))
				x = int((np.cos(time[ti]) * 0.3 + 0.5) * (img_hw[1] - 20))
				cv2.rectangle(img, (x, y), (x + 20, y + 30), 255, -1)
				cv2.imwrite(str(sdir / f"frame_{ti:04d}.png"), img)

			# imu: noisy sinusoids
			imu = np.stack([np.sin(time + k * 0.1) for k in range(s)], axis=1).astype(np.float32)
			imu += 0.05 * np.random.randn(t, s).astype(np.float32)
			np.save(imu_dir / f"{sid}.npy", imu)

			pd_label = int(np.random.rand() < 0.5)
			severity_label = int(np.random.randint(0, 3))
			view_id = int(np.random.randint(0, 8))
			rows.append({"sample_id": sid, "pd_label": pd_label, "severity_label": severity_label, "view_id": view_id})

		csv_path = root / f"labels_{split}.csv"
		pd.DataFrame(rows).to_csv(csv_path, index=False)


if __name__ == "__main__":
	make_dummy(Path("data"))

