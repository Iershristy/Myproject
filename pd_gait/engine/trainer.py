from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from pd_gait.datasets.multimodal import build_loaders
from pd_gait.models.model import MultimodalGaitNet
from pd_gait.utils.metrics import classification_metrics
from pd_gait.engine.grad_cam import GradCAM


class Trainer:
	def __init__(self, cfg: Dict[str, Any]):
		self.cfg = cfg
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.train_loader, self.val_loader = build_loaders(cfg)
		self.model = MultimodalGaitNet(cfg).to(self.device)

		params = [p for p in self.model.parameters() if p.requires_grad]
		self.optimizer = AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
		self.scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True)))

		self.crit_pd = nn.CrossEntropyLoss()
		self.crit_sev = nn.CrossEntropyLoss()

		self.save_dir = Path(cfg["eval"]["save_dir"])
		self.save_dir.mkdir(parents=True, exist_ok=True)

		self.enable_gradcam = bool(cfg["eval"].get("grad_cam", True))

	def fit(self) -> None:
		best_val = 0.0
		for epoch in range(1, int(self.cfg["train"]["max_epochs"]) + 1):
			train_loss, train_pd_acc = self._run_epoch(epoch, self.train_loader, train=True)
			val_loss, val_pd_acc = self._run_epoch(epoch, self.val_loader, train=False)

			if val_pd_acc > best_val:
				best_val = val_pd_acc
				torch.save(self.model.state_dict(), self.save_dir / "best.pt")

			print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_pd_acc:.3f} "
			      f"val_loss={val_loss:.4f} val_acc={val_pd_acc:.3f}")

		torch.save(self.model.state_dict(), self.save_dir / "last.pt")

	def _forward_compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		out = self.model(batch)
		pd_logits = out["pd_logits"]
		sev_logits = out["severity_logits"]

		pd_targets = batch["pd_label"]
		sev_targets = batch["severity_label"]
		sev_mask = (sev_targets >= 0).float()
		sev_loss = torch.tensor(0.0, device=self.device)
		if sev_mask.sum() > 0:
			sev_loss = self.crit_sev(sev_logits[sev_mask > 0], sev_targets[sev_mask > 0])

		pd_loss = self.crit_pd(pd_logits, pd_targets)

		loss = self.cfg["train"]["pd_loss_weight"] * pd_loss + self.cfg["train"]["severity_loss_weight"] * sev_loss
		return loss, out

	def _prepare_batch(self, data: Dict[str, Any]) -> Dict[str, Any]:
		batch = {}
		for k, v in data.items():
			if isinstance(v, torch.Tensor):
				batch[k] = v.to(self.device, non_blocking=True)
			else:
				batch[k] = v
		return batch

	def _run_epoch(self, epoch: int, loader: DataLoader, train: bool) -> Tuple[float, float]:
		self.model.train(train)
		total_loss = 0.0
		all_pd_true = []
		all_pd_pred = []

		pbar = tqdm(loader, desc=("Train" if train else "Val"), leave=False)
		step = 0

		for data in pbar:
			batch = self._prepare_batch({
				"skeleton": data.get("skeleton", torch.empty(0)),
				"skeleton_mask": data.get("skeleton_mask", torch.empty(0)),
				"silhouette": data.get("silhouette", torch.empty(0)),
				"silhouette_mask": data.get("silhouette_mask", torch.empty(0)),
				"imu": data.get("imu", torch.empty(0)),
				"imu_mask": data.get("imu_mask", torch.empty(0)),
				"pd_label": data["pd_label"].long(),
				"severity_label": data["severity_label"].long(),
			})

			with autocast(enabled=bool(self.cfg["train"].get("mixed_precision", True))):
				loss, out = self._forward_compute_loss(batch)

			if train:
				self.scaler.scale(loss).backward()
				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.optimizer.zero_grad(set_to_none=True)

			total_loss += float(loss.detach().cpu().item())

			pd_pred = out["pd_logits"].argmax(dim=-1).detach().cpu().numpy()
			pd_true = batch["pd_label"].detach().cpu().numpy()
			all_pd_pred.append(pd_pred)
			all_pd_true.append(pd_true)

			step += 1
			if train and step % int(self.cfg["train"]["log_interval"]) == 0:
				pbar.set_postfix(loss=float(loss.detach().cpu().item()))

		all_pd_true = np.concatenate(all_pd_true) if len(all_pd_true) else np.array([])
		all_pd_pred = np.concatenate(all_pd_pred) if len(all_pd_pred) else np.array([])
		pd_acc, _ = classification_metrics(all_pd_true, all_pd_pred) if all_pd_true.size else (0.0, 0.0)

		avg_loss = total_loss / max(1, step)

		if not train and self.enable_gradcam:
			try:
				self._gradcam_snapshot(batch, out)
			except Exception as e:
				print(f"[GradCAM] skipped due to error: {e}")

		return avg_loss, pd_acc

	def _gradcam_snapshot(self, batch: Dict[str, Any], out: Dict[str, torch.Tensor]) -> None:
		if batch["silhouette"].numel() == 0:
			return
		b, t, _, h, w = batch["silhouette"].shape
		target = out["pd_logits"].max(dim=-1).values.mean()
		cam = GradCAM(self.model)
		heatmaps = cam.generate(target, b=b, t=t, h=h, w=w)
		save_dir = self.save_dir / "gradcam"
		save_dir.mkdir(parents=True, exist_ok=True)
		import numpy as np
		np.save(save_dir / "val_cam.npy", heatmaps)

