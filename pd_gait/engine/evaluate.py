from typing import Dict, Any
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pd_gait.utils.config import load_config
from pd_gait.datasets.multimodal import build_loaders
from pd_gait.models.model import MultimodalGaitNet
from pd_gait.utils.metrics import full_metrics


@torch.no_grad()
def evaluate(cfg: Dict[str, Any], ckpt_path: str = None) -> Dict[str, Any]:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	_, val_loader = build_loaders(cfg)
	model = MultimodalGaitNet(cfg).to(device)
	if ckpt_path is not None and Path(ckpt_path).exists():
		model.load_state_dict(torch.load(ckpt_path, map_location=device))
	model.eval()

	all_pd_true, all_pd_pred = [], []
	all_sev_true, all_sev_pred = [], []

	for data in tqdm(val_loader, desc="Eval", leave=False):
		batch = {}
		for k, v in data.items():
			if isinstance(v, torch.Tensor):
				batch[k] = v.to(device)
			else:
				batch[k] = v
		out = model(batch)
		pd_pred = out["pd_logits"].argmax(dim=-1).cpu().numpy()
		sev_pred = out["severity_logits"].argmax(dim=-1).cpu().numpy()
		pd_true = batch["pd_label"].cpu().numpy()
		sev_true = batch["severity_label"].cpu().numpy()
		# keep sev where label valid
		valid_mask = (sev_true >= 0)
		all_pd_true.append(pd_true)
		all_pd_pred.append(pd_pred)
		if valid_mask.any():
			all_sev_true.append(sev_true[valid_mask])
			all_sev_pred.append(sev_pred[valid_mask])

	all_pd_true = np.concatenate(all_pd_true) if len(all_pd_true) else np.array([])
	all_pd_pred = np.concatenate(all_pd_pred) if len(all_pd_pred) else np.array([])
	all_sev_true = np.concatenate(all_sev_true) if len(all_sev_true) else np.array([])
	all_sev_pred = np.concatenate(all_sev_pred) if len(all_sev_pred) else np.array([])

	results: Dict[str, Any] = {
		"pd": full_metrics(all_pd_true, all_pd_pred) if all_pd_true.size else {},
		"severity": full_metrics(all_sev_true, all_sev_pred) if all_sev_true.size else {},
	}
	return results


def main():
	import argparse
	parser = argparse.ArgumentParser(description="Evaluate model on validation set")
	parser.add_argument("--config", type=str, default="configs/default.yaml")
	parser.add_argument("--ckpt", type=str, default="experiments/default/best.pt")
	parser.add_argument("--out", type=str, default="experiments/default/metrics.json")
	args = parser.parse_args()

	cfg = load_config(args.config)
	res = evaluate(cfg, ckpt_path=args.ckpt)
	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(res, f, indent=2)
	print("Saved:", args.out)


if __name__ == "__main__":
	main()

