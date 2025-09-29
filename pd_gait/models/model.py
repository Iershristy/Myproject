from typing import Dict, Any
import torch
import torch.nn as nn

from pd_gait.models.encoders import SkeletonPartEncoder, SilhouetteHPPEncoder, IMUEncoder
from pd_gait.models.fusion import AttentionFusion


class MultimodalGaitNet(nn.Module):
	def __init__(self, cfg: Dict[str, Any]):
		super().__init__()
		mcfg = cfg["model"]
		self.use_imu = "imu" in cfg["data"]["modalities"]

		self.skel_enc = SkeletonPartEncoder(
			num_joints=cfg["data"]["skeleton"]["joints"],
			hidden_dim=mcfg["skeleton_encoder"]["hidden_dim"],
			num_layers=mcfg["skeleton_encoder"]["num_layers"],
		)
		self.sil_enc = SilhouetteHPPEncoder(
			base_channels=mcfg["silhouette_encoder"]["base_channels"],
			hpp_bins=mcfg["silhouette_encoder"]["hpp_bins"],
		)
		self.imu_enc = None
		if self.use_imu:
			self.imu_enc = IMUEncoder(
				sensors=cfg["data"]["imu"]["sensors"],
				hidden_dim=mcfg["imu_encoder"]["hidden_dim"],
			)

		h_fuse = mcfg["fusion"]["hidden_dim"]
		h_skel = mcfg["skeleton_encoder"]["hidden_dim"]
		h_sil = mcfg["silhouette_encoder"]["base_channels"] * 4
		h_imu = mcfg["imu_encoder"]["hidden_dim"] if self.use_imu else 0
		self.proj_skel = nn.Linear(h_skel, h_fuse)
		self.proj_sil = nn.Linear(h_sil, h_fuse)
		self.proj_imu = nn.Linear(h_imu, h_fuse) if self.use_imu else None

		self.fusion = AttentionFusion(embed_dim=h_fuse)

		d = h_fuse
		self.pd_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(d, mcfg["heads"]["pd_classes"]))
		self.sev_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(d, mcfg["heads"]["severity_classes"]))

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		out: Dict[str, torch.Tensor] = {}

		skel_embed, skel_part_attn = self.skel_enc(batch["skeleton"], batch["skeleton_mask"])  # [B,H], [B,P]
		skel_embed = self.proj_skel(skel_embed)

		sil_embed, sil_frame_attn = self.sil_enc(batch["silhouette"], batch["silhouette_mask"])  # [B,H], [B,T]
		sil_embed = self.proj_sil(sil_embed)

		embeds = {"skeleton": skel_embed, "silhouette": sil_embed}
		present = {
			"skeleton": (batch["skeleton_mask"].sum(dim=[1, 2]) > 0).float(),
			"silhouette": (batch["silhouette_mask"].sum(dim=1) > 0).float(),
		}

		if self.imu_enc is not None and "imu" in batch:
			imu_embed = self.imu_enc(batch["imu"], batch["imu_mask"])  # [B,H]
			imu_embed = self.proj_imu(imu_embed)
			embeds["imu"] = imu_embed
			present["imu"] = (batch["imu_mask"].sum(dim=1) > 0).float()

		fused, weights, weight_map = self.fusion(embeds, present)
		pd_logits = self.pd_head(fused)
		sev_logits = self.sev_head(fused)

		out.update({
			"pd_logits": pd_logits,
			"severity_logits": sev_logits,
			"fused": fused,
			"skel_part_attn": skel_part_attn,
			"sil_frame_attn": sil_frame_attn,
		})
		for k, v in weight_map.items():
			out[f"w_{k}"] = v
		return out

