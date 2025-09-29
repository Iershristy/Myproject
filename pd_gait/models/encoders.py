from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
	def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p),
			nn.BatchNorm1d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=p),
			nn.BatchNorm1d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class SkeletonPartEncoder(nn.Module):
	"""
	Part-based skeleton encoder inspired by GaitPart, robust to occlusion.
	Input: coords [B,T,J,C], mask [B,T,J]
	Output: embedding [B,H], part_attn [B,P]
	"""
	def __init__(self, num_joints: int = 17, hidden_dim: int = 128, num_layers: int = 3):
		super().__init__()
		self.num_joints = num_joints
		self.hidden_dim = hidden_dim

		self.parts: List[List[int]] = [
			[5, 6, 11, 12, 0],
			[5, 7, 9],
			[6, 8, 10],
			[11, 13, 15],
			[12, 14, 16],
		]
		in_feat = 4
		self.part_blocks = nn.ModuleList()
		ch = in_feat
		for _ in range(num_layers):
			self.part_blocks.append(TemporalConvBlock(ch, hidden_dim))
			ch = hidden_dim

		self.part_attention = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, 1),
		)

		self.out_proj = nn.Linear(hidden_dim, hidden_dim)

	def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		b, t, j, c = coords.shape
		xy = coords[..., :2]
		vel = torch.zeros_like(xy)
		vel[:, 1:] = xy[:, 1:] - xy[:, :-1]
		feat = torch.cat([xy, vel], dim=-1)

		part_feats = []
		part_valid = []
		for joint_ids in self.parts:
			joint_ids = [idx for idx in joint_ids if idx < j]
			if len(joint_ids) == 0:
				part_feats.append(torch.zeros(b, t, 4, device=coords.device))
				part_valid.append(torch.zeros(b, t, device=coords.device))
				continue
			sub = feat[:, :, joint_ids, :]
			sub_mask = mask[:, :, joint_ids]
			denom = sub_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
			pooled = (sub * sub_mask[..., None]).sum(dim=-2) / denom
			part_feats.append(pooled)
			part_valid.append((sub_mask.sum(dim=-1) > 0).float())

		part_stack = torch.stack(part_feats, dim=2)
		valid_stack = torch.stack(part_valid, dim=2)

		p = part_stack.shape[2]
		part_embed = []
		attn_logits = []
		for pi in range(p):
			x = part_stack[:, :, pi, :]
			x = x.transpose(1, 2)
			for block in self.part_blocks:
				x = block(x)
			x_avg = x.mean(dim=-1)
			x_max = x.amax(dim=-1)
			x_pool = (x_avg + x_max) * 0.5
			part_embed.append(x_pool)
			attn_logits.append(self.part_attention(x_pool))

		part_embed = torch.stack(part_embed, dim=1)
		attn_logits = torch.cat(attn_logits, dim=-1)
		part_valid_mask = (valid_stack.sum(dim=1) > 0).float()
		attn_logits = attn_logits.masked_fill(part_valid_mask <= 0, float("-inf"))
		part_attn = torch.softmax(attn_logits, dim=-1)

		embed = (part_embed * part_attn.unsqueeze(-1)).sum(dim=1)
		embed = self.out_proj(embed)
		return embed, part_attn


class SilhouetteHPPEncoder(nn.Module):
	def __init__(self, base_channels: int = 32, hpp_bins: List[int] = [1, 2, 4]):
		super().__init__()
		c = base_channels
		self.backbone = nn.Sequential(
			nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
		)
		self.hpp_bins = hpp_bins
		self.proj = nn.Linear(sum([c * 4 * b for b in hpp_bins]), c * 4)

	def _hpp(self, feat: torch.Tensor) -> torch.Tensor:
		b, c, h, w = feat.shape
		pooled_list = []
		for bins in self.hpp_bins:
			pooled = F.adaptive_avg_pool2d(feat, (bins, 1))
			pooled = pooled.view(b, c * bins)
			pooled_list.append(pooled)
		out = torch.cat(pooled_list, dim=-1)
		return out

	def forward(self, imgs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		b, t, _, h, w = imgs.shape
		x = imgs.view(b * t, 1, h, w)
		fmap = self.backbone(x)
		frame_feats = self._hpp(fmap)
		frame_feats = self.proj(frame_feats)
		frame_feats = frame_feats.view(b, t, -1)

		frame_mask = mask.float().unsqueeze(-1)
		masked = frame_feats * frame_mask
		denom = frame_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
		feat_mean = masked.sum(dim=1) / denom
		feat_max = (frame_feats + (frame_mask - 1.0) * 1e6).amax(dim=1)
		embed = 0.5 * (feat_mean + feat_max)

		attn_logits = (frame_feats @ embed.unsqueeze(-1)).squeeze(-1) / (frame_feats.shape[-1] ** 0.5)
		attn_logits = attn_logits.masked_fill(mask <= 0, float("-inf"))
		attn = torch.softmax(attn_logits, dim=-1)
		return embed, attn


class IMUEncoder(nn.Module):
	def __init__(self, sensors: int = 6, hidden_dim: int = 64):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv1d(sensors, hidden_dim, 5, padding=2),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(inplace=True),
			nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(inplace=True),
		)
		self.proj = nn.Linear(hidden_dim, hidden_dim)

	def forward(self, imu: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		x = imu.transpose(1, 2)
		x = self.net(x)
		x_avg = (x.mean(dim=-1) + x.amax(dim=-1)) * 0.5
		x = self.proj(x_avg)
		return x

