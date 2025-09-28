from typing import Dict, Tuple
import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
	"""
	Attention-based fusion with modality reliability gating and missing-modality masks.
	"""
	def __init__(self, embed_dim: int = 256):
		super().__init__()
		self.key_proj = nn.Linear(embed_dim, embed_dim)
		self.query = nn.Parameter(torch.randn(embed_dim))
		self.gate = nn.Linear(embed_dim, 1)

	def forward(self, embeds: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
		keys = []
		gates = []
		masks = []
		names = list(embeds.keys())
		for name in names:
			e = embeds[name]
			k = self.key_proj(e)
			g = torch.sigmoid(self.gate(e)).squeeze(-1)
			keys.append(k)
			gates.append(g)
			masks.append(present[name].float())

		keys = torch.stack(keys, dim=1)
		gates = torch.stack(gates, dim=1)
		masks = torch.stack(masks, dim=1)

		q = self.query.unsqueeze(0).unsqueeze(1)
		attn_logits = (keys * q).sum(dim=-1) / (keys.shape[-1] ** 0.5)
		attn_logits = attn_logits + (gates.log() - (1 - gates + 1e-6).log())
		attn_logits = attn_logits.masked_fill(masks <= 0, float("-inf"))
		weights = torch.softmax(attn_logits, dim=-1)

		fused = (keys * weights.unsqueeze(-1)).sum(dim=1)
		return fused, weights, {name: weights[:, i] for i, name in enumerate(names)}

