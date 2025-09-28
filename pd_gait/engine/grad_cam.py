from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
	"""
	Grad-CAM for SilhouetteHPPEncoder backbone's last conv layer.
	"""
	def __init__(self, model):
		self.model = model
		self._fmap = None
		self._grad = None
		last_conv = None
		for m in model.sil_enc.backbone.modules():
			if isinstance(m, torch.nn.Conv2d):
				last_conv = m
		assert last_conv is not None
		last_conv.register_forward_hook(self._save_fmap)
		last_conv.register_full_backward_hook(self._save_grad)

	def _save_fmap(self, module, inp, out):
		self._fmap = out.detach()

	def _save_grad(self, module, grad_in, grad_out):
		self._grad = grad_out[0].detach()

	def generate(self, loss: torch.Tensor, b: int, t: int, h: int, w: int) -> np.ndarray:
		self.model.zero_grad(set_to_none=True)
		loss.backward(retain_graph=True)

		fmap = self._fmap
		grad = self._grad
		weights = grad.mean(dim=(2, 3), keepdim=True)
		cam = (weights * fmap).sum(dim=1, keepdim=True)
		cam = F.relu(cam)
		cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-6)
		cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
		cam = cam.view(b, t, h, w)
		return cam.detach().cpu().numpy()

