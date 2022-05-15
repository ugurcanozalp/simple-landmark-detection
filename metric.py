
from typing import List

import numpy as np
import torch
from torchmetrics import Metric


class LandmarkDetectionMetric(Metric):
	def __init__(self, dist_sync_on_step: bool = False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		self.add_state("landmark_l1_errors", default=[], dist_reduce_fx=None)
		self.add_state("landmark_l2_errors", default=[], dist_reduce_fx=None)
		self.add_state("visible_preds", default=[], dist_reduce_fx=None)
		self.add_state("visible_targets", default=[], dist_reduce_fx=None)

	def update(self, preds: torch.Tensor, targets: torch.Tensor):
		landmark_preds = preds[:,:,:-1].cpu().numpy()
		visible_preds = preds[:,:,-1].cpu().numpy()
		landmark_targets = targets[:,:,:-1].cpu().numpy()
		visible_targets = targets[:,:,-1].cpu().numpy()
		landmark_l1 = np.linalg.norm(landmark_preds - landmark_targets, axis=-1, ord=1)
		landmark_l2 = np.linalg.norm(landmark_preds - landmark_targets, axis=-1, ord=2)
		
		self.landmark_l1_errors.append(landmark_l1)
		self.landmark_l2_errors.append(landmark_l2)
		self.visible_preds.append(visible_preds)
		self.visible_targets.append(visible_targets)

	def compute(self):
		landmark_l1_errors = np.concatenate(self.landmark_l1_errors, axis=0)
		landmark_l2_errors = np.concatenate(self.landmark_l2_errors, axis=0)
		visible_preds = np.concatenate(self.visible_preds, axis=0)
		visible_targets = np.concatenate(self.visible_targets, axis=0)

		l1_errors = (landmark_l1_errors*visible_targets).sum()/(visible_targets.sum()+1e-6)
		l2_errors = (landmark_l2_errors*visible_targets).sum()/(visible_targets.sum()+1e-6)
		visible_accuracy = ((visible_preds>0.5) == (visible_targets>0.5)).mean()
		return {
			"l1_errors": l1_errors,
			"l2_errors": l2_errors,
			"visible_accuracy": visible_accuracy,
		}
