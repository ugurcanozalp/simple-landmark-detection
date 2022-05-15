from argparse import ArgumentParser
import os
import numpy as np
from skimage.transform import resize
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models import model_map
from metric import LandmarkDetectionMetric

class LandmarkDetector(pl.LightningModule):

	def __init__(self,
		arch = "resnet",
		configuration = "resnet18",
		pretrained = True, 
		learning_rate: float = 1e-4,
		weight_decay: float = 0,
		num_landmarks: int = 20, 
		image_height = 96, 
		image_width = 128,
		*args,
		**kwargs
		):
		super(LandmarkDetector,self).__init__()
		self.save_hyperparameters("learning_rate", "weight_decay")
		self.num_landmarks = num_landmarks
		self.image_height = image_height 
		self.image_width = image_width
		self.model = model_map[arch](configuration, num_landmarks, image_height, image_width, pretrained=pretrained)
		self.metrics = {"basic": LandmarkDetectionMetric()}

	def forward(self, image):
		return self.model(image)

	@staticmethod
	def wing_loss(landmark_pred, landmark_target, w=10.0, eps=2.0):
		err = torch.abs(landmark_pred - landmark_target)
		C = w - w * np.log(1 + w / eps)
		return torch.mean(torch.where(err < w, w * torch.log(1 + err / eps), err - C))

	@staticmethod
	def loss_fcn(preds, targets):
		"""
		preds: (batch, num_landmark, 3)
		targets: (batch, num_landmark, 3)
		"""
		visible_pred = preds[:,:,-1:] # (batch, num_landmark)
		visible_target = targets[:,:,-1:] # (batch, num_landmark)
		landmark_pred = preds[:,:,:-1] * visible_target # (batch, num_landmark, 2)
		landmark_target = targets[:,:,:-1] * visible_target # (batch, num_landmark, 2)
		landmark_loss = nn.functional.smooth_l1_loss(landmark_pred, landmark_target, beta=0.02)
		#landmark_loss = LandmarkDetector.wing_loss(landmark_pred, landmark_target)
		visible_loss = nn.functional.binary_cross_entropy(visible_pred, visible_target)
		return landmark_loss + 0.01*visible_loss

	def configure_optimizers(self):
		optimizer_grouped_parameters = [
			{
				"params": self.model.parameters(),
				"weight_decay_rate": 0,
				"lr": self.hparams.learning_rate,
			}
		]
		optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
		return optimizer

	def training_step(self, batch, batch_idx):
		inputs, targets = batch
		preds = self.forward(inputs)
		loss = self.loss_fcn(preds, targets)
		tensorboard_logs = {'train_batch_loss': loss}
		for metric, value in tensorboard_logs.items():
			self.log(metric, value, prog_bar=False)
		return {"loss": loss}

	def validation_step(self, batch, batch_idx):
		inputs, targets = batch
		preds = self.forward(inputs)
		loss = self.loss_fcn(preds, targets)
		self.metrics["basic"].update(preds, targets)
		return {"loss": loss}

	def test_step(self, batch, batch_idx):
		inputs, targets = batch
		preds = self.forward(inputs)
		loss = self.loss_fcn(preds, targets)
		self.metrics["basic"].update(preds, targets)
		return {"loss": loss}

	def validation_epoch_end(self, outputs):
		avg_loss = sum(output["loss"].cpu().item() for output in outputs)/len(outputs)
		self.log("avg_val_loss", avg_loss)
		metrics = self.metrics["basic"].compute()
		for name, value in metrics.items():
			self.log(name, value)

	def test_epoch_end(self, outputs):
		avg_loss = sum(output["loss"].cpu().item() for output in outputs)/len(outputs)
		self.log("avg_test_loss", avg_loss)
		metrics = self.metrics["basic"].compute()
		for name, value in metrics.items():
			self.log(name, value)

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--arch', type=str, default='convnext')
		parser.add_argument('--configuration', type=str, default='convnext_tiny')
		parser.add_argument('--learning_rate', type=float, default=1e-4)
		parser.add_argument('--weight_decay', type=float, default=0)
		parser.add_argument('--num_landmarks', type=int, default=20)
		return parser

	@torch.no_grad()
	def predict(self, image, skeleton):
		image = resize(image, (self.image_height, self.image_width))
		if image.dtype == "uint8":
			image = image/255.0
		if len(image.shape) == 2:
			image = np.expand_dims(image, axis=-1).repeat(3, -1)

		image_tensor = torch.tensor(image, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
		output = self.forward(image_tensor)
		output_numpy = output.squeeze(0).cpu().numpy()
		#print(output_numpy)
		import matplotlib.pyplot as plt 
		plt.imshow(image)
		x, y, visible = self.image_width*output_numpy[:,0], self.image_height*output_numpy[:,1], output_numpy[:,2]
		for i, vis in enumerate(visible):
			text = str(i) if vis>0.5 else "x"
			plt.annotate("o", (x[i], y[i]), color='green', fontsize='large')
			plt.annotate(text, (x[i], y[i]), color='red', fontsize='large')
			for skeleton_pair in skeleton:
				if skeleton_pair[0] == i:
					j = skeleton_pair[1]
					plt.plot([x[i], x[j]], [y[i], y[j]], 'green', linestyle="-")
		plt.show()

if __name__=="__main__":
	import imageio
	module = LandmarkDetector(arch="convnext", configuration="convnext_tiny", num_landmarks=20)
	#image = imageio.imread("/home/ugurcan/ai/cv/simple-landmark-detection/data/animalpose/images/co78.jpeg")
	image = imageio.imread("/home/ugurcan/ai/cv/simple-landmark-detection/data/sample_cow.jpg")
	skeleton = [[0,1],[0,2],[1,2],[0,3],[1,4],[2,17],[18,19],[5,9],[6,10],[7,11],[8,12],[9,13],[10,14],[11,15],[12,16]]
	sd = torch.load("checkpoints/convnext_tiny.pt-v1.ckpt", map_location="cpu")["state_dict"]
	module.load_state_dict(sd)
	module.eval()
	module.predict(image, skeleton)
