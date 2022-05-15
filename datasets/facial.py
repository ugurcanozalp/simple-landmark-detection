import os
import numpy as np
import math
import glob
import pandas as pd
import imageio
import torch
from skimage.transform import resize
import random
from torch.utils.data import Dataset

# Dataset for following dataset, but data format is a bit changed..
# https://www.kaggle.com/sharwon/facialkeypointsdetectionimg

def image_to_tensor(img):
	if img.dtype == "uint8":
		img = img/255.0
	return torch.tensor(img, dtype=torch.float32).permute(2,0,1)

class FacialLandmarkDataset(Dataset):

	def __init__(self, path, height=96, width=96, split="train"):
		self.path = path
		info_file = os.path.join(path, split+".csv")
		self.info = pd.read_csv(info_file)
		self.images = []
		self.augment = True if (split=="train" or split=="all") else False
		self.width, self.height = width, height
		self.aspect_ratio = width/height
		for i, sample in self.info.iterrows():
			img_path = os.path.join(path, "images", "img_"+str(int(sample["id"]))+".jpg")
			img = imageio.imread(img_path)
			img = np.expand_dims(img, -1).repeat(3, -1) # gray scale to rgb
			self.images.append(img)

	def __len__(self):
		return len(self.info)

	def __getitem__(self, i):
		"""Returns img (96, 128, 1) and target (20, 3) (num_landmark, #x,y,visible#)
		"""
		img = self.images[i]
		h_img, w_img, _ = img.shape
		landmarks = np.array(self.info.iloc[i].drop("id")).reshape(-1, 2) / 96.0
		isnans = np.isnan(landmarks)
		landmarks[isnans] = 0.0
		visible = np.expand_dims(np.logical_not(isnans.any(axis=-1)), axis=-1)
		img_tensor = image_to_tensor(img)
		target = torch.tensor(np.concatenate([landmarks, visible], axis=-1), dtype=torch.float32)
		return img_tensor, target

if __name__ == "__main__":
	ds = FacialLandmarkDataset("../data/facial")
	inp, tar = ds[0]
	import matplotlib.pyplot as plt 
	plt.imshow(inp.permute(1,2,0))
	plt.scatter(ds.width*tar[:,0], ds.height*tar[:,1], marker="x", color="red", s=100)
	plt.show()