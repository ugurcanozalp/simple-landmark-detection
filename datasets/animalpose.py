import os
import numpy as np
import math
import glob
import json
import imageio
import torch
import matplotlib.pyplot as plt 
from skimage.transform import resize, rotate
import random
from torch.utils.data import Dataset
from torchvision import transforms


skeleton = [[0,1],[0,2],[1,2],[0,3],[1,4],[2,17],[18,19],[5,9],[6,10],[7,11],[8,12],[9,13],[10,14],[11,15],[12,16]]
swap_map = np.array([1,0,2,4,3,6,5,8,7,10,9,12,11,14,13,16,15,17,18,19], dtype=np.int)

def rotate_randomly(image, landmarks, center, visible):
	angle = np.random.uniform(low=-0.1, high=0.1) # 0.1 radian approximately 6 degree
	rot_mat = np.array( [ [ np.cos(angle)  ,   np.sin(angle)],
						[ -np.sin(angle) ,   np.cos(angle)] ] ) 
	center_1 = np.expand_dims(center, 0)
	landmarks = center_1 + np.matmul(landmarks-center_1, rot_mat.transpose()).astype('float32')
	image = rotate(image, 180/np.pi*angle, center=center)
	return image, landmarks, center, visible

def flip_randomly(image, landmarks, center, visible):
	p = random.random()
	if p < 0.5:
		landmarks = landmarks[swap_map]
		landmarks[:,0] = image.shape[1]-landmarks[:,0]
		visible = visible[swap_map]
		image = image[:, ::-1]
		center[0] = image.shape[1]-center[0]
	return image, landmarks, center, visible

def color_jitter(image, sigma=0.05, brightness=0.1, contrast=0.1):
	if image.dtype == "uint8":
		image = image/255.0
	#image = image.copy()
	noise = np.random.normal(0, sigma, image.shape)
	alpha = 1-np.random.randn()*contrast
	beta = np.random.randn()*brightness
	image = alpha*image + beta + noise
	mask_overflow_upper = image >= 1.0
	mask_overflow_lower = image < 0
	image[mask_overflow_upper] = 1.0
	image[mask_overflow_lower] = 0
	return image

def augment(image, landmarks, center, visible):
	image, landmarks, center, visible = rotate_randomly(image, landmarks, center, visible)
	image, landmarks, center, visible = flip_randomly(image, landmarks, center, visible)
	image = color_jitter(image)
	return image, landmarks, center, visible

def image_to_tensor(image):
	if image.dtype == "uint8":
		image = image/255.0
	return torch.tensor(image.copy(), dtype=torch.float32).permute(2,0,1)

class AnimalPoseLandmarkDataset(Dataset):

	def __init__(self, path, height=96, width=128, split="train"):
		self.path = path
		info_file = os.path.join(path, split+".json")
		with open(info_file) as f:
			self.info = json.load(f)
		self.images = {}
		self.augment = True if (split=="train" or split=="all" or split=="augment") else False
		self.width, self.height = width, height
		self.aspect_ratio = width/height
		for image_id, image_name in self.info["images"].items():
			image_path = os.path.join(path, "images", image_name)
			self.images[image_id] = imageio.imread(image_path)
	def __len__(self):
		return len(self.info["annotations"])

	def __getitem__(self, i):
		"""Returns image (h, w, 1) and target (20, 3) (num_landmark, #x,y,visible#)
		"""
		annotation = self.info["annotations"][i]
		image_id = str(annotation["image_id"])
		image = self.images[image_id]
		h_img, w_img, _ = image.shape
		x1, y1, x2, y2 = annotation["bbox"]
		h_bbox, w_bbox = y2-y1, x2-x1
		center = np.array([(x2+x1)/2, (y2+y1)/2], dtype=np.float32)
		landmarks_visible = np.array(annotation["keypoints"])
		landmarks = landmarks_visible[:, :-1]
		visible = landmarks_visible[:, -1:]
		if self.augment:
			image, landmarks, center, visible = augment(image, landmarks, center, visible)
			x1, y1, x2, y2 = center[0]-w_bbox/2, center[1]-h_bbox/2, center[0]+w_bbox/2, center[1]+h_bbox/2
		if self.augment:
			x1 = max(0, x1-w_bbox*np.random.rand()*0.02)
			y1 = max(0, y1-h_bbox*np.random.rand()*0.02)
			x2 = min(w_img, x2+w_bbox*np.random.rand()*0.02)
			y2 = min(h_img, y2+h_bbox*np.random.rand()*0.02)
			h_bbox, w_bbox = y2-y1, x2-x1
			#y_mid, x_mid = (y2+y1)/2, (x2+x1)/2
		aspect_ratio = w_bbox/h_bbox
		if aspect_ratio < self.aspect_ratio: # height is greater
			x_err = (h_bbox*self.aspect_ratio - w_bbox)/2
			x1, x2 = x1 - x_err, x2 + x_err
		else:
			y_err = (w_bbox/self.aspect_ratio - h_bbox)/2
			y1, y2 = y1 - y_err, y2 + y_err
		up_pad, down_pad, left_pad, right_pad = 0, 0, 0, 0
		if x1 < 0:
			left_pad = abs(x1)
			x1 = 0
		if y1 < 0:
			up_pad = abs(y1)
			y1 = 0
		if x2 > w_img:
			right_pad = abs(x2-w_img)
			x2 = w_img
		if y2 > h_img:
			down_pad = abs(y2-h_img)
			y2 = h_img
		crop_image = image[int(y1):int(y2), int(x1):int(x2)]
		x1 -= left_pad
		y1 -= up_pad
		x2 += right_pad
		y2 += down_pad
		if up_pad!=0 or down_pad!=0 or left_pad!=0 or right_pad!=0:
			crop_image = np.pad(crop_image, ((int(up_pad), int(down_pad)), (int(left_pad), int(right_pad)), (0, 0)), "edge")
		crop_image = resize(crop_image, (self.height, self.width), order=2)
		crop_landmarks = (landmarks - np.array([[x1, y1]]))/(np.array([[x2, y2]]) - np.array([[x1, y1]]))
		is_missed = np.logical_or((crop_landmarks>1.0).any(axis=-1, keepdims=True), (crop_landmarks<0.0).any(axis=-1, keepdims=True)) 
		visible = visible * np.logical_not(is_missed)
		crop_img_tensor = image_to_tensor(crop_image)
		target = torch.tensor(np.concatenate([crop_landmarks, visible], axis=-1), dtype=torch.float32)
		return crop_img_tensor, target

if __name__ == "__main__":
	ds = AnimalPoseLandmarkDataset("../data/animalpose")
	for inp, tar in ds:
		plt.imshow(inp.permute(1,2,0))
		#plt.scatter(ds.width*tar[:,0], ds.height*tar[:,1], marker="x", color="red", s=100)
		for i, (x,y,vis) in enumerate(tar):
			text = str(i) if vis>0 else ""
			plt.annotate(text, (ds.width*x, ds.height*y), color='red', fontsize='large')
			#plt.annotate("x", (ds.width*x, ds.height*y), color='red', fontsize='large')
		plt.show()