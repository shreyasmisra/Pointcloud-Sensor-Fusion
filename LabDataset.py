import numpy as np
import math
import cv2
import os
import open3d as o3d

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

from new_utils import (get_random_pointcloud, get_multiple_random_pointclouds, save_pointcloud, get_random_translation, get_random_orientation)

class LabDataset(Dataset):
	def __init__(self, max_r=20., max_t=1.5, with_batch=None):
		super(LabDataset, self).__init__()
	
		self.max_r = max_r
		self.max_t = max_t
		self.root = 'lab_dataset'
		
		#self.rgb_path = os.path.join(self.root, "plane.jpg")
		#self.lidar_path = os.path.join(self.root, "plane.ply")
		self.rgb_path = os.path.join(self.root, "pc_window.jpg")
		self.lidar_path = os.path.join(self.root, "pc_window_with_high_confidence.ply")
		
		self.with_batch = with_batch
		
		self.lidar_pcd = o3d.io.read_point_cloud(self.lidar_path)
		self.lidar_pcd = self.lidar_pcd.voxel_down_sample(voxel_size=7) # 20 previously
		self.lidar = np.asarray(self.lidar_pcd.points)
		
		inds = 2000 > self.lidar[:,2] # thresholding
		self.lidar = self.lidar[inds]
		
		self.lidar = self.lidar * 1e-3
		self.lidar = np.concatenate((self.lidar, np.ones((self.lidar.shape[0], 1))), axis=1) # N,4
		
		self.K = np.array([[1.82146e3, 0, 9.44721e2],
							[0, 1.817312e3, 5.97134e2],
							[0,0,1]])
	
	def custom_transform(self, rgb, img_rotation=0., flip=False):
		to_tensor = transforms.ToTensor()
		normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
		rgb = to_tensor(rgb)
		rgb = normalization(rgb)
		return rgb
	
	def __len__(self):
		if self.with_batch:
			return self.with_batch
		return 1
	
	def __getitem__(self, i):

		self.img = cv2.imread(self.rgb_path)
		img_rotation = 0.
		h_mirror = False
		self.img = self.custom_transform(self.img, img_rotation, h_mirror)	
		
		self.img_shape = self.img.shape[1:]
		
		pc_org = self.lidar.astype(np.float32) # N, 4
		
		r = get_random_orientation(self.max_r)
		t = get_random_translation(self.max_t)
		
		RT = np.concatenate((r, t), axis=1)
		RT = np.concatenate((RT, np.array([0,0,0,1]).reshape(1, 4)), axis=0)
		
#		print(RT)
		RT = np.linalg.inv(RT)
		
		init_pc = pc_org @ RT # N,4 
		
		init_pc = torch.from_numpy(init_pc.astype(np.float32))
		RT = torch.tensor(RT.astype(np.float32))

		R, T = torch.tensor(r), torch.tensor(t) 
		tr_error = T
		rot_error = quaternion_from_matrix(R)	
		
		sample = {'rgb': self.img, 'K':self.K, 'tr_error': tr_error, 'rot_error': rot_error, 'point_cloud': pc_org,	'init_pc':init_pc, 'RT':RT}

		return sample	

if __name__ == '__main__':
	dataset = next(iter(LabDataset()))	
	
	print(dataset['tr_error'])
	
