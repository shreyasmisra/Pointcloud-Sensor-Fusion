import csv
import os
import json

import cv2

#import open3d as o3d

import matplotlib.pyplot as plt

from datetime import datetime

import random
import mathutils
import numpy as np

import math
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

from new_utils import (get_random_pointcloud, get_multiple_random_pointclouds, save_pointcloud, get_random_translation, get_random_orientation)

from torchvision import transforms

class IPadDataset(Dataset):
	def __init__(self, num=1, dataset='ipadDataset', max_r=20., max_t=1.5, get_multiple=None, with_batch=None):
		super(IPadDataset, self).__init__()
		
		
		self.max_r = max_r
		self.max_t = max_t
		self.root_dir = 'ipadDataset' + str(num)
		self.rgb_paths = os.listdir(os.path.join(self.root_dir, 'rgb'))
		self.rgb_paths = [os.path.join(self.root_dir, 'rgb', i) for i in self.rgb_paths]
		self.lidar_path = os.path.join(self.root_dir, 'scan.xyz')
		self.json_path = os.path.join(self.root_dir, 'frame_00000.json')
		self.init_lidar_path = os.path.join(self.root_dir, 'Init_after.ply')
		self.get_multiple = get_multiple
		self.with_batch = with_batch
		
		def read_points(file):
			data = []
			with open(file, 'r') as f:
				for i in f:
				    data.append(list(map(float, i.split(','))))

			data = np.array(data)
			points = data[:,:3]
			# if (points[:,-1] <  0).all():
			#     points[:,-1] *= -1

			return points
		
		def read_json(FILE_PATH):
			f = open(FILE_PATH)
			data = json.load(f)
			intrinsics = np.array(data['intrinsics']).reshape(3,3)
			extrinsic = np.array(data['cameraPoseARFrame']).reshape(4,4)
			projectionmtx = np.array(data['projectionMatrix']).reshape(4,4)
			extrinsic = extrinsic @ projectionmtx

			params = {"ext":extrinsic, "k":intrinsics, 'cam_pose':np.array(data['cameraPoseARFrame']).reshape(4,4), 						'pmtx':np.array(data['projectionMatrix']).reshape(4,4)}
			return params
		
		self.lidar = read_points(self.lidar_path)
		self.lidar = np.concatenate((self.lidar, np.ones((self.lidar.shape[0], 1))), axis=1)
		#self.init_lidar_pcd = o3d.io.read_point_cloud(self.init_lidar_path)
		#self.init_lidar = np.asarray(self.init_lidar_pcd.points) # Nx3
		#self.init_lidar = np.concatenate((self.init_lidar, np.ones((self.init_lidar.shape[0], 1))), axis=1)
		
		self.params = read_json(self.json_path)
		self.K = self.params['k']
		self.cam_pose = self.params['cam_pose']
		self.cam_pose_inv = np.linalg.inv(self.cam_pose)
		self.proj_mtx = self.params['pmtx']
		
		self.identity_ext_matrix = np.array([[1,0,0,0],
											[0,1,0,0],
											[0,0,1,0],
											[0,0,0,1]])
		
		random.seed(datetime.now())
	
	def custom_transform(self, rgb, img_rotation=0., flip=False):
		to_tensor = transforms.ToTensor()
		normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
#		print("before",rgb.shape)
		rgb = to_tensor(rgb)
#		print("after1",rgb.shape)		
		rgb = normalization(rgb)
#		print("after2",rgb.shape)		
		return rgb
		
	def __len__(self):
		if self.with_batch:
			return self.with_batch
		return len(self.rgb_paths) 
	
	def __getitem__(self, idx):
		self.rgb_path = self.rgb_paths[0]

		IMG = cv2.imread(self.rgb_path)
		#IMG = cv2.rotate(IMG, cv2.ROTATE_90_CLOCKWISE)
		IMG = cv2.flip(IMG, 1)

		img_rotation = 0.
		h_mirror = False		
		self.img = self.custom_transform(IMG, img_rotation, h_mirror)
		#print(self.img.shape)
		self.img_shape = self.img.shape[1:]
		
		pc_org = self.lidar.astype(np.float32)
		pc_gt_rot = self.cam_pose_inv @ pc_org.T # 4xN
		pc_gt = torch.from_numpy(pc_gt_rot.astype(np.float32).copy().T) # Nx4
		
		init_pc, R, T, init_RT = get_random_pointcloud(pc_gt_rot, self.max_r, self.max_t, return_ext=True)
		init_pc = torch.from_numpy(init_pc.astype(np.float32).T) # Nx4		

		#dists = pc_gt - init_pc
		#dists_mean = dists.numpy().mean(axis=0)
		
		#init_pc = init_pc + torch.from_numpy(dists_mean)
				
		RT = torch.tensor(init_RT.astype(np.float32))
		R, T = torch.tensor(R), torch.tensor(T) 
		R_quat = quaternion_from_matrix(R)

		sample = {'rgb':self.img, 'K':self.K, 'extrin':self.cam_pose_inv, 'tr_error': T, 'rot_error': R_quat, 'index':idx, 'point_cloud': pc_gt,
				'init_pc':init_pc, 'RT':RT}

		return sample
