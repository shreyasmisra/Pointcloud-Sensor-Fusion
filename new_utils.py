import numpy as np
import torch
import math
import open3d as o3d
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]
    return pcl_uv, pcl_z

def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:, :3].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated, cam_intrinsic)

    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) #& (pcl_z < 0)

    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)
    pc_valid = pc_rotated[mask]

    return depth_img, pcl_uv, pc_valid


def eulerAnglesToRotationMatrix(theta) :
	R_x = np.array([[1,         0,                  0                   ],
	                [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
	                [0,         math.sin(theta[0]), math.cos(theta[0])  ]
	                ])
 
	R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
	                [0,                     1,      0                   ],
	                [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
	                ])

	R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
	                [math.sin(theta[2]),    math.cos(theta[2]),     0],
	                [0,                     0,                      1]
	                ])
 
	R = np.dot(R_z, np.dot( R_y, R_x ))

	return R

def get_random_orientation(max_r, num=1):

	if num > 1:
		
		R = np.zeros((num, 3, 3))
	
		for i in range(num):
			rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
			roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
			rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)

			R[i, :, :] = eulerAnglesToRotationMatrix([rotx, roty, rotz])
					
	else:			
		rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
		roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
		rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)

		R = eulerAnglesToRotationMatrix([rotx, roty, rotz])

	return R

def get_random_translation(max_t, num=1):

	if num>1:
		T = np.zeros((3, num))
		
		for i in range(num):
			transl_x = np.random.uniform(-max_t, max_t)
			transl_y = np.random.uniform(-max_t, max_t)
			transl_z = np.random.uniform(-max_t, max_t)
			
			T[0, i] = transl_x
			T[1, i] = transl_y
			T[2, i] = transl_z
	
	else:
		transl_x = np.random.uniform(-max_t, max_t)
		transl_y = np.random.uniform(-max_t, max_t)
		transl_z = np.random.uniform(-max_t, max_t)

		T = np.array([transl_x, transl_y, transl_z]).reshape(3,1)

	return T

def get_random_pointcloud(pc, max_r, max_t, R=None, T=None, return_ext=False):
	random.seed(datetime.now())
	if not R and not T:
		R = get_random_orientation(max_r)
		T = get_random_translation(max_t)
	elif not R:
		R = get_random_orientation(max_r)
	
	#rt = np.array([1e-5,1e-5,1e-5]).reshape(3, 1)
	
	RT = np.concatenate((R, T), axis=1)
	RT = np.concatenate((RT, np.array([0,0,0,1]).reshape(1,4)), axis=0)

	RT = np.linalg.inv(RT)

	if return_ext:
		return RT @ pc, R, T, RT 

	return RT @ pc

def get_multiple_random_pointclouds(pc, num, max_r, max_t, return_ext=False):

	"""
	pc : 4xN
	"""

	R = get_random_orientation(max_r, num)
	T = get_random_translation(max_t, num)
	
	#T = np.zeros((3, num))
	num = R.shape[0]
	num_points = pc.shape[1]
	
	assert R.shape[0] == T.shape[1]
	
	pointclouds = np.zeros((num, num_points, 4))
	
	for i in range(num):
	
		RT = np.concatenate((R[i, :, :], T[:, i].reshape(3, 1)), axis=1)
		RT = np.concatenate((RT, np.array([0,0,0,1]).reshape(1,4)), axis=0)

		RT = np.linalg.inv(RT)
		
		temp = RT @ pc # 4xN
		
		#dists = pc - temp
		#dists_mean = dists.mean(axis=0)
		#temp = temp + dists_mean
		
		pointclouds[i, :, :] = temp.T # Nx4
	
	return pointclouds
	
def save_pointcloud(name, point_cloud, colors):

	assert len(colors) == 3 and max(colors) == 1
	#print(point_cloud.shape, name)
	point_cloud = point_cloud.detach().cpu().numpy()
	point_cloud = point_cloud / point_cloud[:, 3:]
	
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
	pcd.paint_uniform_color(colors)
	o3d.io.write_point_cloud(os.path.join('pointclouds', name + ".ply"), pcd)	
	
	
def show_depth(rgb, depth):

	plt.figure()
	plt.subplot(121)
	plt.imshow(rgb.cpu()[0])
	plt.subplot(122)
	plt.imshow(depth.detach().cpu().numpy()[0])
	plt.show()
	
	
	
