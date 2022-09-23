import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from projection_utils import lidar_project_depth

# RTs -> model 5, model 4, model 3, model 2, model 1. Highest range to lowest range.

rt1 = np.array([[ 0.9402, -0.3079,  0.1455,  0.7399],
        [ 0.2910,  0.9483,  0.1265, -0.1867],
        [-0.1769, -0.0766,  0.9812, -0.5519],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
rt2 = np.array([[ 0.9978, -0.0295, -0.0597,  0.1245],
        [ 0.0294,  0.9996, -0.0038, -0.0892],
        [ 0.0598,  0.0021,  0.9982,  0.1236],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
rt3 = np.array([[ 0.9995,  0.0164,  0.0268,  0.1666],
        [-0.0167,  0.9998,  0.0135, -0.0117],
        [-0.0266, -0.0140,  0.9995, -0.0309],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
rt4 = np.array([[ 9.9982e-01,  1.8701e-02,  1.7767e-03,  1.7739e-02],
        [-1.8695e-02,  9.9982e-01, -3.3016e-03, -5.6691e-04],
        [-1.8381e-03,  3.2678e-03,  9.9999e-01, -1.0346e-01],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
rt5 = np.array([[ 9.9997e-01, -7.2623e-03,  1.0225e-03,  2.8730e-02],
        [ 7.2666e-03,  9.9996e-01, -4.2275e-03, -1.8743e-02],
        [-9.9173e-04,  4.2348e-03,  9.9999e-01,  3.0232e-02],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

RTs = [rt1, rt2, rt3, rt4, rt5]

pcd = o3d.io.read_point_cloud('lab dataset/init.ply')
img = cv2.imread("lab dataset/plane.jpg")

H, W, _ = img.shape
img_shape = [H, W]

cam_calib = np.array([[1.82146e3, 0, 9.44721e2],
                  [0, 1.817312e3, 5.97134e2],
                  [0,0,1]])

points = np.asarray(pcd.points)
points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1) # Nx4

RT = np.eye(4)

for i in range(5):
    RT = np.matmul(RT, np.linalg.inv(RTs[i]))

# Rotate pointcloud by predicted extrinsic
p = points @ np.linalg.inv(RT) # prediction

# Project it onto the image plane
depth_img, pcl_uv, pc_valid = lidar_project_depth(p, cam_calib, img_shape)
init_depth_img, init_pcl_uv, init_pc_valid = lidar_project_depth(points, cam_calib, img_shape)

depth_img = np.zeros((H, W))
init_depth_img = np.zeros((H, W))
pred_img = img.copy() 
init_img = img.copy()

for i in range(pcl_uv.shape[0]):
    pred_img = cv2.circle(pred_img, (pcl_uv[i, 0], pcl_uv[i,1]), 2, (0, 255, 0), 1)
    depth_img = cv2.circle(depth_img, (pcl_uv[i, 0], pcl_uv[i,1]), 1, (255, 255, 255), 1)

for i in range(init_pcl_uv.shape[0]):
    init_img = cv2.circle(init_img, (init_pcl_uv[i, 0], init_pcl_uv[i,1]), 2, (255, 0, 0), 1)
    init_depth_img = cv2.circle(init_depth_img, (init_pcl_uv[i, 0], init_pcl_uv[i,1]), 1, (255, 255, 255), 1)

cv2.imwrite("depth_img.png", depth_img)
cv2.imwrite("projection_on_img.png", pred_img)
cv2.imwrite("init_projection.png", init_img)
cv2.imwrite("init_depth_img.png", init_depth_img)