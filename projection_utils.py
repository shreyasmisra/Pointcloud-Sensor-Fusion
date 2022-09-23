import open3d as o3d
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import math

def read_points(file):
    data = []
    with open(file, 'r') as f:
        for i in f:
            data.append(list(map(float, i.split(','))))

    data = np.array(data)
    points = data[:,:3]

    return points

def read_json(FILE_PATH):
    f = open(FILE_PATH)
    data = json.load(f)
    intrinsics = np.array(data['intrinsics']).reshape(3,3)
    extrinsic = np.array(data['cameraPoseARFrame']).reshape(4,4)
    projectionmtx = np.array(data['projectionMatrix']).reshape(4,4)
    extrinsic = extrinsic @ projectionmtx

    params = {"ext":extrinsic, "k":intrinsics, 'cam_pose':np.array(data['cameraPoseARFrame']).reshape(4,4), 'pmtx':np.array(data['projectionMatrix']).reshape(4,4)}
    return params

def parse_xyz(FILE_PATH):
    xyz = []
    with open(FILE_PATH, "r") as f:
        for row in f:
            parsed  = row.strip().split(',')
            xyz.append(list(map(float, parsed)))
    
    return np.array(xyz)


def scale(arr, high):
    """
    Scale the arr to values from 0 to high
    """
    min_ = arr.min()
    max_ = arr.max()

    # return arr
    return (arr-min_)*(high-1)/(max_ - min_)


def get_extrinsic_matrix(rvecs, tvecs):
    """
    Get the extrinsic matrix -> (3,4)
    """
    rvecs = np.array(rvecs)[:,:,0]
    tvecs = np.array(tvecs)[:,:,0]

    R = np.mean(rvecs, axis=0).reshape(-1,1)
    t = np.mean(tvecs, axis=0).reshape(-1,1)
    R, _ = cv2.Rodrigues(R)
    R_t = np.concatenate((R,t), axis=1)

    return R_t

def apply_extrinsic_transform(points, C):
    """
    Apply extrinsic transform
    returns -> Points (M, 3)
    """
    points = np.hstack((points, np.ones((points.shape[0],1))))
    transform_points = np.dot(C, points.T)
    transform_points = transform_points.T

    return transform_points

def apply_intrinsic_transform(points, K, N, M):
    """
    Apply intrinsic transform
    returns:
    xyz -> (M,3) where Z != 0
    u -> (M, 1)
    v -> (M, 1)
    z -> (M, 1)
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    x = []
    y = []
    z = []
    idx = []

    u = []
    v = []

    for i in range(points.shape[0]):
        if points[i,2]>=0:
            uu = (fx * points[i,0])/(points[i,2] + 1e-5)
            vv = (fy * points[i,1])/(points[i,2] + 1e-5)
            x.append(points[i,0])
            y.append(points[i,1])
            z.append(points[i,2])
            idx.append(i)
            u.append(uu + cx)
            v.append(vv + cy)

    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    z = np.array(z).reshape(-1,1)

    xyz = np.hstack((x, y, z))
    u = scale(np.array(u), M).astype(np.int64)
    v = scale(np.array(v), N).astype(np.int64)

    return xyz, idx, u, v

def apply_color(img, u, v):
    """
    Get color from the RGB image using the projected points (u,v)
    """
    b = img[v,u,0].reshape(-1,1)/255
    g = img[v,u,1].reshape(-1,1)/255
    r = img[v,u,2].reshape(-1,1)/255
    colors = np.hstack((r,g,b))

    return colors

def mean_normalize(pts):
    return (pts - np.mean(pts))/(np.std(pts) + 1e-10)

def get_depth(N, M, u, v, depth_map = False):
    """
    Generate a mask with black points or a depth map using z
    """
    
    mask = 0*np.ones((N,M)) # black background

    if type(depth_map) != bool:
        r = depth_map[:,2]
        mask[v,u] = r.ravel().astype(np.float32)
        return mask # depth
    
    mask[v,u] = 255
    return mask
    

def project_on_image(img, u, v, colors):
    """
    Project the points on the original image and 
    color those pixels with the corresponding RGB values from the pointcloud (based on depth).
    
    Only works for pointclouds which have RGB color values.

    Returns:
    img -> original image with projected points
    new_image -> new image with RGB values representing distance
    """

    img = img/255
    r = colors[:,0].reshape(-1,1)
    g = colors[:,1].reshape(-1,1)
    b = colors[:,2].reshape(-1,1)
    colors = np.hstack((b,g,r))

    new_img = np.ones(img.shape)
    new_img[v, u, :] = colors
        
    img[v,u, :] = colors

    return img, new_img


def visualize_pointcloud(pcd):

    if type(pcd) == list:
        o3d.visualization.draw_geometries(pcd,
                                    zoom=5,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
    else:
        o3d.visualization.draw_geometries([pcd],
                                    zoom=5,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])

def extract_plane(pcd, n, thresh, iterations):
    plane_model, inliers = pcd.segment_plane(distance_threshold=thresh,
                                         ransac_n=n,
                                         num_iterations=iterations)
    
    return plane_model, inliers

def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]
    return pcl_uv, pcl_z

def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    """
    pc_rotated: Nx4
    """
    pc_rotated = pc_rotated[:, :3]
    cam_intrinsic = cam_calib
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated, cam_intrinsic)

    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) #& (pcl_z < 0)

    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    depth_img = np.zeros((img_shape[0], img_shape[1]))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = depth_img.astype(np.float32)
    pc_valid = pc_rotated[mask]

    return depth_img, pcl_uv, pc_valid

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]])
 
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]])
 
    R = np.dot(R_z, np.dot( R_y, R_x ))        
    return R

def save_point_cloud(p, color):
    """
    p : 4 x N
    """
    
    p = p.T
    p = p / p[:,3:]
    pd = o3d.geometry.PointCloud()
    pd.points = o3d.utility.Vector3dVector(p[:, :3])
    pd.paint_uniform_color(color)
    o3d.io.write_point_cloud("saved_point_clouds.ply", pd)


if __name__ == "__main__":    
    file = "data\\points.xyz"

    print(parse_xyz(file).shape)