import csv
import random
import open3d as o3d
import cv2

import mathutils
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from skimage import io

from tqdm import tqdm
import time

from models.LCCNet import LCCNet
from StanfordDataset import StanfordDataset
from ipad_dataset import IPadDataset

from quaternion_distances import quaternion_distance

from utils import *
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from math import radians
from torchvision import transforms

weights = [
   'pretrained/kitti_iter1.tar',
   'pretrained/kitti_iter2.tar',
   'pretrained/kitti_iter3.tar',
   'pretrained/kitti_iter4.tar',
   'pretrained/kitti_iter5.tar'
]

config = {
    'RUN': 4,
    'dataset': 'stanford',
    'dataset_num': 3,
    'max_t': 1.5,
    'max_r': 20.,
    'occlusion_kernel': 5,
    'occlusion_threshold': 3.0,
    'network': 'Res_f1', # can be changed but will have to train it all over again
    'norm' : 'bn',
    'save_log': True,
    'dropout': 0.0,
    'max_depth': -1.684,
    'iterative_method': 'multi_range', # ['multi_range', 'single_range', 'single']
    'outlier_filter': 0,
    'outlier_filter_th': 10,
    'use_reflectance':False,
    'use_prev_output':False,
    'iterative_method' :'multi_range'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]
    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)

    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)

    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.cuda()
    depth_img = depth_img.permute(2, 0, 1)
    pc_valid = pc_rotated.T[mask]

    return depth_img, pcl_uv, pc_valid

def calculate_error(t_pred, r_pred, target_transl, target_rot):

    total_trasl_error = torch.tensor(0.0)
    print(r_pred.shape, target_rot.shape)
    total_rot_error = quaternion_distance(target_rot, r_pred, target_rot.device)
    total_rot_error = total_rot_error * 180. / np.pi
    for j in range(3):
        total_trasl_error += torch.norm(target_transl[0,j] - t_pred[0,j]) #* 100.

    return total_trasl_error.item(), total_rot_error.sum().item()


def main(config):
    global EPOCH, weights

    if config['iterative_method'] == 'single':
        weights = [weights[0]]

    #dataset_class = StanfordDataset(max_t=config['max_t'], max_r=config['max_r'])
    dataset_class = IPadDataset(num=config['dataset_num'], max_t=config['max_t'], max_r=config['max_r'])    
    
    img_shape = (1440, 1920) # change this
    #img_shape = (1920, 1440)
    #img_shape = (1080, 1080)
    input_size = (256, 512)


    num_worker = 1 
    batch_size = 1

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_class,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                drop_last=False,
                                                pin_memory=True)

    models = [] # iterative model
    for i in range(len(weights)):
        # network choice and settings
        if config['network'].startswith('Res'):
            feat = 1
            md = 4
            split = config['network'].split('_')
            for item in split[1:]:
                if item.startswith('f'):
                    feat = int(item[-1])
                elif item.startswith('md'):
                    md = int(item[2:])
            assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
            assert 0 < md, "md must be positive"
            model = LCCNet(input_size, use_feat_from=feat, md=md,
                             use_reflectance=config['use_reflectance'], dropout=config['dropout'])
        else:
            raise TypeError("Network unknown")

        checkpoint = torch.load(weights[i], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)

    prev_tr_error = None
    prev_rot_error = None

    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):
        #N = 100 # 500

        log_string = [str(batch_idx)]

        lidar_input = []
        rgb_input = []
        lidar_gt = []
        shape_pad_input = []
        real_shape_input = []
        pc_rotated_input = []
        RTs = []
        rt = []
        shape_pad = [0, 0, 0, 0]
        outlier_filter = False

        if batch_idx == 0 or not config['use_prev_output']:
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
        else:
            sample['tr_error'] = prev_tr_error
            sample['rot_error'] = prev_rot_error

        for idx in range(len(sample['rgb'])):
            real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

            pc_rotated = sample['pc_rotated'][idx].cuda() # Nx4
            sample['point_cloud'][0] = sample['point_cloud'][0].cuda()
            
            #R = quat2mat(sample['rot_error'][idx])
            #T = tvector2mat(sample['tr_error'][idx])

            depth_img, uv_input, pc_input_valid = lidar_project_depth(pc_rotated, sample['K'][idx], real_shape)  # image_shape
            depth_img /= config['max_depth']

            plt.figure()
            plt.subplot(121)
            plt.imshow(sample['rgb'][0].cpu()[0])
            plt.subplot(122)
            plt.imshow(depth_img.detach().cpu().numpy()[0])
            #plt.show()
            
            print(pc_rotated.shape)
            print(sample['point_cloud'].shape)
            
            pcd_init = o3d.geometry.PointCloud()
            pcd_gt = o3d.geometry.PointCloud()

            pcd_init.points = o3d.utility.Vector3dVector(pc_rotated.t().detach().cpu()[:, :3])
            pcd_gt.points = o3d.utility.Vector3dVector(sample['point_cloud'][0].t().detach().cpu()[:, :3])            
            
            pcd_init.paint_uniform_color([0, 0, 1])
            pcd_gt.paint_uniform_color([0, 1, 0])
            
            o3d.io.write_point_cloud("GT.ply", pcd_gt)
            o3d.io.write_point_cloud("init.ply", pcd_init)            
            
            #np.save(f"depth_init_window", depth_img.detach().cpu().numpy()[0])
            
            #plt.imsave(f"depth_init_window.png", depth_img.detach().cpu().numpy()[0])

            rgb = sample['rgb'][idx].to(device) # 3, H, W
            shape_pad = [0, 0, 0, 0]

            shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
            shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

            #rgb = F.pad(rgb, shape_pad)
            #depth_img = F.pad(depth_img, shape_pad).cuda()

            rgb_input.append(rgb)
            lidar_input.append(depth_img)
            real_shape_input.append(real_shape)
            shape_pad_input.append(shape_pad)
            pc_rotated_input.append(pc_rotated)
            #RTs.append(RT)
            rt = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
            rt = torch.from_numpy(rt).type(torch.FloatTensor)
            
            #rt = rt.cuda()

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)
        rgb_input = rgb_input.to(device)
        lidar_input = lidar_input.to(device)
        rgb_resize = F.interpolate(rgb_input.type(torch.FloatTensor), size=[256, 512], mode="bilinear")
        lidar_resize = F.interpolate(lidar_input.type(torch.FloatTensor), size=[256, 512], mode="bilinear")
        rgb = rgb_input.to(device)
        lidar = lidar_input.to(device)
        rgb_resize = rgb_resize.to(device)
        lidar_resize = lidar_resize.to(device)
 
        with torch.no_grad():
            for iteration in [0, 1, 2, 3, 4]:
                # Run the i-th network
                
                T_predicted, R_predicted = models[iteration](rgb_resize, lidar_resize)
                
                # Project the points in the new pose predicted by the i-th network
                R_predicted = quat2mat(R_predicted[0]) # We are taking the inverse of the predicted matrix to compute the new pointcloud.
                T_predicted = tvector2mat(T_predicted[0]) # Hence, i think the output of the model is the cam pose. So inv of cam pose will be the extrinsic matrix
                RT_predicted = torch.mm(T_predicted, R_predicted)
                #RTs.append(torch.mm(RTs[iteration], RT_predicted)) # inv(H_gt)*H_pred_1*H_pred_2*.....H_pred_n
                rt = torch.mm(rt, RT_predicted.detach().cpu())
                
                #total_trans_err, total_rot_err = calculate_error(t_pred, r_pred, target_transl, target_rot)
                
                #print(f"Translation error iteration {iteration+1} = {total_trans_err}")
                #print(f"Rotation error iteration {iteration+1} = {total_rot_err}")
                                
                #print("pred", RT_predicted)
                #print("inv", np.linalg.inv(RT_predicted.cpu()))
                #print(RTs[iteration])
                
                if iteration == 0 :
                    rotated_point_cloud = pc_rotated_input[0]
                else:
                    rotated_point_cloud = rotated_point_cloud
                
                #rotated_point_cloud = torch.mm(rotated_point_cloud, RT_predicted.inverse())
                rotated_point_cloud = rotate_forward(rotated_point_cloud, RT_predicted)
                print(rotated_point_cloud.shape)
                # rotated_point_cloud =  pc_rotated @ RT_predicted.clone().inverse() DOES NOT WORK
                # rotated_point_cloud =  RT_predicted.clone().inverse() @ rotated_point_cloud.t() DOES NOT WORK
                #rotated_point_cloud = rotate_forward(sample['point_cloud'][0].cuda(), RT_predicted) - DO NOT USE
                                
                depth_img_pred, uv_pred, pc_pred_valid = lidar_project_depth(rotated_point_cloud, sample['K'][0], real_shape_input[0]) # image_shape
                #print(depth_img_pred.flatten().max())
                depth_img_pred /= config['max_depth']
                
                #if batch_idx == 50:
                plt.figure()
                plt.subplot(121)
                plt.imshow(sample['rgb'][0].cpu()[0])
                plt.subplot(122)
                plt.imshow(depth_img_pred.detach().cpu().numpy()[0])
                #plt.show()
                
                pcd_pred = o3d.geometry.PointCloud()
                
                pcd_pred.points = o3d.utility.Vector3dVector(rotated_point_cloud.t().detach().cpu()[:, :3])
                
                pcd_pred.paint_uniform_color([1, 0, 0])
                
                o3d.io.write_point_cloud(f"pred_{iteration+1}.ply", pcd_pred)
                
                #np.save(f"depth_pred_window_{iteration}", depth_img_pred.detach().cpu().numpy()[0])
                
                #plt.imsave(f"depth_pred_window_{iteration}.png", depth_img_pred.detach().cpu().numpy()[0])
                
                #rotated_point_cloud = rotate_back(rotated_point_cloud,  RT_predicted)
                
                depth_pred = F.pad(depth_img_pred, shape_pad_input[0])
                lidar = depth_pred.unsqueeze(0)
                lidar_resize = F.interpolate(lidar.type(torch.FloatTensor), size=[256, 512], mode="bilinear")
                lidar_resize = lidar_resize.to(device)
                    
        pc_final = rotate_forward(pc_rotated_input[0].detach().cpu(), rt.detach().cpu())
        pcd_final = o3d.geometry.PointCloud()
                
        pcd_final.points = o3d.utility.Vector3dVector(pc_final.t().detach().cpu()[:, :3])
        
        pcd_final.paint_uniform_color([1, 1, 0])
        
        o3d.io.write_point_cloud("pred_final.ply", pcd_final)

if __name__ == '__main__':
	main(config)
 

