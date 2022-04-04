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
from ipad_dataset import IPadDataset

from quaternion_distances import quaternion_distance

from utils import *
from new_utils import *
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from math import radians
from torchvision import transforms

finetune_weights = ['finetune/finetune_rot_20_trans_1.5.pth',
                    'finetune/finetune_rot_10_trans_1.0_16.52434944152832.pth',
                    'finetune/finetune_rot_5_trans_0.5_6.223089616298676.pth',
                    'finetune/finetune_rot_2_trans_0.2_4.030896157026291.pth',
                    'finetune/finetune_rot_1_trans_0.1_1.8796743541955947.pth']

weights = [
   'pretrained/kitti_iter1.tar',
   'pretrained/kitti_iter2.tar',
   'pretrained/kitti_iter3.tar',
   'pretrained/kitti_iter4.tar',
   'pretrained/kitti_iter5.tar'
]

_config = {
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
    'max_depth': 3.5,
    'iterative_method': 'multi_range', # ['multi_range', 'single_range', 'single']
    'outlier_filter': 0,
    'outlier_filter_th': 10,
    'use_reflectance':False,
    'use_prev_output':False,
    'iterative_method' :'multi_range'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1


def main(config):
    global EPOCH, finetune_weights
    
    dataset_class = IPadDataset(num=config['dataset_num'], max_t=config['max_t'], max_r=config['max_r'])    
    
    img_shape = (1440, 1920) 
    input_size = (256, 512)
    
    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_class,
                                                shuffle=False,
                                                batch_size=1,
                                                num_workers=1,
                                                drop_last=False,
                                                pin_memory=True)
    
    models = [] # iterative model
    for i in range(len(finetune_weights)):
        # network choice and settings
        if _config['network'].startswith('Res'):
            feat = 1
            md = 4
            split = _config['network'].split('_')
            for item in split[1:]:
                if item.startswith('f'):
                    feat = int(item[-1])
                elif item.startswith('md'):
                    md = int(item[2:])
            assert 0 < feat < 7, "Feature Number from PWC have to be between 1 and 6"
            assert 0 < md, "md must be positive"
            model = LCCNet(input_size, use_feat_from=feat, md=md,
                             use_reflectance=_config['use_reflectance'], dropout=_config['dropout'])
        else:
            raise TypeError("Network unknown")

        checkpoint = torch.load(finetune_weights[i], map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        models.append(model)

    
    prev_tr_error = None
    prev_rot_error = None
    
    for batch_idx, sample in enumerate(tqdm(TestImgLoader)):

        lidar_input = []
        rgb_input = []
        lidar_gt = []
        shape_pad_input = []
        real_shape_input = []
        pc_init_input = []
        RTs = []
        shape_pad = [0, 0, 0, 0]
        outlier_filter = False

        if batch_idx == 0 or not _config['use_prev_output']:
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()
        else:
            sample['tr_error'] = prev_tr_error
            sample['rot_error'] = prev_rot_error

        for idx in range(len(sample['rgb'])): 
            
            real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

            sample['point_cloud'][idx] = sample['point_cloud'][0].cuda() 

            pc_init = sample['init_pc'][0].cuda()

            depth_img, uv_input, pc_input_valid = lidar_project_depth(pc_init, sample['K'][idx], real_shape)  # image_shape
            depth_img /= _config['max_depth']

            #show_depth(sample['rgb'][0], depth_img)
            save_pointcloud(f"GT", sample['point_cloud'][0], [0, 1, 0])
            save_pointcloud(f"init", pc_init, [0, 0, 1])

            # PAD ONLY ON RIGHT AND BOTTOM SIDE
            rgb = sample['rgb'][idx].cuda() # 3, H, W
            shape_pad = [0, 0, 0, 0]

            shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
            shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

            rgb = F.pad(rgb, shape_pad)
            depth_img = F.pad(depth_img, shape_pad)

            rgb_input.append(rgb)
            lidar_input.append(depth_img)
            real_shape_input.append(real_shape)
            shape_pad_input.append(shape_pad)
            pc_init_input.append(pc_init)

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)
        rgb_resize = F.interpolate(rgb_input.type(torch.FloatTensor), size=[256, 512], mode="bilinear", align_corners=False)
        lidar_resize = F.interpolate(lidar_input.type(torch.FloatTensor), size=[256, 512], mode="bilinear", align_corners=False)
        rgb = rgb_input.to(device)
        lidar = lidar_input.to(device)
        rgb_resize = rgb_resize.to(device)
        lidar_resize = lidar_resize.to(device)

        rt = np.array([ [1,0,0,0], 
        				[0,1,0,0], 
        				[0,0,1,0], 
        				[0,0,0,1]])
        rt = torch.from_numpy(rt).type(torch.FloatTensor)
        rt = rt.cuda()
        count = 0
        with torch.no_grad():
            for iteration in [0, 1, 2, 3, 4]:
                
                T_predicted, R_predicted = models[iteration](rgb_resize, lidar_resize)
                R_predicted = quat2mat(R_predicted[0])
                T_predicted = tvector2mat(T_predicted[0])
                #print(T_predicted)
                RT_predicted = torch.mm(T_predicted, R_predicted)
                #RTs.append(torch.mm(RTs[iteration], RT_predicted)) # inv(H_gt)*H_pred_1*H_pred_2*.....H_pred_n
                #rt = rt.cuda() @ RT_predicted
                if count == 0:
                    pc_init = pc_init_input[0]
                else:
                    pc_init = pc_init
                
                pc_init = rotate_back(pc_init.type(torch.FloatTensor).cuda(), RT_predicted) # H_pred*X_init

                depth_img_pred, uv_pred, pc_pred_valid = lidar_project_depth(pc_init, sample['K'][0], real_shape_input[0]) # image_shape
                depth_img_pred /= _config['max_depth']
                
                #show_depth(sample['rgb'][0], depth_img_pred)    
                save_pointcloud(f"pred_{count}", pc_init, [1,1,0])            
               	count+=1
                depth_pred = F.pad(depth_img_pred, shape_pad_input[0])
                lidar = depth_pred.unsqueeze(0)
                lidar_resize = F.interpolate(lidar.type(torch.FloatTensor), size=[256, 512], mode="bilinear", align_corners=False)
                lidar_resize = lidar_resize.to(device)

#        pc_final = rotate_forward(pc_rotated_input[0].detach().cpu(), rt.detach().cpu())
#        pcd_final = o3d.geometry.PointCloud()
#        pcd_final.points = o3d.utility.Vector3dVector(pc_final.t().detach().cpu()[:, :3])
#        pcd_final.paint_uniform_color([1, 1, 0])
#        o3d.io.write_point_cloud("pred_final.ply", pcd_final)       

if __name__ == '__main__':
    main(_config)
