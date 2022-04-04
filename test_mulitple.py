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
   #'pretrained/kitti_iter1.tar',
   #'pretrained/kitti_iter2.tar',
   #'pretrained/kitti_iter3.tar',
   #'pretrained/kitti_iter4.tar',
   #'pretrained/kitti_iter5.tar'
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
    
    dataset_class = IPadDataset(num=config['dataset_num'], max_t=config['max_t'], max_r=config['max_r'], get_multiple=10)    
    
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
        model = LCCNet(input_size, use_feat_from=1, md=4,
                            use_reflectance=_config['use_reflectance'], dropout=_config['dropout'])
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
        pc_rotated_input = []
        RTs = []
        shape_pad = [0, 0, 0, 0]
        outlier_filter = False

        for idx in range(len(sample['rgb'])): 
            
            real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

            sample['point_cloud'][idx] = sample['point_cloud'][0].cuda() 
            save_pointcloud(f"GT", sample['point_cloud'][idx], [0,1,0]) # save the groundtruth pointcloud

            init_pcs = sample['init_pcs'][0]

            rgb = sample['rgb'][idx].cuda() # 3, H, W
            shape_pad = [0, 0, 0, 0]

            shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
            shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

            rgb = F.pad(rgb, shape_pad)
            
            for i in range(init_pcs.shape[0]):
                pc = init_pcs[i].cuda()
                
                save_pointcloud(f"init_{i}", pc, [0,0,1]) # save the initial random pointclouds
                
                lidar_resize = []

                depth_img, uv_input, pc_input_valid = lidar_project_depth(pc, sample['K'][idx], real_shape)  # image_shape
                depth_img /= _config['max_depth']

                depth_img = F.pad(depth_img, shape_pad)
                lidar_resize.append(depth_img)
                lidar_resize = torch.stack(lidar_resize)

                lidar_resize = F.interpolate(lidar_resize.type(torch.FloatTensor), size=[256, 512], mode="bilinear", align_corners=False)

                lidar_input.append(lidar_resize[0])

            rgb_input.append(rgb)
            real_shape_input.append(real_shape)
            shape_pad_input.append(shape_pad)

        lidar_inputs = torch.stack(lidar_input) # num, 1, 256, 512

        rgb_input = torch.stack(rgb_input)
        rgb_resize = F.interpolate(rgb_input.type(torch.FloatTensor), size=[256, 512], mode="bilinear", align_corners=False)
        rgb = rgb_input.to(device)
        rgb_resize = rgb_resize.to(device)
        
        rt = np.array([ [1,0,0,0], 
        				[0,1,0,0], 
        				[0,0,1,0], 
        				[0,0,0,1]])
        rt = torch.from_numpy(rt).type(torch.FloatTensor)
        rt = rt.cuda()
        
        with torch.no_grad():
            for i in range(lidar_inputs.shape[0]):
                lidar_inp = []
                lidar_inp.append(lidar_inputs[i])
                lidar_inp = torch.stack(lidar_inp) # 1, 1, 256, 512
                lidar_inp = lidar_inp.cuda()
                count = 0
                for iteration in [0, 1, 2, 3, 4]:            
                    T_predicted, R_predicted = models[iteration](rgb_resize, lidar_inp)
                    R_predicted = quat2mat(R_predicted[0])
                    T_predicted = tvector2mat(T_predicted[0])
                    RT_predicted = torch.mm(T_predicted, R_predicted)
                    
                    if count == 0:
                        pc_init = init_pcs[i]
                    else:
                        pc_init = pc_init
                    
                    pc_init = rotate_back(pc_init.type(torch.FloatTensor).cuda(), RT_predicted)
                    depth_img_pred, uv_pred, pc_pred_valid = lidar_project_depth(pc_init, sample['K'][0], real_shape_input[0]) # image_shape
                    depth_img_pred /= _config['max_depth']
                    
                    depth_pred = F.pad(depth_img_pred, shape_pad_input[0])
                    lidar = depth_pred.unsqueeze(0)
                    lidar_inp = F.interpolate(lidar.type(torch.FloatTensor), size=[256, 512], mode="bilinear", align_corners=False)
                    lidar_inp = lidar_inp.to(device)
                    count += 1

                save_pointcloud(f"final_pred_{i}", pc_init, [1,0,0]) # save each random pointcloud after running through all models                
      

if __name__ == '__main__':
    main(_config)
