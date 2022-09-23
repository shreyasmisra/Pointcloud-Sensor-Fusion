import math
import os
import random
import time

import open3d as o3d

import mathutils
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from losses import DistancePoints3D, GeometricLoss, L1Loss, ProposedLoss, CombinedLoss
from models.LCCNet import LCCNet

from quaternion_distances import quaternion_distance

from utils import (mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)

#from dataset import IPadDataset        
from LabDataset import LabDataset


from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds


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

    return depth_img, pcl_uv

                 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

ex = Experiment("LCCNet")
ex.captured_out_filter = apply_backspaces_and_linefeeds

# do 2 deg and 1 deg

_config = {
    'checkpoints': './checkpoints/',
    'use_reflectance': False,
    'epochs':100,
    'BASE_LEARNING_RATE': 1e-5,
    'loss': 'combined',
    'dataset_num': 3, # for ipadDataset
    'max_t': 0.5,
    'max_r': 5.,
    'occlusion_kernel': 5,
    'occlusion_threshold': 3.0,
    'network': 'Res_f1',
    'optimizer': 'adam',
    'weights': './pretrained/kitti_iter3.tar',
    'rescale_rot': 1.0,
    'rescale_transl': 2.0,
    'resume': True,
    'precision': "O0",
    'norm': 'bn',
    'dropout': 0.0,
    'save_log': True,
    'dropout': 0.0,
    'max_depth': 2.,
    'weight_point_cloud': 0.6,
    'starting_epoch': 1,
}

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
EPOCH = 1


#model_20_1_5 = os.path.join("finetune", "modelsfinetune_rot_20_trans_1.5.pth")

@ex.capture
def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, loss_fn, point_clouds, loss):
    model.train()

    optimizer.zero_grad()

    # Run model

    transl_err, rot_err = model(rgb_img.cuda(), refl_img.cuda())


    if loss == 'points_distance' or loss == 'combined':
        losses = loss_fn(point_clouds.cuda(), target_transl[:,:,0].type(torch.FloatTensor).cuda(), target_rot.type(torch.FloatTensor).cuda(), transl_err, rot_err)
    else:
        losses = loss_fn(target_transl[:,:,0].type(torch.FloatTensor).cuda(), target_rot.type(torch.FloatTensor).cuda(), transl_err, rot_err)

    losses['total_loss'].backward()
    optimizer.step()

    return losses, rot_err, transl_err

def main(_config):
    global EPOCH
    print('Loss Function Choice: {}'.format(_config['loss']))
    
    img_shape = (1200, 1920) 
    input_size = (256, 512)
    

    model_savepath = os.path.join(_config['checkpoints'], 'finetune', 'models')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)   

    #dataset_class = IPadDataset(num=_config['dataset_num'], max_r=_config['max_r'], max_t=_config['max_t'], with_batch=4000)
    dataset_class = LabDataset(max_r=_config['max_r'], max_t=_config['max_t'], with_batch=4000)

    # Training and validation set creation
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_class,
                                                shuffle=True,
                                                batch_size=20,
                                                num_workers=1,
                                                drop_last=False,
                                                pin_memory=True)

    # loss function choice
    if _config['loss'] == 'simple':
        loss_fn = ProposedLoss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'geometric':
        loss_fn = GeometricLoss()
        loss_fn = loss_fn.cuda()
    elif _config['loss'] == 'points_distance':
        loss_fn = DistancePoints3D()
    elif _config['loss'] == 'L1':
        loss_fn = L1Loss(_config['rescale_transl'], _config['rescale_rot'])
    elif _config['loss'] == 'combined':
        loss_fn = CombinedLoss(_config['rescale_transl'], _config['rescale_rot'], _config['weight_point_cloud'])
    else:
        raise ValueError("Unknown Loss Function")

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
                         use_reflectance=_config['use_reflectance'], dropout=_config['dropout'],
                         Action_Func='leakyrelu', attention=False, res_num=18)
#        model_prev = LCCNet(input_size, use_feat_from=feat, md=md,
#                         use_reflectance=_config['use_reflectance'], dropout=_config['dropout'],
#                         Action_Func='leakyrelu', attention=False, res_num=18)
    else:
        raise TypeError("Network unknown")
    if _config['weights'] is not None:
        print(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        model.load_state_dict(saved_state_dict)
     
#    model_prev.load_state_dict(torch.load(model_20_1_5, map_location='cpu'))

    model = model.cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if _config['loss'] == 'geometric':
        parameters += list(loss_fn.parameters())
    if _config['optimizer'] == 'adam':
        optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=2e-6)
        # Probably this scheduler is not used
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[130], gamma=0.1)
    else:
        optimizer = optim.SGD(parameters, lr=_config['BASE_LEARNING_RATE'], momentum=0.9,
                              weight_decay=2e-6, nesterov=True)

    starting_epoch = _config['starting_epoch']
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        if starting_epoch != 0:
            starting_epoch = checkpoint['epoch']
            _config['epochs'] += starting_epoch

    # Allow mixed-precision if needed
    # model, optimizer = apex.amp.initialize(model, optimizer, opt_level=_config["precision"])

    start_full_time = time.time()
    old_save_filename = None

    train_iter = 0
    for epoch in range(starting_epoch, _config['epochs'] + 1):
        EPOCH = epoch
        print('This is %d-th epoch' % epoch)
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.

        ## Training ##
        time_for_50ep = time.time()
        for batch_idx, sample in enumerate(TrainImgLoader):

            start_time = time.time()
            lidar_input = []
            rgb_input = []
            lidar_gt = []
            shape_pad_input = []
            real_shape_input = []
            pc_rotated_input = []

            # gt pose
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            start_preprocess = time.time()
            for idx in range(sample['rgb'].shape[0]):
                real_shape = [sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2], sample['rgb'][idx].shape[0]]

                sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda() 
                pc_lidar = sample['point_cloud'][idx].clone()

                depth_gt, uv = lidar_project_depth(pc_lidar, sample['K'][idx], real_shape) # image_shape
                depth_gt /= _config['max_depth']

                #print(sample['rot_error'].shape)
                #R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
                #R.resize_4x4()
                #T = mathutils.Matrix.Translation(sample['tr_error'][idx])
                #RT = T * R
                #RT = sample['RT'][idx]

                #pc_rotated = rotate_back(sample['point_cloud'][idx], RT) # Pc` = RT * Pc
                pc_rotated = sample['init_pc'][idx].cuda()

                depth_img, uv = lidar_project_depth(pc_rotated, sample['K'][idx], real_shape) # image_shape
                depth_img /= _config['max_depth']

                # PAD ONLY ON RIGHT AND BOTTOM SIDE
                rgb = sample['rgb'][idx].cuda()
                shape_pad = [0, 0, 0, 0]

                shape_pad[3] = (img_shape[0] - rgb.shape[1])  # // 2
                shape_pad[1] = (img_shape[1] - rgb.shape[2])  # // 2 + 1

                rgb = F.pad(rgb, shape_pad)
                depth_img = F.pad(depth_img, shape_pad)
                depth_gt = F.pad(depth_gt, shape_pad)

                rgb_input.append(rgb)
                lidar_input.append(depth_img)
                lidar_gt.append(depth_gt)
                real_shape_input.append(real_shape)
                shape_pad_input.append(shape_pad)
                pc_rotated_input.append(pc_rotated)

            lidar_input = torch.stack(lidar_input).cuda()
            rgb_input = torch.stack(rgb_input)
            rgb_show = rgb_input.clone()
            lidar_show = lidar_input.clone()
            rgb_input = F.interpolate(rgb_input.type(torch.FloatTensor), size=[256, 512], mode="bilinear", align_corners=False)
            lidar_input = F.interpolate(lidar_input.type(torch.FloatTensor), size=[256, 512], mode="bilinear", align_corners=False)
            end_preprocess = time.time()
            
            loss, R_predicted,  T_predicted = train(model, optimizer, rgb_input, lidar_input,
                                                   sample['tr_error'], sample['rot_error'],
                                                   loss_fn, sample['point_cloud'], _config['loss'])

            for key in loss.keys():
                if loss[key].item() != loss[key].item():
                    raise ValueError("Loss {} is NaN".format(key))

            local_loss += loss['total_loss'].item()

            if batch_idx % 50 == 0 and batch_idx != 0:

                print(f'Iter {batch_idx}/{len(TrainImgLoader)} training loss = {local_loss/50:.3f}, '
                      f'time = {(time.time() - start_time)/lidar_input.shape[0]:.4f}, '
                      #f'time_preprocess = {(end_preprocess-start_preprocess)/lidar_input.shape[0]:.4f}, '
                      f'time for 50 iter: {time.time()-time_for_50ep:.4f}')
                time_for_50ep = time.time()
                local_loss = 0.
            total_train_loss += loss['total_loss'].item() * len(sample['rgb'])
            train_iter += 1
            # total_iter += len(sample['rgb'])

        print("------------------------------------")
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(dataset_class)))
        print('Total epoch time = %.2f' % (time.time() - epoch_start_time))
        print("------------------------------------")
    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    torch.save(model.state_dict(), model_savepath + f'/lab_sensors_finetune_rot_5_trans_0.5_{total_train_loss / len(dataset_class)}.pth') 

if __name__ == '__main__':
	main(_config)    
