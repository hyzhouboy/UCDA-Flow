from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import os.path as osp
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import skimage.io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
from aanet import AANet
from posenet import PoseNet
import evaluate
import datasets

from glob import glob
from tool import Logger, Project3D, BackprojectDepth, gemerate_haze, EMA
from loss import sequence_loss, disp_pyramid_loss, motion_consis_loss, cross_domain_consis_loss, self_supervised_loss, loss_KL_div

from utils.utils import transformation_from_parameters, disp_to_depth, filter_base_params, filter_specific_params
from PIL import Image
from utils import transforms

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    print("dummy GradScaler for PyTorch < 1.6")
    class GradScaler:
        def __init__(self, enabled=True):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
VAL_FREQ = 5000

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler



class Model(object):
    def __init__(self, args):
        self.args = args
        # 内参
        K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        K[0, :] *= self.args.camera_size[0]
        K[1, :] *= self.args.camera_size[1]
        inv_K = np.linalg.pinv(K)

        self.K_ = torch.from_numpy(K).unsqueeze(dim=0)
        self.K = torch.cat([self.K_, self.K_], 0).cuda()
        
        self.inv_K_ = torch.from_numpy(inv_K).unsqueeze(dim=0)
        self.inv_K = torch.cat([self.inv_K_, self.inv_K_], 0).cuda()
        # print()

    # train depth and flow network
    def stage_1_train(self):
        # dataset_loader
        train_loader = datasets.fetch_dataloader(self.args)
        # train_loader = datasets.fetch_clean_dataloader(self.args)
        
        # param setting
        # raft
        model_flow = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow))

        if self.args.restore_flow_ckpt is not None:
            model_flow.load_state_dict(torch.load(self.args.restore_flow_ckpt), strict=False)

        model_flow.cuda()
        model_flow.train()

        # aanet
        model_depth = AANet(self.args)
        print("Parameter Count: %d" % count_parameters(model_depth))
       
        if self.args.restore_disp_ckpt is not None:
            model_depth.load_state_dict(torch.load(self.args.restore_disp_ckpt), strict=False)
            
        model_depth = nn.DataParallel(model_depth, device_ids=self.args.gpus)
        
        model_depth.cuda()
        model_depth.train()

        if self.args.stage != 'chairs':
            model_flow.module.freeze_bn()

        # training process
        flow_optimizer = optim.AdamW(model_flow.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)
        flow_scheduler = optim.lr_scheduler.OneCycleLR(flow_optimizer, self.args.lr, self.args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')


        # disp optimizer
        specific_params = list(filter(filter_specific_params,
                                  model_depth.named_parameters()))
        base_params = list(filter(filter_base_params,
                                model_depth.named_parameters()))
        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = self.args.lr * 0.1
        milestones = [400, 600, 800, 900]
        params_group = [
            {'params': base_params, 'lr': self.args.lr},
            {'params': specific_params, 'lr': specific_lr},
        ]
        depth_optimizer = optim.Adam(params_group, weight_decay=self.args.wdecay*10)
        depth_scheduler = optim.lr_scheduler.MultiStepLR(depth_optimizer, milestones=milestones, gamma=0.5, last_epoch=-1)

        # depth_scheduler = optim.lr_scheduler.OneCycleLR(depth_optimizer, self.args.lr, self.args.num_steps+100,
        #     pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
        # optimizer, scheduler = fetch_optimizer(self.args, model_flow)

        total_steps = 0
        flow_scaler = GradScaler(enabled=self.args.mixed_precision)
        depth_scaler = GradScaler(enabled=self.args.mixed_precision)
        logger = Logger('checkpoints/', model_flow, flow_scheduler)

        VAL_FREQ = 5000
        VAL_SUMMARY_FREQ = 500

        add_noise = False
        should_keep_training = True

        adaptive_weight = 0.0

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                flow_optimizer.zero_grad()
                depth_optimizer.zero_grad()
                image1_left = data_blob['left_1'].cuda()
                image2_left = data_blob['left_2'].cuda()
                image1_right = data_blob['right_1'].cuda()
                image2_right = data_blob['right_2'].cuda()
                flow = data_blob['flow'].cuda()
                valid = data_blob['valid'].cuda()
                disp = data_blob['disp'].cuda()
                # image1, image2, flow, valid, disp  = [x.cuda() for x in data_blob]

                disp_mask = (disp > 0) & (disp < self.args.max_disp)

                if self.args.load_pseudo_gt:
                    pseudo_gt_disp = data_blob['pseudo_disp'].cuda()
                    pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < self.args.max_disp) & (~disp_mask)  # inverse mask

                if not disp_mask.any():
                    continue

                if add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1_left = (image1_left + stdv * torch.randn(*image1_left.shape).cuda()).clamp(0.0, 255.0)
                    image2_left = (image2_left + stdv * torch.randn(*image2_left.shape).cuda()).clamp(0.0, 255.0)
                    image1_right = (image1_right + stdv * torch.randn(*image1_right.shape).cuda()).clamp(0.0, 255.0)
                    image2_right = (image2_right + stdv * torch.randn(*image2_right.shape).cuda()).clamp(0.0, 255.0)

                _, flow_predictions = model_flow(image1_left, image2_left, iters=self.args.iters)      
                depth_predictions = model_depth(image1_left, image1_right)  # list of H/12, H/6, H/3, H/2, H      

                flow_loss, flow_metrics = sequence_loss(flow_predictions, flow, valid, self.args.gamma)
                if self.args.load_pseudo_gt:
                    disp_loss, disp_metrics = disp_pyramid_loss(depth_predictions, disp, disp_mask, pseudo_gt_disp, pseudo_mask, self.args.load_pseudo_gt)
                else:
                    disp_loss, disp_metrics = disp_pyramid_loss(depth_predictions, disp, disp_mask, disp, disp_mask, self.args.load_pseudo_gt)

                # flow network update
                flow_scaler.scale(flow_loss).backward()
                flow_scaler.unscale_(flow_optimizer)                
                torch.nn.utils.clip_grad_norm_(model_flow.parameters(), self.args.clip)
                
                flow_scaler.step(flow_optimizer)
                flow_scheduler.step()
                flow_scaler.update()

                # disp network update 
                depth_scaler.scale(adaptive_weight * disp_loss).backward()
                depth_scaler.unscale_(depth_optimizer)                
                # torch.nn.utils.clip_grad_norm_(model_flow.parameters(), self.args.clip)
                
                depth_scaler.step(depth_optimizer)
                depth_scheduler.step()
                depth_scaler.update()
                
                total_steps += 1
                
                # print info
                dict_metric = dict(flow_metrics, **disp_metrics)
                logger.push(dict_metric)
                # logger.push(disp_metrics)

                # test
                # disp = depth_predictions[-1][-1].detach().cpu().numpy()  # [H, W]
                # skimage.io.imsave('pred_disp/' + str(total_steps) + '.png', (disp * 256.).astype(np.uint16))

                if total_steps % VAL_SUMMARY_FREQ == 0:
                    img_summary = dict()
                    img_summary['left_1'] = image1_left
                    img_summary['right_1'] = image1_right
                    img_summary['gt_flow'] = flow
                    img_summary['pred_flow'] = flow_predictions[-1]
                    img_summary['gt_disp'] = disp
                    img_summary['pred_disp'] = depth_predictions[-1]
                    logger.save_image('train', img_summary)

                    
                if total_steps % VAL_FREQ == 0:
                    FLOW_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'raft')
                    torch.save(model_flow.state_dict(), FLOW_PATH)

                    DISP_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'aanet')
                    torch.save(model_depth.state_dict(), DISP_PATH)

                    
                    # results = {}
                    # for val_dataset in self.args.validation:
                    #     if val_dataset == 'chairs':
                    #         results.update(evaluate.validate_chairs(model_flow.module))
                    #     elif val_dataset == 'sintel':
                    #         results.update(evaluate.validate_sintel(model_flow.module))
                    #     elif val_dataset == 'kitti':
                    #         results.update(evaluate.validate_kitti(model_flow.module))

                    # logger.write_dict(results)
                    
                    # model_flow.train()
                    # if self.args.stage != 'chairs':
                    #     model_flow.module.freeze_bn()

                if total_steps > self.args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        FLOW_PATH = 'checkpoints/%s.pth' % 'raft'
        DISP_PATH = 'checkpoints/%s.pth' % 'aanet'
        torch.save(model_flow.state_dict(), FLOW_PATH)
        torch.save(model_depth.state_dict(), DISP_PATH)

        return FLOW_PATH, DISP_PATH


    # train depth, flow network and ego-motion
    def stage_2_train(self):
        fx = 721.53
        baseline = 53.72 
        # dataset_loader
        train_loader = datasets.fetch_dataloader(self.args)
        # train_loader = datasets.fetch_clean_dataloader(self.args)
        
        # param setting
        # raft
        model_flow = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow))

        # load model
        if self.args.restore_flow_ckpt is not None:
            model_flow.load_state_dict(torch.load(self.args.restore_flow_ckpt), strict=False)
            
        # aanet
        model_depth = AANet(self.args)
        print("Parameter Count: %d" % count_parameters(model_depth))
       
        if self.args.restore_disp_ckpt is not None:
            model_depth.load_state_dict(torch.load(self.args.restore_disp_ckpt), strict=False)
            
        model_depth = nn.DataParallel(model_depth, device_ids=self.args.gpus)
        
        
        # posenet
        model_pose = {}
        model_pose['encoder'] = PoseNet(self.args)['pose_encoder']
        model_pose['decoder'] = PoseNet(self.args)['pose_decoder']
        model_pose['encoder'].load_state_dict(torch.load(self.args.restore_pose_encoder_ckpt), strict=False)
        model_pose['decoder'].load_state_dict(torch.load(self.args.restore_pose_decoder_ckpt), strict=False)
        print("Parameter Count: %d" % count_parameters(model_pose['encoder']))
        

        model_flow.cuda()
        model_flow.train()

        model_depth.cuda()
        model_depth.train()

        model_pose['encoder'].cuda()
        model_pose['encoder'].eval()
        model_pose['decoder'].cuda()
        model_pose['decoder'].eval()

        self.backproject_depth = BackprojectDepth(self.args.batch_size, self.args.image_size[0], self.args.image_size[1])
        self.backproject_depth.cuda()
        self.project_3d = Project3D(self.args.batch_size, self.args.image_size[0], self.args.image_size[1])
        self.project_3d.cuda()

        if self.args.stage != 'chairs':
            model_flow.module.freeze_bn()

        # training process
        flow_optimizer = optim.AdamW(model_flow.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)
        flow_scheduler = optim.lr_scheduler.OneCycleLR(flow_optimizer, self.args.lr, self.args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')



        # disp optimizer
        specific_params = list(filter(filter_specific_params,
                                  model_depth.named_parameters()))
        base_params = list(filter(filter_base_params,
                                model_depth.named_parameters()))
        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        specific_lr = self.args.lr * 0.1
        milestones = [400, 600, 800, 900]
        params_group = [
            {'params': base_params, 'lr': self.args.lr},
            {'params': specific_params, 'lr': specific_lr},
        ]
        depth_optimizer = optim.Adam(params_group, weight_decay=self.args.wdecay*10)
        depth_scheduler = optim.lr_scheduler.MultiStepLR(depth_optimizer, milestones=milestones, gamma=0.5, last_epoch=-1)


        # depth_optimizer = optim.Adam(model_depth.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)
        # depth_scheduler = optim.lr_scheduler.OneCycleLR(depth_optimizer, self.args.lr, self.args.num_steps+100,
        #     pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
        # optimizer, scheduler = fetch_optimizer(self.args, model_flow)

        total_steps = 0
        flow_scaler = GradScaler(enabled=self.args.mixed_precision)
        depth_scaler = GradScaler(enabled=self.args.mixed_precision)
        logger = Logger('checkpoints/', model_flow, flow_scheduler)

        VAL_FREQ = 5000
        VAL_SUMMARY_FREQ = 500

        add_noise = False
        should_keep_training = True

        adaptive_weight = 0.0

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                flow_optimizer.zero_grad()
                depth_optimizer.zero_grad()
                image1_left = data_blob['left_1'].cuda()
                image2_left = data_blob['left_2'].cuda()
                image1_right = data_blob['right_1'].cuda()
                image2_right = data_blob['right_2'].cuda()
                flow = data_blob['flow'].cuda()
                valid = data_blob['valid'].cuda()
                disp = data_blob['disp'].cuda()
                # image1, image2, flow, valid, disp  = [x.cuda() for x in data_blob]

                disp_mask = (disp > 0) & (disp < self.args.max_disp)

                if self.args.load_pseudo_gt:
                    pseudo_gt_disp = data_blob['pseudo_disp'].cuda()
                    pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < self.args.max_disp) & (~disp_mask)  # inverse mask

                if not disp_mask.any():
                    continue

                if add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1_left = (image1_left + stdv * torch.randn(*image1_left.shape).cuda()).clamp(0.0, 255.0)
                    image2_left = (image2_left + stdv * torch.randn(*image2_left.shape).cuda()).clamp(0.0, 255.0)
                    image1_right = (image1_right + stdv * torch.randn(*image1_right.shape).cuda()).clamp(0.0, 255.0)
                    image2_right = (image2_right + stdv * torch.randn(*image2_right.shape).cuda()).clamp(0.0, 255.0)

                _, flow_predictions = model_flow(image1_left, image2_left, iters=self.args.iters)      
                depth_predictions = model_depth(image1_left, image1_right)  # list of H/12, H/6, H/3, H/2, H      
                
                # 计算pose，不参与训练
                with torch.no_grad():
                    adjacent_images = torch.cat([image1_left, image2_left], 1)
                    features = [model_pose['encoder'](adjacent_images)]
                    axisangle, translation = model_pose['decoder'](features)

                    # pred_poses = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy()
                    pred_poses = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                    # 3D 几何投影
                    disp_ = (depth_predictions[-1]*256.0).astype(np.uint16)
                    depth = 1 / disp_ * fx * baseline
                    # _, depth = disp_to_depth(depth_predictions[-1], self.args.min_depth, self.args.max_depth)
                    cam_points, src_pix_coords = self.backproject_depth(depth, self.inv_K)
                    tgt_pix_coords = self.project_3d(cam_points, self.K, pred_poses)
                    rigid_flow_predictions = tgt_pix_coords.cuda() - src_pix_coords.cuda()
                    rigid_flow_predictions = rigid_flow_predictions.permute(0, 3, 1, 2)
                    

                flow_loss, flow_metrics = sequence_loss(flow_predictions, flow, valid, self.args.gamma)
                if self.args.load_pseudo_gt:
                    disp_loss, disp_metrics = disp_pyramid_loss(depth_predictions, disp, disp_mask, pseudo_gt_disp, pseudo_mask, self.args.load_pseudo_gt)
                else:
                    disp_loss, disp_metrics = disp_pyramid_loss(depth_predictions, disp, disp_mask, disp, disp_mask, self.args.load_pseudo_gt)

                # 多模运动几何运动一致性损失约束
                geo_motion_loss, motion_metrics = motion_consis_loss(flow_predictions, rigid_flow_predictions, valid)
                # geo_motion_loss = 0.01 * geo_motion_loss

                # flow network update
                flow_loss = flow_loss + geo_motion_loss
                
                flow_scaler.scale(flow_loss).backward()
                flow_scaler.unscale_(flow_optimizer)                
                torch.nn.utils.clip_grad_norm_(model_flow.parameters(), self.args.clip)
                
                flow_scaler.step(flow_optimizer)
                flow_scheduler.step()
                flow_scaler.update()

                # disp network update 
                depth_scaler.scale(adaptive_weight*disp_loss).backward()
                depth_scaler.unscale_(depth_optimizer)                
                # torch.nn.utils.clip_grad_norm_(model_flow.parameters(), self.args.clip)
                
                depth_scaler.step(depth_optimizer)
                depth_scheduler.step()
                depth_scaler.update()


                total_steps += 1
                
                # print info
                dict_metric = dict(flow_metrics, **disp_metrics)
                dict_metric = dict(dict_metric, **motion_metrics)
                logger.push(dict_metric)
                # logger.push(disp_metrics)

                if total_steps % VAL_SUMMARY_FREQ == 0:
                    img_summary = dict()
                    img_summary['left_1'] = image1_left
                    img_summary['right_1'] = image1_right
                    img_summary['gt_flow'] = flow
                    img_summary['pred_flow'] = flow_predictions[-1]
                    img_summary['gt_disp'] = disp
                    img_summary['pred_disp'] = depth_predictions[-1]
                    img_summary['pred_rigid_flow'] = rigid_flow_predictions
                    logger.save_image('train', img_summary)

                    
                if total_steps % VAL_FREQ == 0:
                    FLOW_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'raft')
                    torch.save(model_flow.state_dict(), FLOW_PATH)

                    DISP_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'aanet')
                    torch.save(model_depth.state_dict(), DISP_PATH)

                    
                    # results = {}
                    # for val_dataset in self.args.validation:
                    #     if val_dataset == 'chairs':
                    #         results.update(evaluate.validate_chairs(model_flow.module))
                    #     elif val_dataset == 'sintel':
                    #         results.update(evaluate.validate_sintel(model_flow.module))
                    #     elif val_dataset == 'kitti':
                    #         results.update(evaluate.validate_kitti(model_flow.module))

                    # logger.write_dict(results)
                    
                    # model_flow.train()
                    # if self.args.stage != 'chairs':
                    #     model_flow.module.freeze_bn()

                if total_steps > self.args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        FLOW_PATH = 'checkpoints/%s.pth' % 'raft'
        DISP_PATH = 'checkpoints/%s.pth' % 'aanet'
        torch.save(model_flow.state_dict(), FLOW_PATH)
        torch.save(model_depth.state_dict(), DISP_PATH)

        return FLOW_PATH, DISP_PATH
    

    def generate_depth_foggy_images(self):
        if not os.path.isdir('generate_images'):
            os.mkdir('generate_images')
        
        if not os.path.isdir('pred_depth'):
            os.mkdir('pred_depth')
        
        if not os.path.isdir('pred_disp'):
            os.mkdir('pred_disp')
        
        fx = 721.53
        baseline = 53.72 # cm
        
        k = 0.88  #atmospheric
        beta = 0.06   #attenuation factor

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


        # all_samples = sorted(glob(self.args.data_dir + '/training/image_2/*.png'))
        all_samples = sorted(glob(self.args.data_dir + '/image_2/*.png'))
        num_samples = len(all_samples)
        print('=> %d samples found in the data dir' % num_samples)

        # param setting
        # aanet
        model_disp = AANet(self.args)
        print("Parameter Count: %d" % count_parameters(model_disp))
       
        if self.args.restore_disp_ckpt is not None:
            model_disp.load_state_dict(torch.load(self.args.restore_disp_ckpt), strict=False)
            
        model_disp = nn.DataParallel(model_disp, device_ids=self.args.gpus)

        model_disp.cuda()
        model_disp.eval()

        for i, sample_name in enumerate(all_samples):
            if i % 100 == 0:
                print('=> Inferencing %d/%d' % (i, num_samples))
            
            left_name = sample_name
            right_name = left_name.replace('image_2', 'image_3')

            left = np.array(Image.open(left_name).convert('RGB')).astype(np.float32)
            right = np.array(Image.open(right_name).convert('RGB')).astype(np.float32)

            temp_left = cv2.imread(left_name)

            sample = {'left': left,
                  'right': right}

            sample = test_transform(sample)  # to tensor and normalize

            left = sample['left'].cuda()  # [3, H, W]
            left = left.unsqueeze(0)  # [1, 3, H, W]
            right = sample['right'].cuda()
            right = right.unsqueeze(0)

            ori_height, ori_width = left.size()[2:]

            # Automatic
            factor = 48
            img_height = math.ceil(ori_height / factor) * factor
            img_width = math.ceil(ori_width / factor) * factor

            if ori_height < img_height or ori_width < img_width:
                top_pad = img_height - ori_height
                right_pad = img_width - ori_width

                # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
                left = F.pad(left, (0, right_pad, top_pad, 0))
                right = F.pad(right, (0, right_pad, top_pad, 0))
            
            with torch.no_grad():
                pred_disp = model_disp(left, right)[-1]
            
            
            
            if pred_disp.size(-1) < left.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
                                        mode='bilinear') * (left.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]
            
            # Crop
            if ori_height < img_height or ori_width < img_width:
                if right_pad != 0:
                    pred_disp = pred_disp[:, top_pad:, :-right_pad]
                else:
                    pred_disp = pred_disp[:, top_pad:]
            
            disp = pred_disp[0].detach().cpu().numpy()  # [H, W]
            saved_disp_name = 'pred_disp/' + os.path.basename(left_name)
            disp = (disp * 256.).astype(np.uint16)
            skimage.io.imsave(saved_disp_name, disp)
            
            saved_depth_name = 'pred_depth/' + os.path.basename(left_name)
            depth = 1/disp * fx * baseline
            im_color=cv2.applyColorMap(cv2.convertScaleAbs(depth,alpha=15),cv2.COLORMAP_BONE)
            im=Image.fromarray(im_color)
            im.save(saved_depth_name)

            saved_foggy_name = 'generate_images/' + os.path.basename(left_name)
            fog = gemerate_haze(temp_left, depth, k, beta)
            cv2.waitKey(3)
            cv2.imwrite(saved_foggy_name, fog)

   
    # dataset: kitti_foggy
    # using domain adaptation to train flow network of degraded domain
    def stage_4_train(self):
        # dataset_loader
        train_loader = datasets.fetch_dataloader(self.args)
        # train_loader = datasets.fetch_clean_dataloader(self.args)
        
        # param setting
        # raft-clean
        model_flow_clean = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow_clean))
        # raft-foggy
        model_flow_foggy = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow_foggy))


        # load model
        if self.args.restore_flow_ckpt is not None:
            model_flow_clean.load_state_dict(torch.load(self.args.restore_flow_ckpt), strict=False)

        if self.args.restore_flow_synimg_ckpt is not None:
            model_flow_foggy.load_state_dict(torch.load(self.args.restore_flow_synimg_ckpt), strict=False)
             

        model_flow_clean.cuda()
        model_flow_clean.eval()

        model_flow_foggy.cuda()
        model_flow_foggy.train()


        if self.args.stage != 'chairs':
            model_flow_foggy.module.freeze_bn()

        # training process
        flow_optimizer = optim.AdamW(model_flow_foggy.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)
        flow_scheduler = optim.lr_scheduler.OneCycleLR(flow_optimizer, self.args.lr, self.args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

        total_steps = 0
        flow_scaler = GradScaler(enabled=self.args.mixed_precision)
        logger = Logger('checkpoints/', model_flow_foggy, flow_scheduler)

        VAL_FREQ = 5000
        VAL_SUMMARY_FREQ = 500

        add_noise = False
        should_keep_training = True

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                flow_optimizer.zero_grad()
                image1_clean = data_blob['clean_1'].cuda()
                image2_clean = data_blob['clean_2'].cuda()
                image1_foggy = data_blob['foggy_1'].cuda()
                image2_foggy = data_blob['foggy_2'].cuda()
                flow = data_blob['flow'].cuda()
                valid = data_blob['valid'].cuda()
                # image1, image2, flow, valid, disp  = [x.cuda() for x in data_blob]

                # disp_mask = (disp > 0) & (disp < self.args.max_disp)

                if add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1_clean = (image1_clean + stdv * torch.randn(*image1_clean.shape).cuda()).clamp(0.0, 255.0)
                    image2_clean = (image2_clean + stdv * torch.randn(*image2_clean.shape).cuda()).clamp(0.0, 255.0)
                    image1_foggy = (image1_foggy + stdv * torch.randn(*image1_foggy.shape).cuda()).clamp(0.0, 255.0)
                    image2_foggy = (image2_foggy + stdv * torch.randn(*image2_foggy.shape).cuda()).clamp(0.0, 255.0)
                
                with torch.no_grad():
                    _, flow_clean_predictions = model_flow_clean(image1_clean, image2_clean, iters=20, test_mode=True)
               
                # 计算浓雾天图像光流
                _, flow_foggy_predictions = model_flow_foggy(image1_foggy, image2_foggy, iters=self.args.iters)     
                
                flow_loss, flow_metrics = sequence_loss(flow_foggy_predictions, flow, valid, self.args.gamma)
                flow_DA_loss, flow_DA_metrics = cross_domain_consis_loss(flow_foggy_predictions[-1], flow_clean_predictions)
                    
                # flow network update
                flow_loss = flow_loss + flow_DA_loss
                
                flow_scaler.scale(flow_loss).backward()
                flow_scaler.unscale_(flow_optimizer)                
                torch.nn.utils.clip_grad_norm_(model_flow_foggy.parameters(), self.args.clip)
                
                flow_scaler.step(flow_optimizer)
                flow_scheduler.step()
                flow_scaler.update()

                total_steps += 1
                
                # print info
                dict_metric = dict(flow_metrics, **flow_DA_metrics)
                logger.push(dict_metric)

                if total_steps % VAL_SUMMARY_FREQ == 0:
                    img_summary = dict()
                    img_summary['clean_1'] = image1_clean
                    img_summary['foggy_1'] = image1_foggy
                    img_summary['gt_flow'] = flow
                    img_summary['pred_flow'] = flow_foggy_predictions[-1]
                    logger.save_image('train', img_summary)

                    
                if total_steps % VAL_FREQ == 0:
                    FLOW_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'raft_synfog')
                    torch.save(model_flow_foggy.state_dict(), FLOW_PATH)

                if total_steps > self.args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        FLOW_PATH = 'checkpoints/%s.pth' % 'raft_synfog'
        torch.save(model_flow_foggy.state_dict(), FLOW_PATH)

        return FLOW_PATH
       

    
    # dataset: foggy-kitti and real images
    # using pseudo stategy to train flow network of real degraded domain
    # train temporal cost volume, pseudo and monunent update.
    def stage_5_train(self):
        # dataset_loader
        train_loader = datasets.fetch_dataloader(self.args)
        # train_loader = datasets.fetch_clean_dataloader(self.args)
        
        # param setting
        # raft-foggy
        model_flow_syn = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow_syn))
        # raft-foggy
        model_flow_real = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow_real))


        # load model
        if self.args.restore_flow_synimg_ckpt is not None:
            model_flow_syn.load_state_dict(torch.load(self.args.restore_flow_synimg_ckpt), strict=False)
             
             
        if self.args.restore_flow_realimg_ckpt is not None:
            model_flow_real.load_state_dict(torch.load(self.args.restore_flow_realimg_ckpt), strict=False)

        model_flow_syn.cuda()
        model_flow_syn.train()

        model_flow_real.cuda()
        model_flow_real.train()


        if self.args.stage != 'chairs':
            model_flow_syn.module.freeze_bn()
            model_flow_real.module.freeze_bn()

        # training process
        params_group = [
            {'params': model_flow_syn.parameters()},
            {'params': model_flow_real.parameters()}
        ]

        flow_optimizer = optim.AdamW(params_group, lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)
        flow_scheduler = optim.lr_scheduler.OneCycleLR(flow_optimizer, self.args.lr, self.args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

        total_steps = 0
        flow_scaler = GradScaler(enabled=self.args.mixed_precision)
        logger = Logger('checkpoints/', model_flow_real, flow_scheduler)

        VAL_FREQ = 5000
        VAL_SUMMARY_FREQ = 500

        add_noise = False
        should_keep_training = True

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                flow_optimizer.zero_grad()
                image1_syn = data_blob['syn_1'].cuda()
                image2_syn = data_blob['syn_2'].cuda()
                image1_real = data_blob['real_1'].cuda()
                image2_real = data_blob['real_2'].cuda()
                flow = data_blob['flow'].cuda()
                valid = data_blob['valid'].cuda()
                # image1, image2, flow, valid, disp  = [x.cuda() for x in data_blob]

                # disp_mask = (disp > 0) & (disp < self.args.max_disp)

                if add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1_syn = (image1_syn + stdv * torch.randn(*image1_syn.shape).cuda()).clamp(0.0, 255.0)
                    image2_syn = (image2_syn + stdv * torch.randn(*image2_syn.shape).cuda()).clamp(0.0, 255.0)
                    image1_real = (image1_real + stdv * torch.randn(*image1_real.shape).cuda()).clamp(0.0, 255.0)
                    image2_real = (image2_real + stdv * torch.randn(*image2_real.shape).cuda()).clamp(0.0, 255.0)
                
                # with torch.no_grad():
                _, flow_syn_predictions = model_flow_syn(image1_syn, image2_syn, iters=self.args.iters)

                # 计算真实浓雾天图像光流
                with torch.no_grad():
                    _, pseudo_flow_predictions = model_flow_syn(image1_real, image2_real, iters=self.args.iters)

                _, flow_real_predictions = model_flow_real(image1_real, image2_real, iters=self.args.iters)     
                
                

                # 损失：动量更新、epe、kl散度、伪标签
                # epe between gt and syn flow
                syn_flow_loss, syn_flow_metrics = sequence_loss(flow_syn_predictions, flow, valid, self.args.gamma)

                # self-supervised pseduo loss
                pseudo_loss, pseudo_metrics = self_supervised_loss(flow_real_predictions[-1], pseudo_flow_predictions[-1])

                # kl散度
                # flow_DA_loss, flow_DA_metrics = cross_domain_consis_loss(flow_foggy_predictions[-1], flow_clean_predictions)
                
                # flow network update
                flow_loss = syn_flow_loss + pseudo_loss
                
                flow_scaler.scale(flow_loss).backward()
                flow_scaler.unscale_(flow_optimizer)                
                torch.nn.utils.clip_grad_norm_(model_flow_syn.parameters(), self.args.clip)
                torch.nn.utils.clip_grad_norm_(model_flow_real.parameters(), self.args.clip)
                
                flow_scaler.step(flow_optimizer)
                flow_scheduler.step()
                flow_scaler.update()

                total_steps += 1
                
                # print info
                dict_metric = dict(syn_flow_metrics, **pseudo_metrics)
                logger.push(dict_metric)

                if total_steps % VAL_SUMMARY_FREQ == 0:
                    img_summary = dict()
                    img_summary['clean_1'] = image1_syn
                    img_summary['foggy_1'] = image1_real
                    img_summary['pred_syn_flow'] = flow_syn_predictions[-1]
                    img_summary['pred_real_flow'] = flow_real_predictions[-1]
                    logger.save_image('train', img_summary)

                    
                if total_steps % VAL_FREQ == 0:
                    SYN_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'raft_synfog')
                    torch.save(model_flow_syn.state_dict(), SYN_PATH)
                
                if total_steps % VAL_FREQ == 0:
                    REAL_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'raft_realfog')
                    torch.save(model_flow_real.state_dict(), REAL_PATH)

                if total_steps > self.args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        SYN_PATH = 'checkpoints/%s.pth' % 'raft_synfog'
        torch.save(model_flow_syn.state_dict(), SYN_PATH)

        REAL_PATH = 'checkpoints/%s.pth' % 'raft_realfog'
        torch.save(model_flow_real.state_dict(), REAL_PATH)

        return SYN_PATH, REAL_PATH





    # dataset: foggy-kitti and real images
    # using pseudo stategy to train flow network of real degraded domain
    # train temporal cost volume, pseudo and monunent update, kl散度.
    def stage_6_train(self):
        # dataset_loader
        train_loader = datasets.fetch_dataloader(self.args)
        # train_loader = datasets.fetch_clean_dataloader(self.args)
        
        # param setting
        # raft-foggy
        model_flow_syn = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow_syn))
        # raft-foggy
        model_flow_real = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow_real))


        # load model
        if self.args.restore_flow_synimg_ckpt is not None:
            model_flow_syn.load_state_dict(torch.load(self.args.restore_flow_synimg_ckpt), strict=False)
             
             
        if self.args.restore_flow_realimg_ckpt is not None:
            model_flow_real.load_state_dict(torch.load(self.args.restore_flow_realimg_ckpt), strict=False)

        model_flow_syn.cuda()
        model_flow_syn.train()

        model_flow_real.cuda()
        model_flow_real.train()


        if self.args.stage != 'chairs':
            model_flow_syn.module.freeze_bn()
            model_flow_real.module.freeze_bn()
        
        model_ema = EMA(model_flow_real, 0.999)
        model_ema.register()
        # training process
        params_group = [
            {'params': model_flow_syn.parameters()},
            {'params': model_flow_real.parameters()}
        ]

        flow_optimizer = optim.AdamW(params_group, lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)
        flow_scheduler = optim.lr_scheduler.OneCycleLR(flow_optimizer, self.args.lr, self.args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

        total_steps = 0
        flow_scaler = GradScaler(enabled=self.args.mixed_precision)
        logger = Logger('checkpoints/', model_flow_real, flow_scheduler)

        VAL_FREQ = 5000
        VAL_SUMMARY_FREQ = 500

        add_noise = False
        should_keep_training = True

        # 定义KL函数
        # loss_kl_cost = nn.KLDivLoss(size_average=True, reduce=True)

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                flow_optimizer.zero_grad()
                image1_syn = data_blob['syn_1'].cuda()
                image2_syn = data_blob['syn_2'].cuda()
                image1_real = data_blob['real_1'].cuda()
                image2_real = data_blob['real_2'].cuda()
                flow = data_blob['flow'].cuda()
                valid = data_blob['valid'].cuda()
                # image1, image2, flow, valid, disp  = [x.cuda() for x in data_blob]

                # disp_mask = (disp > 0) & (disp < self.args.max_disp)

                if add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1_syn = (image1_syn + stdv * torch.randn(*image1_syn.shape).cuda()).clamp(0.0, 255.0)
                    image2_syn = (image2_syn + stdv * torch.randn(*image2_syn.shape).cuda()).clamp(0.0, 255.0)
                    image1_real = (image1_real + stdv * torch.randn(*image1_real.shape).cuda()).clamp(0.0, 255.0)
                    image2_real = (image2_real + stdv * torch.randn(*image2_real.shape).cuda()).clamp(0.0, 255.0)
                
                # with torch.no_grad():
                cost_volume_syn, flow_syn_predictions = model_flow_syn(image1_syn, image2_syn, iters=self.args.iters)

                # 计算真实浓雾天图像光流
                with torch.no_grad():
                    _, pseudo_flow_predictions = model_flow_syn(image1_real, image2_real, iters=self.args.iters)

                cost_volume_real, flow_real_predictions = model_flow_real(image1_real, image2_real, iters=self.args.iters)     
                
                

                # 损失：动量更新、epe、kl散度、伪标签
                # epe between gt and syn flow
                syn_flow_loss, syn_flow_metrics = sequence_loss(flow_syn_predictions, flow, valid, self.args.gamma)

                # self-supervised pseduo loss
                pseudo_loss, pseudo_metrics = self_supervised_loss(flow_real_predictions[-1], pseudo_flow_predictions[-1])
                
                # kl散度
                # 计算分布
                # print(cost_volume_syn[-1].shape)
                distribution_syn = torch.histc(cost_volume_syn[-1])
                # print(distribution_syn.shape)
                distribution_real = torch.histc(cost_volume_real[-1])
                # flow_DA_loss, flow_DA_metrics = cross_domain_consis_loss(flow_foggy_predictions[-1], flow_clean_predictions)
                loss_kl = 0.001 * loss_KL_div(distribution_real, distribution_syn)
                # print(loss_kl)
                kl_metrics = {
                    'kl_loss': loss_kl,
                }
                
                # flow network update
                flow_loss = syn_flow_loss + pseudo_loss + loss_kl
                
                flow_scaler.scale(flow_loss).backward()
                flow_scaler.unscale_(flow_optimizer)                
                torch.nn.utils.clip_grad_norm_(model_flow_syn.parameters(), self.args.clip)
                # torch.nn.utils.clip_grad_norm_(model_flow_real.parameters(), self.args.clip)
                
                flow_scaler.step(flow_optimizer)
                # EMA update
                model_ema.update()
                
                flow_scheduler.step()
                flow_scaler.update()

                total_steps += 1
                
                # print info
                dict_metric = dict(syn_flow_metrics, **pseudo_metrics)
                dict_metric = dict(dict_metric, **kl_metrics)
                logger.push(dict_metric)

                if total_steps % VAL_SUMMARY_FREQ == 0:
                    img_summary = dict()
                    img_summary['clean_1'] = image1_syn
                    img_summary['foggy_1'] = image1_real
                    img_summary['pred_syn_flow'] = flow_syn_predictions[-1]
                    img_summary['pred_real_flow'] = flow_real_predictions[-1]
                    logger.save_image('train', img_summary)

                    
                if total_steps % VAL_FREQ == 0:
                    SYN_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'raft_synfog')
                    torch.save(model_flow_syn.state_dict(), SYN_PATH)
                
                if total_steps % VAL_FREQ == 0:
                    model_ema.apply_shadow()
                    REAL_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'raft_realfog')
                    torch.save(model_flow_real.state_dict(), REAL_PATH)
                    model_ema.restore()

                if total_steps > self.args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        SYN_PATH = 'checkpoints/%s.pth' % 'raft_synfog'
        torch.save(model_flow_syn.state_dict(), SYN_PATH)
        
        model_ema.apply_shadow()
        REAL_PATH = 'checkpoints/%s.pth' % 'raft_realfog'
        torch.save(model_flow_real.state_dict(), REAL_PATH)

        return SYN_PATH, REAL_PATH
    


    
    # dataset: foggy-kitti and real images
    # using pseudo stategy to train flow network of real degraded domain
    # train temporal cost volume, pseudo and monunent update, kl散度.  
    # add spatial context feature
    def stage_7_train(self):
        # dataset_loader
        train_loader = datasets.fetch_dataloader(self.args)
        # train_loader = datasets.fetch_clean_dataloader(self.args)
        
        # param setting
        # raft-foggy
        model_flow_syn = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow_syn))
        # raft-foggy
        model_flow_real = nn.DataParallel(RAFT(self.args), device_ids=self.args.gpus)
        print("Parameter Count: %d" % count_parameters(model_flow_real))


        # load model
        if self.args.restore_flow_synimg_ckpt is not None:
            model_flow_syn.load_state_dict(torch.load(self.args.restore_flow_synimg_ckpt), strict=False)
             
             
        if self.args.restore_flow_realimg_ckpt is not None:
            model_flow_real.load_state_dict(torch.load(self.args.restore_flow_realimg_ckpt), strict=False)

        model_flow_syn.cuda()
        model_flow_syn.train()

        model_flow_real.cuda()
        model_flow_real.train()


        if self.args.stage != 'chairs':
            model_flow_syn.module.freeze_bn()
            model_flow_real.module.freeze_bn()
        
        model_ema = EMA(model_flow_real, 0.999)
        model_ema.register()
        # training process
        params_group = [
            {'params': model_flow_syn.parameters()},
            {'params': model_flow_real.parameters()}
        ]

        flow_optimizer = optim.AdamW(params_group, lr=self.args.lr, weight_decay=self.args.wdecay, eps=self.args.epsilon)
        flow_scheduler = optim.lr_scheduler.OneCycleLR(flow_optimizer, self.args.lr, self.args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

        total_steps = 0
        flow_scaler = GradScaler(enabled=self.args.mixed_precision)
        logger = Logger('checkpoints/', model_flow_real, flow_scheduler)

        VAL_FREQ = 5000
        VAL_SUMMARY_FREQ = 500

        add_noise = False
        should_keep_training = True

        # 定义KL函数
        # loss_kl_cost = nn.KLDivLoss(size_average=True, reduce=True)

        while should_keep_training:
            for i_batch, data_blob in enumerate(train_loader):
                flow_optimizer.zero_grad()
                image1_syn = data_blob['syn_1'].cuda()
                image2_syn = data_blob['syn_2'].cuda()
                image1_real = data_blob['real_1'].cuda()
                image2_real = data_blob['real_2'].cuda()
                flow = data_blob['flow'].cuda()
                valid = data_blob['valid'].cuda()
                # image1, image2, flow, valid, disp  = [x.cuda() for x in data_blob]

                # disp_mask = (disp > 0) & (disp < self.args.max_disp)

                if add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1_syn = (image1_syn + stdv * torch.randn(*image1_syn.shape).cuda()).clamp(0.0, 255.0)
                    image2_syn = (image2_syn + stdv * torch.randn(*image2_syn.shape).cuda()).clamp(0.0, 255.0)
                    image1_real = (image1_real + stdv * torch.randn(*image1_real.shape).cuda()).clamp(0.0, 255.0)
                    image2_real = (image2_real + stdv * torch.randn(*image2_real.shape).cuda()).clamp(0.0, 255.0)
                
                # with torch.no_grad():
                cost_volume_syn, flow_syn_predictions = model_flow_syn(image1_syn, image2_syn, iters=self.args.iters)

                # 计算真实浓雾天图像光流
                with torch.no_grad():
                    _, pseudo_flow_predictions = model_flow_syn(image1_real, image2_real, iters=self.args.iters)

                cost_volume_real, flow_real_predictions = model_flow_real(image1_real, image2_real, iters=self.args.iters)     
                
                

                # 损失：动量更新、epe、kl散度、伪标签
                # epe between gt and syn flow
                syn_flow_loss, syn_flow_metrics = sequence_loss(flow_syn_predictions, flow, valid, self.args.gamma)

                # self-supervised pseduo loss
                pseudo_loss, pseudo_metrics = self_supervised_loss(flow_real_predictions[-1], pseudo_flow_predictions[-1])
                
                # kl散度
                # 计算分布
                distribution_syn = torch.histc(cost_volume_syn[-1])
                # print(distribution_syn.shape)
                distribution_real = torch.histc(cost_volume_real[-1])
                # flow_DA_loss, flow_DA_metrics = cross_domain_consis_loss(flow_foggy_predictions[-1], flow_clean_predictions)
                # loss_kl = 0.1 * loss_kl_cost(distribution_real, distribution_syn)
                loss_kl = 0.001 * loss_KL_div(distribution_real, distribution_syn)
                kl_metrics = {
                    'kl_loss': loss_kl,
                }
                
                # flow network update
                flow_loss = syn_flow_loss + pseudo_loss + loss_kl
                
                flow_scaler.scale(flow_loss).backward()
                flow_scaler.unscale_(flow_optimizer)                
                torch.nn.utils.clip_grad_norm_(model_flow_syn.parameters(), self.args.clip)
                # torch.nn.utils.clip_grad_norm_(model_flow_real.parameters(), self.args.clip)
                
                flow_scaler.step(flow_optimizer)
                # EMA update
                model_ema.update()
                flow_scheduler.step()
                
                flow_scaler.update()

                total_steps += 1
                
                # print info
                dict_metric = dict(syn_flow_metrics, **pseudo_metrics)
                dict_metric = dict(dict_metric, **kl_metrics)
                logger.push(dict_metric)

                if total_steps % VAL_SUMMARY_FREQ == 0:
                    img_summary = dict()
                    img_summary['clean_1'] = image1_syn
                    img_summary['foggy_1'] = image1_real
                    img_summary['pred_syn_flow'] = flow_syn_predictions[-1]
                    img_summary['pred_real_flow'] = flow_real_predictions[-1]
                    logger.save_image('train', img_summary)

                    
                if total_steps % VAL_FREQ == 0:
                    SYN_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'ucda_flow_syn')
                    torch.save(model_flow_syn.state_dict(), SYN_PATH)
                
                if total_steps % VAL_FREQ == 0:
                    # 更新EMA策略更新权重保存模型
                    model_ema.apply_shadow()
                    REAL_PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, 'ucda_flow_realfog')
                    torch.save(model_flow_real.state_dict(), REAL_PATH)
                    model_ema.restore()

                if total_steps > self.args.num_steps:
                    should_keep_training = False
                    break

        logger.close()
        SYN_PATH = 'checkpoints/%s.pth' % 'ucda_flow_synfog'
        torch.save(model_flow_syn.state_dict(), SYN_PATH)

        model_ema.apply_shadow()
        REAL_PATH = 'checkpoints/%s.pth' % 'ucda_flow_realfog'
        torch.save(model_flow_real.state_dict(), REAL_PATH)

        return SYN_PATH, REAL_PATH