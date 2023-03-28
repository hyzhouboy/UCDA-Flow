# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor, SparseFlowDispAugmentor, Pseudo_SparseFlowDispAugmentor, CrossDomainSparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

# 写这部分的数据读取
class StereoFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=True, load_pseudo_gt=True):
        self.augmentor = None
        self.sparse = sparse
        self.load_pseudo_gt = load_pseudo_gt
        
        if aug_params is not None:
            if self.sparse:
                if self.load_pseudo_gt:
                # self.augmentor = SparseFlowAugmentor(**aug_params)
                    self.augmentor = Pseudo_SparseFlowDispAugmentor(**aug_params)
                else:
                    self.augmentor = SparseFlowDispAugmentor(**aug_params)
                    


        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.disp_list = []
        self.pseudo_disp_list = []

    def __getitem__(self, index):
        sample = {}
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True


        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        disp = frame_utils.read_disp(self.disp_list[index], subset=False)

        if self.load_pseudo_gt:
            pseudo_disp = frame_utils.read_disp(self.pseudo_disp_list[index], subset=False)

        img1_left = frame_utils.read_gen(self.image_list[index][0])
        img2_left = frame_utils.read_gen(self.image_list[index][1])
        img1_right = frame_utils.read_gen(self.image_list[index][2])
        img2_right = frame_utils.read_gen(self.image_list[index][3])

        flow = np.array(flow).astype(np.float32)
        img1_left = np.array(img1_left).astype(np.uint8)
        img2_left = np.array(img2_left).astype(np.uint8)
        img1_right = np.array(img1_right).astype(np.uint8)
        img2_right = np.array(img2_right).astype(np.uint8)
        
        # grayscale images
        if len(img1_left.shape) == 2:
            img1_left = np.tile(img1_left[...,None], (1, 1, 3))
            img2_left = np.tile(img2_left[...,None], (1, 1, 3))
            img1_right = np.tile(img1_right[...,None], (1, 1, 3))
            img2_right = np.tile(img2_right[...,None], (1, 1, 3))
        else:
            img1_left = img1_left[..., :3]
            img2_left = img2_left[..., :3]
            img1_right = img1_right[..., :3]
            img2_right = img2_right[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                if self.load_pseudo_gt:
                    img1_left, img2_left, img1_right, img2_right, flow, valid, disp, pseudo_disp = self.augmentor(img1_left, img2_left, img1_right, img2_right, flow, valid, disp, pseudo_disp)
                else:
                    img1_left, img2_left, img1_right, img2_right, flow, valid, disp = self.augmentor(img1_left, img2_left, img1_right, img2_right, flow, valid, disp)

        
        img1_left = torch.from_numpy(img1_left).permute(2, 0, 1).float()
        img2_left = torch.from_numpy(img2_left).permute(2, 0, 1).float()
        img1_right = torch.from_numpy(img1_right).permute(2, 0, 1).float()
        img2_right = torch.from_numpy(img2_right).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        disp = torch.from_numpy(disp).permute(0, 1).float()
        if self.load_pseudo_gt:
            pseudo_disp = torch.from_numpy(pseudo_disp).permute(0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        sample['left_1'] = img1_left
        sample['left_2'] = img2_left
        sample['right_1'] = img1_right
        sample['right_2'] = img2_right
        sample['flow'] = flow
        sample['valid'] = valid.float()
        sample['disp'] = disp
        sample['name'] = self.image_list[index]
        if self.load_pseudo_gt:
            sample['pseudo_disp'] = pseudo_disp

        return sample


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)




# 写这部分的数据读取
class CrossDomainFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=True):
        self.augmentor = None
        self.sparse = sparse
        
        if aug_params is not None:
            if self.sparse:
                # self.augmentor = SparseFlowAugmentor(**aug_params)
                    self.augmentor = CrossDomainSparseFlowAugmentor(**aug_params)

        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        sample = {}
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True


        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])


        img1_clean = frame_utils.read_gen(self.image_list[index][0])
        img2_clean = frame_utils.read_gen(self.image_list[index][1])
        img1_foggy = frame_utils.read_gen(self.image_list[index][2])
        img2_foggy = frame_utils.read_gen(self.image_list[index][3])

        flow = np.array(flow).astype(np.float32)
        img1_clean = np.array(img1_clean).astype(np.uint8)
        img2_clean = np.array(img2_clean).astype(np.uint8)
        img1_foggy = np.array(img1_foggy).astype(np.uint8)
        img2_foggy = np.array(img2_foggy).astype(np.uint8)
        
        # grayscale images
        if len(img1_clean.shape) == 2:
            img1_clean = np.tile(img1_clean[...,None], (1, 1, 3))
            img2_clean = np.tile(img2_clean[...,None], (1, 1, 3))
            img1_foggy = np.tile(img1_foggy[...,None], (1, 1, 3))
            img2_foggy = np.tile(img2_foggy[...,None], (1, 1, 3))
        else:
            img1_clean = img1_clean[..., :3]
            img2_clean = img2_clean[..., :3]
            img1_foggy = img1_foggy[..., :3]
            img2_foggy = img2_foggy[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1_clean, img2_clean, img1_foggy, img2_foggy, flow, valid = self.augmentor(img1_clean, img2_clean, img1_foggy, img2_foggy, flow, valid)
                
        img1_clean = torch.from_numpy(img1_clean).permute(2, 0, 1).float()
        img2_clean = torch.from_numpy(img2_clean).permute(2, 0, 1).float()
        img1_foggy = torch.from_numpy(img1_foggy).permute(2, 0, 1).float()
        img2_foggy = torch.from_numpy(img2_foggy).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
   
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        sample['clean_1'] = img1_clean
        sample['clean_2'] = img2_clean
        sample['foggy_1'] = img1_foggy
        sample['foggy_2'] = img2_foggy
        sample['flow'] = flow
        sample['valid'] = valid.float()
  
        return sample


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)







# 仿真-实测数据读取
class CrossSceneFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=True):
        self.augmentor = None
        self.sparse = sparse
        
        if aug_params is not None:
            if self.sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)

        self.init_seed = False
        self.flow_list = []
        self.syn_image_list = []
        self.real_image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        sample = {}
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True


        index = index % len(self.syn_image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])


        img1_syn = frame_utils.read_gen(self.syn_image_list[index][0])
        img2_syn = frame_utils.read_gen(self.syn_image_list[index][1])
        img1_real = frame_utils.read_gen(self.real_image_list[index][0])
        img2_real = frame_utils.read_gen(self.real_image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1_syn = np.array(img1_syn).astype(np.uint8)
        img2_syn = np.array(img2_syn).astype(np.uint8)
        img1_real = np.array(img1_real).astype(np.uint8)
        img2_real = np.array(img2_real).astype(np.uint8)
        
        # grayscale images
        if len(img1_syn.shape) == 2:
            img1_syn = np.tile(img1_syn[...,None], (1, 1, 3))
            img2_syn = np.tile(img2_syn[...,None], (1, 1, 3))
            img1_real = np.tile(img1_real[...,None], (1, 1, 3))
            img2_real = np.tile(img2_real[...,None], (1, 1, 3))
        else:
            img1_syn = img1_syn[..., :3]
            img2_syn = img2_syn[..., :3]
            img1_real = img1_real[..., :3]
            img2_real = img2_real[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                temp_flow = flow.copy()
                temp_valid = valid.copy()
                img1_syn, img2_syn, flow, valid = self.augmentor(img1_syn, img2_syn, flow, valid)
                img1_real, img2_real, _, __ = self.augmentor(img1_real, img2_real, temp_flow, temp_valid)
                
        img1_syn = torch.from_numpy(img1_syn).permute(2, 0, 1).float()
        img2_syn = torch.from_numpy(img2_syn).permute(2, 0, 1).float()
        img1_real = torch.from_numpy(img1_real).permute(2, 0, 1).float()
        img2_real = torch.from_numpy(img2_real).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
   
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        sample['syn_1'] = img1_syn
        sample['syn_2'] = img2_syn
        sample['real_1'] = img1_real
        sample['real_2'] = img2_real
        sample['flow'] = flow
        sample['valid'] = valid.float()
  
        return sample


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.syn_image_list = v * self.syn_image_list
        return self
        
    def __len__(self):
        return len(self.syn_image_list)




class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


# ours dataset
class KITTI_Clean(StereoFlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI', load_pseudo_gt=True):
        super(KITTI_Clean, self).__init__(aug_params, sparse=True, load_pseudo_gt=load_pseudo_gt)
        root = osp.join(root, split)
        images1_left = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2_left = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        images1_right = sorted(glob(osp.join(root, 'image_3/*_10.png')))
        images2_right = sorted(glob(osp.join(root, 'image_3/*_11.png')))

        for img1_l, img2_l, img1_r, img2_r in zip(images1_left, images2_left, images1_right, images2_right):
            frame_id = img1_l.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1_l, img2_l, img1_r, img2_r] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            self.disp_list = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
        
            if load_pseudo_gt:
                self.pseudo_disp_list = sorted(glob(osp.join(root, 'disp_occ_0_pseudo_gt/*_10.png')))


class KITTI_Foggy(CrossDomainFlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI_Foggy, self).__init__(aug_params, sparse=True)
        root = osp.join(root, split)

        images1_clean = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2_clean = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        images1_foggy = sorted(glob(osp.join(root, 'image_foggy/*_10.png')))
        images2_foggy = sorted(glob(osp.join(root, 'image_foggy/*_11.png')))

        for img1_clean, img2_clean, img1_foggy, img2_foggy in zip(images1_clean, images2_clean, images1_foggy, images2_foggy):
            frame_id = img1_clean.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1_clean, img2_clean, img1_foggy, img2_foggy] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        



class Real_Foggy(CrossSceneFlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(Real_Foggy, self).__init__(aug_params, sparse=True)
        root = osp.join(root, split)

        images1_synfoggy = sorted(glob(osp.join(root, 'image_foggy/*_10.png')))
        images2_synfoggy = sorted(glob(osp.join(root, 'image_foggy/*_11.png')))

        images1_realfoggy = sorted(glob(osp.join(root, 'real_foggy/*_10.png')))
        images2_realfoggy = sorted(glob(osp.join(root, 'real_foggy/*_11.png')))

        for img1_syn, img2_syn  in zip(images1_synfoggy, images2_synfoggy):
            frame_id = img1_syn.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.syn_image_list += [ [img1_syn, img2_syn] ]
        
        for img1_real, img2_real in zip(images1_realfoggy, images2_realfoggy):
            self.real_image_list += [ [img1_real, img2_real] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))



# def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
#     """ Create the data loader for the corresponding trainign set """

#     if args.stage == 'chairs':
#         aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
#         train_dataset = FlyingChairs(aug_params, split='training')
    
#     elif args.stage == 'things':
#         aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
#         clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
#         final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
#         train_dataset = clean_dataset + final_dataset

#     elif args.stage == 'sintel':
#         aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
#         things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
#         sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
#         sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

#         if TRAIN_DS == 'C+T+K+S+H':
#             kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
#             hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
#             train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

#         elif TRAIN_DS == 'C+T+K/S':
#             train_dataset = 100*sintel_clean + 100*sintel_final + things

#     elif args.stage == 'kitti':
#         aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
#         train_dataset = KITTI(aug_params, split='training')

#     train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
#         pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

#     print('Training with %d image pairs' % len(train_dataset))
#     return train_loader



def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI_Clean(aug_params, split='training', root=args.data_dir, load_pseudo_gt=args.load_pseudo_gt)
        # train_dataset = KITTI_Clean(split='training', root=args.data_dir, load_pseudo_gt=args.load_pseudo_gt)

    elif args.stage == 'kitti_foggy':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI_Foggy(aug_params, split='training', root=args.data_dir)
    elif args.stage == 'real_foggy':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Real_Foggy(aug_params, split='training', root=args.data_dir)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=False, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
