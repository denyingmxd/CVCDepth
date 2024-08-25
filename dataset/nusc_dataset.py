# Copyright (c) 2023 42dot. All rights reserved.
import os

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image
from pyquaternion import Quaternion

import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
# from external.utils import Camera, generate_depth_map, make_list
# from external.dataset import DGPDataset, SynchronizedSceneDataset, stack_sample
from external.dataset import stack_sample
import kornia
import random
_DEL_KEYS = ['rgb', 'rgb_context', 'rgb_original', 'rgb_context_original', 'intrinsics', 'contexts', 'splitname']
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def transform_mask_sample(sample, data_transform):
    """
    This function transforms masks to match input rgb images.
    """
    image_shape = data_transform.keywords['image_shape']
    sample['mask'] = torch.ones(image_shape).unsqueeze(0)
    return sample




def align_dataset(sample, scales, contexts,has_context):
    """
    This function reorganize samples to match our trainer configuration.
    """
    K = sample['intrinsics']
    aug_images = sample['rgb']
    org_images = sample['rgb_original']

    if has_context:
        aug_contexts = sample['rgb_context']
        org_contexts = sample['rgb_context_original']

    n_cam, _, w, h = aug_images.shape

    # initialize intrinsics
    resized_K = np.expand_dims(np.eye(4), 0).repeat(n_cam, axis=0)
    resized_K[:, :3, :3] = K

    # augment images and intrinsics in accordance with scales
    for scale in scales:
        scaled_K = resized_K.copy()
        scaled_K[:, :2, :] /= (2 ** scale)

        sample[('K', scale)] = scaled_K.copy()
        sample[('inv_K', scale)] = np.linalg.pinv(scaled_K).copy()

        resized_org = F.interpolate(org_images,
                                    size=(w // (2 ** scale), h // (2 ** scale)),
                                    mode='bilinear',
                                    align_corners=False)
        resized_aug = F.interpolate(aug_images,
                                    size=(w // (2 ** scale), h // (2 ** scale)),
                                    mode='bilinear',
                                    align_corners=False)

        sample[('color', 0, scale)] = resized_org
        sample[('color_aug', 0, scale)] = resized_aug

    # for context data
    if has_context:
        for idx, frame in enumerate(contexts):
            sample[('color', frame, 0)] = org_contexts[idx]
            sample[('color_aug', frame, 0)] = aug_contexts[idx]

    # delete unused arrays
    for key in list(sample.keys()):
        if key in _DEL_KEYS:
            try:
                del sample[key]
            except:
                pass
    return sample


class NUSCdataset(torch.utils.data.Dataset):
    """
    Superclass for DGP dataset loaders of the packnet_sfm repository.
    """

    def __init__(self, cfg,mode,**kwargs):
        super(NUSCdataset).__init__()
        self.cameras = kwargs['cameras']
        scale_range = kwargs['scale_range']
        self.scales = np.arange(scale_range + 2)
        ## self-occ masks
        self.with_mask = kwargs['with_mask']
        self.with_pose = kwargs['with_pose']
        self.num_cams = len(self.cameras)
        self.mode=mode
        self.with_depth = self.mode=='val'
        self.with_input_depth = False
        with open('/data/laiyan/airs/VFDepth/dataset/nusc/info_{}.pkl'.format(mode), 'rb') as f:
            self.info = pickle.load(f)

        if cfg['eval']['type']=='surrounddepth':
            with open("/data/laiyan/airs/SurroundDepth/datasets/nusc/{}.txt".format(mode), 'r') as f:
                self.filenames = f.readlines()
        else:
            with open('/data/laiyan/airs/VFDepth/dataset/nusc/{}_vf.txt'.format(mode), 'r') as f:
                self.filenames = f.readlines()

        if cfg['eval'].get('overlap') is True:
            with open("/data/laiyan/airs/SurroundDepth/datasets/nusc/{}_overlap.txt".format(mode), 'r') as f:
                self.filenames = f.readlines()

        self.data_root = '/data/laiyan/datasets/nuscenes/v1.0/'
        self.data_transform = kwargs['data_transform']
        self.cameras = [i.upper() for i in self.cameras]
        self.cfg=cfg
        self.alias_cams=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

    def __len__(self):
        return len(self.filenames)


    def get_K(self,index_spatial,infos):

        return np.array(infos[self.alias_cams[index_spatial]]['intrinsics'], dtype=np.float32)


    def __getitem__(self, idx):
        import time
        # get DGP sample (if single sensor, make it a list)
        index_temporal = self.filenames[idx].strip()
        # index_temporal ='15616458266936490'
        # loop over all cameras
        sample = []
        contexts = [-1,1]
        self.has_context = self.mode=='train'

        infos = self.info[index_temporal]

        for index_spatial,cam in enumerate(self.alias_cams):

            rgb_filename = os.path.join(self.data_root,infos[self.alias_cams[index_spatial]]['rgb_filenames'][0])
            rgb_cur = pil_loader(rgb_filename)
            data = {
                'idx': idx,
                'index_temporal':index_temporal,
                'sensor_name':cam,
                'contexts': contexts,
                'rgb':rgb_cur,
                'intrinsics': self.get_K(index_spatial,infos),
            }

            # if depth is returned
            if self.with_depth:

                if self.cfg['eval'].get('overlap') is True:
                    data.update({
                        'my_depth': np.load(rgb_filename.replace('jpg', 'npz').replace('samples', 'overlap_depth/samples'))['arr_0'][None,:]
                    })
                else:
                    data.update({
                        'my_depth':np.load(rgb_filename.replace('jpg','npy').replace('samples','depth/samples'))[None,:]
                    })
            # if depth is returned


            # if pose is returned
            if self.with_pose:
                xxx = Quaternion(infos[self.alias_cams[index_spatial]]['extrinsics']['quat']).transformation_matrix
                xxx[:3, 3] = np.array(infos[self.alias_cams[index_spatial]]['extrinsics']['tvec'])
                data.update({
                    'extrinsics':xxx
                })

            # if context is returned
            if self.has_context:
                rgb_contexts = []
                for iddddx, i in enumerate(contexts):

                    rgb_context_filename = os.path.join(self.data_root,infos[self.alias_cams[index_spatial]]['rgb_filenames'][iddddx+1])
                    rgb_context = pil_loader(rgb_context_filename)
                    rgb_contexts.append(rgb_context)
                data.update({
                    'rgb_context':rgb_contexts
                })

            sample.append(data)


        # apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]

            sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]


        # stack and align dataset for our trainer
        sample = stack_sample(sample)
        sample = align_dataset(sample, self.scales, contexts,self.has_context)
        # import pickle
        # with open('vf_my.pickle', 'wb') as handle:
        #     pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # exit()

        assert (self.cfg.get('training').get('flip_version') is not None) + (self.cfg.get('training').get('random_aug_intrinsics') is not None) <= 1
        if self.cfg.get('training').get('flip_version') is not None and self.mode=='train':
            flip_version = self.cfg.get('training').get('flip_version')
            here_contexts=contexts.copy()
            here_contexts.append(0)
            do_flip =  self.mode=='train' and random.random() > 0.5
            sample['flip_version']=flip_version
            if do_flip:
                hflip = kornia.geometry.transform.Hflip()
                for scale in self.scales:
                    if scale>0:
                        continue
                    for context in here_contexts:
                        color_aug = sample[('color_aug',context,scale)].clone()

                        if flip_version==1 or flip_version==3 or flip_version>=5:
                            sample[('color_aug_flip',context,scale)] = hflip(color_aug)
                            sample['flips'] = torch.ones(6,1,1)
                        elif flip_version==2 or flip_version==4:
                            sample[('color_aug_flip', context, scale)] = []
                            sample['flips'] =  torch.bernoulli(torch.empty((6, 1, 1)).uniform_(0, 1))
                            for i in range(6):
                                color_aug_i =color_aug[i]
                                if sample['flips'][i].sum().item()>0:
                                    sample[('color_aug_flip', context, scale)].append(hflip(color_aug_i))
                                else:
                                    sample[('color_aug_flip', context, scale)].append(color_aug_i)
                            sample[('color_aug_flip', context, scale)] = torch.stack(sample[('color_aug_flip', context, scale)])

            else:
                sample['flips'] = torch.zeros(6, 1, 1)
                for scale in self.scales:
                    if scale>0:
                        continue
                    for context in here_contexts:
                        color_aug = sample[('color_aug',context,scale)].clone()
                        sample[('color_aug_flip', context, scale)] = color_aug

        if self.cfg.get('training').get('random_aug_intrinsics') is not None and self.mode=='train':
            do_flip = self.mode == 'train' and random.random() > 0.5
            # do_flip = True
            if do_flip:
                here_contexts = contexts.copy()
                here_contexts.append(0)
                hflip = kornia.geometry.transform.Hflip()
                for scale in self.scales:

                    for context in here_contexts:
                        color_aug = sample[('color_aug', context, scale)].clone()
                        sample[('color_aug', context, scale)] = hflip(color_aug)

                        color = sample[('color', context, scale)].clone()
                        sample[('color', context, scale)] = hflip(color)

                    bb,cc,hh,ww = sample[('color_aug', context, scale)].shape
                    KK = sample[('K',scale)].copy()
                    KK[:,0,0] = -KK[:,0,0]
                    KK[:, 0, 2] = ww-KK[:, 0, 2]
                    sample[('K', scale)] = KK
                    sample[('inv_K', scale)] = np.linalg.pinv(KK.copy())

        return sample