# Copyright (c) 2023 42dot. All rights reserved.
import os

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
# from external.utils import Camera, generate_depth_map, make_list
# from external.dataset import DGPDataset, SynchronizedSceneDataset, stack_sample
from external.dataset import stack_sample
import kornia
import random
_DEL_KEYS = ['rgb', 'rgb_context', 'rgb_original', 'rgb_context_original', 'contexts', 'splitname','intrinsics']
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
    # resize transform
    resize_transform = transforms.Resize(image_shape, interpolation=transforms.InterpolationMode.LANCZOS)
    sample['mask'] = resize_transform(sample['mask'])
    # totensor transform
    tensor_transform = transforms.ToTensor()
    sample['mask'] = tensor_transform(sample['mask'])
    return sample


def mask_loader_scene(path, mask_idx, cam):
    """
    This function loads mask that correspondes to the scene and camera.
    """
    fname = os.path.join(path, str(mask_idx), '{}_mask.png'.format(cam.upper()))
    with open(fname, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


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


class DDADdataset(torch.utils.data.Dataset):
    """
    Superclass for DGP dataset loaders of the packnet_sfm repository.
    """

    def __init__(self, cfg,mode,**kwargs):
        super(DDADdataset).__init__()
        self.cameras = kwargs['cameras']
        scale_range = kwargs['scale_range']
        self.scales = np.arange(scale_range + 2)
        ## self-occ masks
        self.with_mask = kwargs['with_mask']
        self.with_pose = kwargs['with_pose']
        self.num_cams = len(self.cameras)
        self.mask_loader = mask_loader_scene
        self.mode=mode
        self.with_depth = self.mode=='val'
        self.with_input_depth = False
        with open('./dataset/ddad/info_{}.pkl'.format(mode), 'rb') as f:
            self.info = pickle.load(f)
        if cfg['eval'].get('vis_only') and cfg['eval'].get('vis_only') == True:
            with open('./{}.txt'.format('vis'), 'r') as f:
                self.filenames = f.readlines()
        else:
            with open('./dataset/ddad/{}.txt'.format(mode), 'r') as f:
                self.filenames = f.readlines()

        self.rgb_path = '/data/laiyan/ssd/ddad/raw_data/'
        self.depth_path = '/data/laiyan/ssd/ddad/depth'
        self.match_path = '/data/laiyan/ssd/ddad/match'
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.mask_path = os.path.join(cur_path, 'ddad_mask')
        file_name = os.path.join(self.mask_path, 'mask_idx_dict.pkl')
        self.mask_idx_dict = pd.read_pickle(file_name)
        self.data_transform = kwargs['data_transform']
        self.cameras = [i.upper() for i in self.cameras]
        self.cfg=cfg

    def __len__(self):
        return len(self.filenames)


    def get_K(self,index_temporal,index_spatial):
        K = np.eye(3).astype(np.float32)
        K[:3, :3] = self.info[index_temporal][self.cameras[index_spatial]]['intrinsics']
        return K


    def __getitem__(self, idx):
        import time
        # get DGP sample (if single sensor, make it a list)
        index_temporal = self.filenames[idx].strip()
        # index_temporal ='15616458266936490'
        # loop over all cameras
        sample = []
        contexts = [-1,1]
        self.has_context = self.mode=='train'

        # for self-occ mask
        scene_name = self.info[index_temporal]['scene_name']
        mask_idx = self.mask_idx_dict[int(scene_name)]

        for index_spatial,cam in enumerate(self.cameras):

            rgb_filename = os.path.join(self.rgb_path, scene_name, 'rgb',
                                self.cameras[index_spatial], index_temporal + '.jpg')
            filename = scene_name+'/'+'{}'+'/'+cam+'/'+index_temporal
            data = {
                'idx': idx,
                'index_temporal':int(index_temporal),
                'sensor_name':cam,
                'contexts': contexts,
                'splitname': '%s_%010d' % (self.mode, idx),
                'rgb': pil_loader(rgb_filename),
                'intrinsics': self.get_K(index_temporal, index_spatial),
                'intrinsics_org': self.get_K(index_temporal, index_spatial),
            }

            # if depth is returned
            if self.with_depth:
                if self.cfg['eval'].get('overlap') is True:
                    data.update({
                        'depth':np.load(os.path.join(self.depth_path, scene_name, 'depth_overlap',
                                self.cameras[index_spatial], index_temporal + '.npz'))['arr_0'][None,:]
                    })
                else:
                    data.update({
                        'depth': np.load(os.path.join(self.depth_path, scene_name, 'depth',
                                                         self.cameras[index_spatial], index_temporal + '.npz'))[
                                        'arr_0'][None, :]
                    })
            # if depth is returned
            if self.with_input_depth:
                data.update({
                    'input_depth': np.load(os.path.join(self.depth_path, scene_name, 'depth',
                            self.cameras[index_spatial], index_temporal + '.npz'))['arr_0']
                })

            # if pose is returned
            if self.with_pose:
                xxx = self.info[index_temporal][self.cameras[index_spatial]]['extrinsics']['quat'].transformation_matrix
                xxx[:3, 3] = self.info[index_temporal][self.cameras[index_spatial]]['extrinsics']['tvec']
                data.update({
                    'extrinsics':xxx

                })
            # with mask
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, mask_idx, self.cameras[index_spatial].lower())
                })

            # if context is returned
            if self.has_context:
                rgb_contexts = []
                for iddddx, i in enumerate(contexts):
                    index_temporal_i = self.info[index_temporal]['context'][iddddx]

                    rgb_context_filename = os.path.join(self.rgb_path, scene_name, 'rgb',
                                                self.cameras[index_spatial], index_temporal_i + '.jpg')
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