import os
from nuscenes.nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from layers import *
import torch.nn.functional as F
import pickle
import tqdm
from vis_spatial_projection_nusc import proj_two_cams,scatter_depth_on_rgb
from utils.misc import _REL_CAM_DICT
import multiprocessing as mp
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
fpath = "/data/laiyan/airs/VFDepth/dataset/nusc/val_vf.txt"
train_filenames = readlines(fpath)
full_path = "/data/laiyan/airs/SurroundDepth/datasets/nusc/val.txt"
train_full_filenames = readlines(full_path)
nusc = None
with open('/data/laiyan/airs/VFDepth/dataset/nusc/info_{}.pkl'.format('val'), 'rb') as f:
    train_pkl = pickle.load(f)
cams = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
alias_cams=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
data_root = '/data/laiyan/datasets/nuscenes/v1.0/'



full_res_shape = (900,1600)
flip=False


for name in tqdm.tqdm(train_full_filenames):
    if name in train_filenames:
        continue
    sample = name.strip()
    # sample = '15621787638931470'
    for cam_index in range(len(cams)):
        overlap_depth =np.zeros(full_res_shape,dtype=np.float64)
        for rel_cam in _REL_CAM_DICT[cam_index]:
            aaa,ref_color,ref_dpeth_path=proj_two_cams(sample, rel_cam, cam_index,train_pkl)
            overlap_depth = overlap_depth+aaa
        print(ref_dpeth_path)
        # scatter_depth_on_rgb(ref_color,overlap_depth)
        target_depth_path = ref_dpeth_path.replace('depth','overlap_depth').replace('npy','npz')
        os.makedirs('/'.join(target_depth_path.split('/')[:-1]),exist_ok=True)
        np.savez_compressed(target_depth_path,overlap_depth)
    # exit()




# def generate(index):
#     name = train_filenames[index]
#     sample = name.strip()
#     print(index)
#     for cam_index in range(len(cams)):
#         overlap_depth =np.zeros(full_res_shape,dtype=np.float64)
#         for rel_cam in _REL_CAM_DICT[cam_index]:
#             aaa,ref_color,ref_dpeth_path=proj_two_cams(sample, rel_cam, cam_index,train_pkl)
#             overlap_depth = overlap_depth+aaa
#             # print(ref_dpeth_path)
#         # scatter_depth_on_rgb(ref_color,overlap_depth)
#         target_depth_path = ref_dpeth_path.replace('depth','overlap_depth').replace('npy','npz')
#         print(target_depth_path)
#         os.makedirs('/'.join(target_depth_path.split('/')[:-1]),exist_ok=True)
#         np.savez_compressed(target_depth_path,overlap_depth)
#
# p = mp.Pool(12)
# info_list = [i for i in range(len(train_filenames))]
# p.map_async(generate, info_list)
# p.close()
# p.join()
