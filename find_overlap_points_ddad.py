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
from vis_spatial_projection_ddad import proj_two_cams,scatter_depth_on_rgb
from utils.misc import _REL_CAM_DICT
import multiprocessing as mp
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
fpath = "/data/laiyan/airs/SurroundDepth/datasets/ddad/{}.txt"
train_filenames = readlines(fpath.format("val"))
nusc = None
with open('/data/laiyan/ssd/ddad/meta_data/info_{}.pkl'.format('val'), 'rb') as f:
    train_pkl = pickle.load(f)
cams = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
data_root = '/data/laiyan/ssd/ddad/raw_data/'
rgb_path = '/data/laiyan/ssd/ddad/raw_data/'
mask_path = '/data/laiyan/ssd/ddad/mask/'
depth_path = '/data/laiyan/ssd/ddad/depth/'
with open("/data/laiyan/airs/VFDepth/dataset/ddad_mask/mask_idx_dict.pkl", 'rb') as f:
    mask_pkl = pickle.load(f)


full_res_shape = (1216,1936)
flip=False


# for name in tqdm.tqdm(train_filenames):
#     sample = name.strip()
#     # sample = '15621787638931470'
#     scene_id = train_pkl[sample]['scene_name']
#     for cam_index in range(len(cams)):
#         overlap_depth =np.zeros(full_res_shape,dtype=np.float64)
#         for rel_cam in _REL_CAM_DICT[cam_index]:
#             aaa,ref_color=proj_two_cams(sample, cams[rel_cam], cams[cam_index], scene_id,train_pkl)
#             overlap_depth = overlap_depth+aaa
#         # scatter_depth_on_rgb(ref_color,overlap_depth)
#         print(os.path.join(depth_path, scene_id, 'depth_overlap', cams[cam_index], sample + '.npz'))
#         os.makedirs(os.path.join(depth_path, scene_id, 'depth_overlap', cams[cam_index]),exist_ok=True)
#         np.savez_compressed(os.path.join(depth_path, scene_id, 'depth_overlap', cams[cam_index], sample + '.npz'),overlap_depth)



def generate(index):
    name = train_filenames[index]
    sample = name.strip()
    # sample = '15621787638931470'
    scene_id = train_pkl[sample]['scene_name']
    for cam_index in range(1):
        overlap_depth =np.zeros(full_res_shape,dtype=np.float64)
        for rel_cam in _REL_CAM_DICT[cam_index]:
            aaa,ref_color=proj_two_cams(sample, cams[rel_cam], cams[cam_index], scene_id,train_pkl)
            overlap_depth = overlap_depth+aaa
        scatter_depth_on_rgb(ref_color,overlap_depth)
        # print(os.path.join(depth_path, scene_id, 'depth_overlap', cams[cam_index], sample + '.npz'))
        # os.makedirs(os.path.join(depth_path, scene_id, 'depth_overlap', cams[cam_index]),exist_ok=True)
        # np.savez_compressed(os.path.join(depth_path, scene_id, 'depth_overlap', cams[cam_index], sample + '.npz'),overlap_depth)

p = mp.Pool(2)
info_list = [i for i in range(len(train_filenames))]
p.map_async(generate, info_list)
p.close()
p.join()
