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
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
fpath = "/data/laiyan/airs/VFDepth/dataset/nusc/val_vf.txt"
train_filenames = readlines(fpath)
nusc = None
with open('/data/laiyan/airs/VFDepth/dataset/nusc/info_{}.pkl'.format('val'), 'rb') as f:
    train_pkl = pickle.load(f)
cams = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
alias_cams=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
data_root = '/data/laiyan/datasets/nuscenes/v1.0/'


# index_temporal = self.filenames[idx].strip()
# index_temporal ='fd8420396768425eabec9bdddf7e64b6'
# loop over all cameras



class Project3D_vis(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=0):
        super(Project3D_vis, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = torch.cat((pix_coords, cam_points[:, 2:3, :]),dim=1)
        pix_coords = pix_coords.view(self.batch_size, 3, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)

        return pix_coords



def get_org_K(cam,infos):
    resized_K =np.eye(4,dtype=np.float32)
    resized_K[ :3, :3] = np.array(infos[alias_cams[cam]]['intrinsics'], dtype=np.float32)
    return resized_K


def get_depth_from_npz( cam,infos):
    rgb_filename = os.path.join(data_root, infos[alias_cams[cam]]['rgb_filenames'][0])
    depth_gt = np.load(rgb_filename.replace('jpg','npy').replace('samples','depth/samples'))
    return depth_gt,rgb_filename.replace('jpg','npy').replace('samples','depth/samples')




def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_rgb_pil(cam,infos):

    full_color_path = os.path.join(data_root, infos[alias_cams[cam]]['rgb_filenames'][0])
    color = pil_loader(full_color_path)
    return color

def get_CS(cam,infos):
    xxx = Quaternion(infos[alias_cams[cam]]['extrinsics']['quat']).transformation_matrix
    xxx[:3, 3] = np.array(infos[alias_cams[cam]]['extrinsics']['tvec'])
    return xxx.astype(np.float32)

def scatter_depth_on_rgb(rgb, depth):
    fig, axs = plt.subplots(1, 1)
    # rgb = torch.from_numpy(rgb)
    axs.imshow(rgb)
    y, x = np.nonzero(depth)
    axs.scatter(x, y, c=depth[np.nonzero(depth)], s=0.1,vmin=0,vmax=25)
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    plt.axis('off')
    plt.show()
    return fig

def proj_two_cams(index_temporal,src_cam, ref_cam,train_pkl,vis=False):
    infos = train_pkl[index_temporal]
    src_color = get_rgb_pil(src_cam,infos)
    ref_color = get_rgb_pil(ref_cam,infos)
    if vis:
        plt.imshow(src_color);plt.show()
        plt.imshow(ref_color);plt.show()

    src_K = torch.from_numpy(get_org_K(src_cam,infos))
    ref_K = torch.from_numpy(get_org_K(ref_cam,infos))

    src_cs = get_CS(src_cam,infos).astype(np.float32)
    ref_cs = get_CS(ref_cam,infos).astype(np.float32)
    ref_to_src_pose_tensor = torch.from_numpy(np.linalg.inv(src_cs) @ ref_cs)
    # src_to_ref_pose_tensor = torch.from_numpy(np.linalg.inv(ref_cs) @ src_cs)
    ref_depth,ref_depth_path = get_depth_from_npz(ref_cam,infos)
    src_depth,src_depth_path = get_depth_from_npz(src_cam,infos)
    if vis:
        scatter_depth_on_rgb(src_color,src_depth)
        scatter_depth_on_rgb(ref_color,ref_depth)
    # exit()
    bs=1
    w,h = src_color.size
    backproj = BackprojectDepth(bs,h,w)
    project_3d = Project3D_vis(bs,h,w)
    depth_tensor = torch.from_numpy(ref_depth).unsqueeze(0).unsqueeze(0).contiguous().float()
    # depth_tensor.fill_(0)
    cam_points = backproj(depth_tensor, torch.linalg.inv(ref_K.unsqueeze(0)))
    pix_coords = project_3d(cam_points, src_K.unsqueeze(0), ref_to_src_pose_tensor.unsqueeze(0))
    # ref_color_tensor = transforms.ToTensor()(ref_color)[None,:]
    # from_ref_to_src_rgb = F.grid_sample(ref_color_tensor,pix_coords,padding_mode="zeros", align_corners=True)
    # plt.imshow(from_ref_to_src_rgb[0].permute((1,2,0)))
    # plt.show()
    # exit()
    pix_coords = pix_coords.numpy()[0] #(900,1600,2)
    pix_coords_flatten = pix_coords.reshape(-1,3)#???
    # pix_coords_flatten[:,0]=1600- pix_coords_flatten[:,0]
    mask_x = (pix_coords_flatten[:,0]>1) & (pix_coords_flatten[:,0]<w-1)
    mask_y = (pix_coords_flatten[:,1]>1) & (pix_coords_flatten[:,1]<h-1)
    mask_depth = (pix_coords_flatten[:,2]>1) & (pix_coords_flatten[:,2]<250.0001)
    # print(mask_x.sum(),mask_y.sum())

    mask = (mask_x) & (mask_y) & (mask_depth)
    if mask.sum()==0:
        print('no overlap')
        aaa = np.zeros((h,w),dtype=np.float64)
        return aaa,ref_color,ref_depth_path
    # mask = (mask_x) & (mask_y)
    # print(mask_x.sum(), mask_y.sum(),mask_depth.sum(),mask.sum())
    valid = pix_coords_flatten[mask]
    if vis:
        show_color_depth(src_color,valid,0.1)

    aaa = ref_depth.copy().flatten()
    aaa[mask==0]=0
    aaa = aaa.reshape(h,w)
    # print(aaa.sum())
    if vis:
        scatter_depth_on_rgb(ref_color,aaa)
    return aaa,ref_color,ref_depth_path

def show_color_depth(color,depth,s):
    plt.imshow(color)
    plt.scatter(depth[:, 0], depth[:, 1], c=depth[:, 2], s=s,vmin=0)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    index_temporal ='fd8420396768425eabec9bdddf7e64b6'
    proj_two_cams(index_temporal,2,0,train_pkl,vis=True)
