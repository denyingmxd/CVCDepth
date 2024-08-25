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



def get_org_K(line,cam,train_pkl):
    K = np.eye(4).astype(np.float32)
    K[:3, :3] = train_pkl[line][cam]['intrinsics']
    return K


def get_depth_from_npz(line, cam,scene_id):

    # depth_path=os.path.join(self.depth_path, scene_id, 'depth', cam, line + '.npy')
    # depth_gt = np.load(depth_path)

    ppp = os.path.join(depth_path, scene_id, 'depth', cam, line + '.npz')
    depth_gt = np.load(ppp)['arr_0']

    return depth_gt




def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_rgb_pil(line,scene_id,cam):
    full_color_path = os.path.join(rgb_path, scene_id, 'rgb',
                                   cam, line + '.jpg')
    color = pil_loader(full_color_path)
    return color

def get_CS(line,cam,train_pkl):
    pose_0_spatial = train_pkl[line][cam]['extrinsics'][
        'quat'].transformation_matrix
    pose_0_spatial[:3, 3] = train_pkl[line][cam]['extrinsics']['tvec']
    return pose_0_spatial.astype(np.float32)

def scatter_depth_on_rgb(rgb, depth):
    import cv2
    fig, axs = plt.subplots(1, 1)
    # rgb = torch.from_numpy(rgb)
    # axs.imshow(rgb)
    # from skimage.transform import resize
    # depth = cv2.resize(depth, dsize=(640, 352), interpolation=cv2.INTER_NEAREST)
    h, w = depth.shape
    y, x = np.nonzero(depth)
    axs.scatter(x, h-y, c=depth[np.nonzero(depth)], s=0.1,vmin=0)
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    plt.axis('off')

    plt.xlim(0,w)
    plt.ylim(0,h)
    plt.show()
    return fig

def proj_two_cams(sample,src_cam, ref_cam,scene_id,train_pkl,vis=False):

    src_color = get_rgb_pil(sample,scene_id,src_cam)
    ref_color = get_rgb_pil(sample,scene_id,ref_cam)
    if vis:
        plt.imshow(src_color);plt.show()
        plt.imshow(ref_color);plt.show()
    # exit()
    src_K = torch.from_numpy(get_org_K(sample, src_cam,train_pkl))
    ref_K = torch.from_numpy(get_org_K(sample, ref_cam,train_pkl))

    src_cs = get_CS(sample,src_cam,train_pkl).astype(np.float32)
    ref_cs = get_CS(sample,ref_cam,train_pkl).astype(np.float32)
    ref_to_src_pose_tensor = torch.from_numpy(np.linalg.inv(src_cs) @ ref_cs)
    # src_to_ref_pose_tensor = torch.from_numpy(np.linalg.inv(ref_cs) @ src_cs)
    ref_depth = get_depth_from_npz(sample, ref_cam,scene_id)
    src_depth = get_depth_from_npz(sample, src_cam,scene_id)
    if vis:
        scatter_depth_on_rgb(src_color,src_depth)
        scatter_depth_on_rgb(ref_color,ref_depth)
    exit()
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
        return aaa,ref_color
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
    return aaa,ref_color

def show_color_depth(color,depth,s):
    plt.imshow(color)
    plt.scatter(depth[:, 0], depth[:, 1], c=depth[:, 2], s=s,vmin=0)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    sample = "1569370961886777"
    scene_id = train_pkl[sample]['scene_name']
    proj_two_cams(sample,cams[1],cams[0],scene_id,train_pkl,vis=True)
