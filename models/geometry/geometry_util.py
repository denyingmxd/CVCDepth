# Copyright (c) 2023 42dot. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix
from collections import Counter
def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1
        
def vec_to_matrix(rot_angle, trans_vec, invert=False):
    """
    This function transforms rotation angle and translation vector into 4x4 matrix.
    """
    # initialize matrices
    b, _, _ = rot_angle.shape
    R_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)
    T_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)

    R_mat[:, :3, :3] = axis_angle_to_matrix(rot_angle).squeeze(1)
    t_vec = trans_vec.clone().contiguous().view(-1, 3, 1)

    if invert == True:
        R_mat = R_mat.transpose(1,2)
        t_vec = -1 * t_vec

    T_mat[:, :3,  3:] = t_vec

    if invert == True:
        P_mat = torch.matmul(R_mat, T_mat)
    else :
        P_mat = torch.matmul(T_mat, R_mat)
    return P_mat


class Projection(nn.Module):
    """
    This class computes projection and reprojection function. 
    """
    def __init__(self, batch_size, height, width, device):
        super().__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        
        # initialize img point grid
        img_points = np.meshgrid(range(width), range(height), indexing='xy')
        img_points = torch.from_numpy(np.stack(img_points, 0)).float()
        img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
        img_points = img_points.to(device)
        
        self.to_homo = torch.ones([batch_size, 1, width*height]).to(device)
        self.homo_points = torch.cat([img_points, self.to_homo], 1)

    def backproject(self, invK, depth):
        """
        This function back-projects 2D image points to 3D.
        """
        depth = depth.view(self.batch_size, 1, -1)

        points3D = torch.matmul(invK[:, :3, :3], self.homo_points)
        points3D = depth*points3D
        return torch.cat([points3D, self.to_homo], 1)
    
    def reproject(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """
        # project points 
        points2D = (K @ T)[:,:3, :] @ points3D
        # normalize projected points for grid sample function
        norm_points2D = points2D[:, :2, :]/(points2D[:, 2:, :] + 1e-7)
        norm_points2D = norm_points2D.view(self.batch_size, 2, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        norm_points2D[..., 0 ] /= self.width - 1
        norm_points2D[..., 1 ] /= self.height - 1
        norm_points2D = (norm_points2D-0.5)*2
        return norm_points2D

    def reproject_unnormed(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """

        # project points
        points2D = (K @ T)[:, :3, :] @ points3D

        # normalize projected points for grid sample function
        points2D[:,:2,:]/=(points2D[:, 2:, :] + 1e-7)
        norm_points2D = points2D
        norm_points2D = norm_points2D.view(self.batch_size, 3, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        bs = points2D.shape[0]
        aaaas = []
        for b in range(bs):
            local_norm_points2D = norm_points2D[b].reshape(-1, 3)
            zz = local_norm_points2D[:, 2:]
            # local_norm_points2D = local_norm_points2D.detach().clone()
            local_norm_points2D[:, 0] = torch.round(local_norm_points2D[:, 0]) - 1
            local_norm_points2D[:, 1] = torch.round(local_norm_points2D[:, 1]) - 1
            val_inds = (local_norm_points2D[:, 0] >= 0) & (local_norm_points2D[:, 1] >= 0)
            val_inds = val_inds & (local_norm_points2D[:, 0] < self.width) & (local_norm_points2D[:, 1] < self.height)
            local_norm_points2D = local_norm_points2D[val_inds, :]
            zz = zz[val_inds, :]
            aaa = torch.zeros((self.height, self.width), device=points3D.device, dtype=points3D.dtype)
            aaa[local_norm_points2D[:, 1].long(), local_norm_points2D[:, 0].long()] = zz[:, 0]

            aaaas.append(aaa.unsqueeze(0))
        aaaas = torch.stack(aaaas)

        return aaaas

    def reproject_transform(self,K, points3D, T):
        points2D = (K @ T)[:, :3, :] @ points3D

        # normalize projected points for grid sample function
        points2D = points2D
        points2D = points2D.view(self.batch_size, 3, self.height, self.width)
        return points2D

    def forward(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject(rp_K, cam_points, T)
        return pix_coords

    def get_unnormed_projects(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject_unnormed(rp_K, cam_points, T)
        return pix_coords

    def transform_depth(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject_transform(rp_K, cam_points, T)
        return pix_coords

