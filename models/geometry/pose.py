# Copyright (c) 2023 42dot. All rights reserved.
import torch

from .geometry_util import vec_to_matrix


class Pose:
    """
    Class for multi-camera pose calculation 
    """
    def __init__(self, cfg):
        self.read_config(cfg)
        
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def compute_pose(self, net, inputs):
        """
        This function computes multi-camera posse in accordance with the network structure.
        """
        if self.pose_model == 'fusion':
            pose = self.get_single_pose(net, inputs, None)
            pose = self.distribute_pose(pose, inputs['extrinsics'], inputs['extrinsics_inv'])
        elif self.pose_model =='front':
            pose = self.get_single_pose(net, inputs, self.pose_model)
            pose = self.distribute_pose(pose, inputs['extrinsics'], inputs['extrinsics_inv'])
        elif self.pose_model =='joint' or self.pose_model=='joint_trans' or self.pose_model=='joint_attn':
            pose = self.get_single_pose(net, inputs, self.pose_model)
            pose = self.distribute_pose_central(pose, inputs['extrinsics'], inputs['extrinsics_inv'])
        elif self.pose_model =='joint_front' or self.pose_model =='front_trans' or self.pose_model=='front_attn':
            pose = self.get_single_pose(net, inputs, self.pose_model)
            pose = self.distribute_pose(pose, inputs['extrinsics'], inputs['extrinsics_inv'])
        elif self.pose_model=='fsm' or self.pose_model=='fsm_pose':
            pose = {}
            for cam in range(self.num_cams):
                pose[('cam', cam)] = self.get_single_pose(net, inputs, cam)
        else:
            raise NotImplementedError
        return pose
    
    def get_single_pose(self, net, inputs ,cam):
        """
        This function computes pose for a single camera.
        """
        output = {}
        for f_i in self.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            frame_ids = [-1, 0] if f_i < 0 else [0, 1]
            axisangle, translation = net(inputs, frame_ids, cam)
            output[('cam_T_cam', 0, f_i)] = vec_to_matrix(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            if inputs.get('flip_version') is not None and inputs['flip_version']>=3:
                assert self.pose_model!='fsm'
                bs = output[('cam_T_cam', 0, f_i)].shape[0]
                res=[]
                for b in range(bs):
                    if inputs['flips'][b][0]>0:
                        mat = output[('cam_T_cam', 0, f_i)][b]
                        mat[0,1:] = -mat[0,1:]
                        mat[1:-1,0] = -mat[1:-1,0]
                    else:
                        mat = output[('cam_T_cam', 0, f_i)][b]
                    res.append(mat.unsqueeze(0))

                output[('cam_T_cam', 0, f_i)] = torch.cat(res)
        return output
        
    def distribute_pose(self, poses, exts, exts_inv):
        """
        This function distrubutes pose to each camera by using the canonical pose and camera extrinsics.
        (default: reference camera 0)
        """
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam',cam)] = {}
        # Refernce camera(canonical)
        ref_ext = exts[:, 0, ...]
        ref_ext_inv = exts_inv[:, 0, ...]
        for f_i in self.frame_ids[1:]:
            ref_T = poses['cam_T_cam', 0, f_i].float() # canonical pose      
            # Relative cameras(canonical)            
            for cam in range(self.num_cams):
                cur_ext = exts[:,cam,...]
                cur_ext_inv = exts_inv[:,cam,...]                
                cur_T = cur_ext_inv @ ref_ext @ ref_T @ ref_ext_inv @ cur_ext
    
                outputs[('cam',cam)][('cam_T_cam', 0, f_i)] = cur_T            
        return outputs

    def distribute_pose_central(self, poses, exts, exts_inv):
        """
        This function distrubutes pose to each camera by using the canonical pose and camera extrinsics.
        (default: reference camera 0)
        """
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}
        # Refernce camera(canonical)


        for f_i in self.frame_ids[1:]:
            ref_T = poses['cam_T_cam', 0, f_i].float()  # canonical pose
            # Relative cameras(canonical)
            for cam in range(self.num_cams):
                cur_ext = exts[:, cam, ...]
                cur_ext_inv = exts_inv[:, cam, ...]
                cur_T = cur_ext_inv  @ ref_T  @ cur_ext

                outputs[('cam', cam)][('cam_T_cam', 0, f_i)] = cur_T
        return outputs

    def compute_relative_cam_poses(self, inputs, outputs, cam):
        """
        This function computes spatio & spatio-temporal transformation for images from different viewpoints.
        """
        ref_ext = inputs['extrinsics'][:, cam, ...]
        target_view = outputs[('cam', cam)]
        
        rel_pose_dict = {}
        # precompute the relative pose
        if self.spatio:
            # current time step (spatio)
            for cur_index in self.rel_cam_list[cam]:
                # for partial surround view training
                if cur_index >= self.num_cams:
                    continue

                cur_ext_inv = inputs['extrinsics_inv'][:, cur_index, ...]
                rel_pose_dict[(0, cur_index)] = torch.matmul(cur_ext_inv, ref_ext)

        if self.spatio_temporal:
            # different time step (spatio-temporal)
            for frame_id in self.frame_ids[1:]:                 
                for cur_index in self.rel_cam_list[cam]:
                    # for partial surround view training
                    if cur_index >= self.num_cams:
                        continue

                    T = target_view[('cam_T_cam', 0, frame_id)]
                    # assuming that extrinsic doesn't change
                    rel_ext = rel_pose_dict[(0, cur_index)]
                    rel_pose_dict[(frame_id, cur_index)] = torch.matmul(rel_ext, T) # using matmul speed up
        return rel_pose_dict