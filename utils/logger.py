# Copyright (c) 2023 42dot. All rights reserved.
import os
import time
from collections import defaultdict
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as pil

from tensorboardX import SummaryWriter

from .visualize import colormap
from .misc import pretty_ts, cal_depth_error
import matplotlib
import matplotlib.pyplot as plt
from layers import BackprojectDepth

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

def set_tb_title(*args):
    """
    This function sets title for tensorboard plot.
    """    
    title = ''
    for i, s in enumerate(args):
        if not i%2: title += '/'
        s = s if isinstance(s, str) else str(s)
        title += s
    return title[1:]
    

def resize_for_tb(image):
    """
    This function resizes images for tensorboard plot.
    """     
    h, w = image.size()[-2:]
    return F.interpolate(image, [h//2, w//2], mode='bilinear', align_corners=True) 
    

def plot_tb(writer, step, img, title, j=0):
    """
    This function plots images on tensotboard.
    """     
    img_resized = resize_for_tb(img)    
    writer.add_image(title, img_resized[j].data, step)


def plot_norm_tb(writer, step, img, title, j=0):
    """
    This function plots normalized images on tensotboard.
    """     
    img_resized = torch.clamp(resize_for_tb(img), 0., 1.)
    writer.add_image(title, img_resized[j].data, step)


def plot_disp_tb(writer, step, disp, title, j=0):
    """
    This function plots disparity maps on tensotboard.
    """  
    disp_resized = resize_for_tb(disp).float()
    disp_resized = colormap(disp_resized[j, 0])
    writer.add_image(title, disp_resized, step)    

    
class Logger:
    """
    Logger class to monitor training
    """
    def __init__(self, cfg, use_tb=True):
        self.read_config(cfg)
        os.makedirs(self.log_path, exist_ok=True)
        
        if use_tb: 
            self.init_tb()
            
        if self.eval_visualize:
            self.init_vis()

        self._metric_names = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
        
    def init_tb(self):
        self.writers = {}
        for mode in ['train', 'val']:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        
    def close_tb(self):
        for mode in ['train', 'val']:
            self.writers[mode].close()

    def init_vis(self):
        vis_path = os.path.join(self.log_path, 'vis_results')
        os.makedirs(vis_path, exist_ok=True)
        
        self.cam_paths = []
        for cam_id in range(self.num_cams):
            cam_path = os.path.join(vis_path, f'cam{cam_id:d}')
            os.makedirs(cam_path, exist_ok=True)
            self.cam_paths.append(cam_path)
            
        if self.syn_visualize:
            self.syn_path = os.path.join(self.log_path, 'syn_results')
            os.makedirs(self.syn_path, exist_ok=True)
            
    def get_metric_names(self):
        return self._metric_names
    
    def update(self, mode, epoch, world_size, batch_idx, step, start_time, before_op_time, inputs, outputs, losses):
        """
        Display logs with respect to the log frequency
        """    
        # iteration duration
        duration = time.time() - before_op_time

        if self.is_checkpoint(step):
            self.log_time(epoch, batch_idx * world_size, duration, losses, start_time)
            self.log_tb(mode, inputs, outputs, losses, step)
                
    def is_checkpoint(self, step):
        """ 
        Log less frequently after the early phase steps
        """
        early_phase = (step % self.log_frequency == 0) and (step < self.early_phase)
        late_phase = step % self.late_log_frequency == 0
        return (early_phase or late_phase)

    def log_time(self, epoch, batch_idx, duration, loss, start_time):
        """
        This function prints epoch, iteration, duration, loss and spent time.
        """
        rep_loss = loss['total_loss'].item()
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - start_time
        print(f'epoch: {epoch:2d} | batch: {batch_idx:6d} |' + \
              f'examples/s: {samples_per_sec:5.1f} | loss: {rep_loss:.3f} | time elapsed: {pretty_ts(time_sofar)}')
        
    def log_tb(self, mode, inputs, outputs, losses, step):
        """
        This function logs outputs for monitoring using tensorboard.
        """
        writer = self.writers[mode]
        # loss
        for l, v in losses.items():
            writer.add_scalar(f'{l}', v, step)
        
        scale = 0 # plot the maximum scale
        for cam_id in range(self.num_cams):
            target_view = outputs[('cam', cam_id)]
            
            plot_tb(writer, step, inputs[('color', 0, scale)][:, cam_id, ...], set_tb_title('cam', cam_id)) # frame_id 0            
            plot_disp_tb(writer, step, target_view[('disp', scale)], set_tb_title('cam', cam_id, 'disp')) # disparity
            plot_tb(writer, step, target_view[('reproj_loss', scale)], set_tb_title('cam', cam_id, 'reproj')) # reprojection image
            plot_tb(writer, step, target_view[('reproj_mask', scale)], set_tb_title('cam', cam_id, 'reproj_mask')) # reprojection mask
            plot_tb(writer,  step, inputs['mask'][:, cam_id, ...], set_tb_title('cam', cam_id, 'self_occ_mask'))
    
            if self.spatio:
                plot_norm_tb(writer, step, target_view[('overlap', 0, scale)], set_tb_title('cam', cam_id, 'sp'))
                plot_tb(writer, step, target_view[('overlap_mask', 0, scale)], set_tb_title('cam', cam_id, 'sp_mask'))
                plot_tb(writer, step, target_view[('sp_loss', 0, scale)]*target_view[('overlap_mask', 0, scale)],set_tb_title('cam', cam_id, 'sp_loss'))  # reprojection image
                
            if self.spatio_temporal:
                for frame_id in self.frame_ids:
                    if frame_id == 0:
                        continue
                    plot_norm_tb(writer, step, target_view[('color', frame_id, scale)], set_tb_title('cam', cam_id, 'pred_', frame_id))                      
                    plot_norm_tb(writer, step, target_view[('overlap', frame_id, scale)], set_tb_title('cam', cam_id, 'sp_tm_', frame_id))
                    plot_tb(writer, step, target_view[('overlap_mask', frame_id, scale)], set_tb_title('cam', cam_id, 'sp_tm_mask_', frame_id))

            if hasattr(self,'spatial_depth_consistency_loss_weight'):
                plot_disp_tb(writer, step, target_view[('overlap_depth', 0, scale)], set_tb_title('cam', cam_id, 'sp_con'))

            if hasattr(self, 'sp_tp_recon_con_loss_weight'):
                for frame_id in self.frame_ids:
                    if frame_id == 0:
                        continue
                    plot_tb(writer, step, target_view[('sp_tp_recon_con_loss', scale,frame_id)],set_tb_title('cam', cam_id, 'sp_tp_recon_con_loss'))  # reprojection image
                    
            if self.aug_depth:
                plot_disp_tb(writer, step, target_view[('disp', scale, 'aug')], set_tb_title('view_aug', cam_id))                

    def log_result(self, inputs, outputs, idx, depth_eval_metric_cams,syn_visualize=False):
        """
        This function logs outputs for visualization.
        """        
        scale = 0

        for cam_id in range(self.num_cams):
            target_view = outputs[('cam', cam_id)]
            disps = target_view['disp', scale]
            pred_depths = target_view['depth', scale].cpu().numpy()
            _,_,h,w = disps.shape
            for jdx, disp in enumerate(disps):
                disp = colormap(disp)[0,...].transpose(1,2,0)
                disp = pil.fromarray((disp * 255).astype(np.uint8))
                cur_idx = idx*len(pred_depths) + jdx
                # disp.save(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_disp.jpg'))
                # plt.imshow(disp)
                # plt.axis('off')
                # plt.show()

                gt_d = inputs['depth'][jdx][cam_id][0].cpu().numpy()
                rgb = inputs[('color',0,0)][jdx][cam_id].cpu()
                # plt.imshow(rgb.permute((1,2,0)))
                # plt.axis('off')
                # plt.show()
                #plot rgb and lidar
                # rgb = pil.fromarray((rgb.permute(1,2,0).numpy()*255).astype(np.uint8))
                # rgb.save(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_rgb.jpg'))
                #plot error map
                # y, x = np.nonzero(gt_d)
                # plt.imshow(rgb);plt.scatter(x, y, c=gt_d[np.nonzero(gt_d)], s=0.1);plt.axis('off')
                # plt.savefig(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_rgb_lidar.jpg'))
                #
                pred_depth = pred_depths[jdx][0]
                np.savez_compressed(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_depth.npz'), pred_depth=pred_depth, gt_depth=gt_d)
                ratio = np.abs(pred_depth-gt_d) / gt_d
                ratio[gt_d==0]=0
                # y, x = np.nonzero(gt_d)
                # plt.plot()
                # print(123)
                # plt.scatter(x, disps.shape[2]-y, c=ratio[np.nonzero(ratio)], s=0.1)
                # plt.imshow(ratio,vmin=0,vmax=1,cmap='gray')
                # plt.axis('off')
                # plt.xlim(0,w)
                # plt.ylim(0,h)
                # plt.show()
                # plt.imshow(ratio,vmin=0,vmax=1);plt.axis('off')
                # plt.savefig(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_abs_rel.jpg'))
                # ratio = pil.fromarray(ratio)
                # plt.imsave(os.path.join(self.cam_paths[cam_id], f'{cur_idx:03d}_abs_rel.jpg'), ratio,vmin=0,vmax=1,cmap='gray')
                # exit()
                # break
            # break
            # exit()

        if syn_visualize:    
            syn_disps = outputs['disp_vis']
            for kdx, syn_disp in enumerate(syn_disps):
                syn_disp = colormap(syn_disp)[0,...].transpose(1,2,0)
                syn_disp = pil.fromarray((syn_disp * 255).astype(np.uint8))
                syn_disp.save(os.path.join(self.syn_path, f'{kdx:03d}_syndisp.jpg'))
                
    def compute_depth_losses(self, inputs, outputs, vis_scale=False):
        """
        This function computes depth metrics, to allow monitoring of training process on validation dataset.
        """
        min_eval_depth = self.eval_min_depth
        max_eval_depth = self.eval_max_depth

        med_scale = []

        error_metric_dict = defaultdict(float)
        error_median_dict = defaultdict(float)
        error_metric_dict_cams = {}

        for cam in range(self.num_cams):
            target_view = outputs['cam', cam]
            error_metric_dict_cams[cam] = defaultdict(float)
            depth_gt = inputs['depth'][:, cam, ...]
            _, _, h, w = depth_gt.shape

            depth_pred = target_view[('depth', 0)].to(depth_gt.device)
            depth_pred = torch.clamp(F.interpolate(
                        depth_pred, [h, w], mode='bilinear', align_corners=False),
                         min_eval_depth, max_eval_depth)
            depth_pred = depth_pred.detach()

            mask = (depth_gt > min_eval_depth) * (depth_gt < max_eval_depth)

            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]

            # calculate median scale
            scale_val = torch.median(depth_gt) / torch.median(depth_pred)
            med_scale.append(round(scale_val.cpu().numpy().item(), 2))

            depth_pred_metric = torch.clamp(depth_pred, min=min_eval_depth, max=max_eval_depth)
            depth_errors_metric = cal_depth_error(depth_pred_metric, depth_gt)

            depth_pred_median = torch.clamp(depth_pred * scale_val, min=min_eval_depth, max=max_eval_depth)
            depth_errors_median = cal_depth_error(depth_pred_median, depth_gt)

            for i in range(len(depth_errors_metric)):
                key = self._metric_names[i]
                error_metric_dict[key] += depth_errors_metric[i]
                error_median_dict[key] += depth_errors_median[i]
                error_metric_dict_cams[cam][key]+=depth_errors_metric[i]
        if vis_scale==True:
            # print median scale
            print(f'          | median scale = {med_scale}')

        for key in error_metric_dict.keys():
            error_metric_dict[key] = error_metric_dict[key].cpu().numpy() / self.num_cams
            error_median_dict[key] = error_median_dict[key].cpu().numpy() / self.num_cams
            for cam in range(self.num_cams):
                error_metric_dict_cams[cam][key] = error_metric_dict_cams[cam][key].cpu().numpy()

        return error_metric_dict, error_median_dict, error_metric_dict_cams

    def print_perf(self, loss, scale): 
        """
        This function prints various metrics for depth estimation accuracy.
        """
        perf = ' '*3 + scale
        for k, v in loss.items():
            perf += ' | ' + str(k) + f': {v:.3f}'
        print(perf)
            