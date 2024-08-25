# Copyright (c) 2023 42dot. All rights reserved.
import argparse
import os.path

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

import utils
from models import VFDepthAlgo
from trainer import VFDepthTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='VFdepth evaluation script')
    parser.add_argument('--config_file', default='./configs/surround_fusion.yaml', type=str, help='Config yaml file')
    parser.add_argument('--weight_path', default=None, type=str, help='Pretrained weight path')
    args = parser.parse_args()
    return args



def v_test_ddp(rank, cfg):
    print("Evaluating")
    utils.setup_ddp(rank, cfg['ddp']['world_size'], cfg)

    model = VFDepthAlgo(cfg, rank)
    if os.path.isdir(model.load_weights_dir):
        trainer = VFDepthTrainer(cfg, rank, use_tb=False)
        trainer.visualize(model, vis_results=True)
    utils.clear_ddp()


if __name__ == '__main__':
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='eval', weight_path=None)
    cfg['eval']['eval_batch_size']=1
    cfg['eval']['eval_num_workers']=1
    cfg['eval']['vis_only']=True
    cfg['eval']['eval_visualize']=True
    if args.weight_path is not None:
        epoch = args.weight_path

        subs =cfg['data']['load_weights_dir'].split('/')
        subs[-1] = 'weights_{}'.format(epoch)
        cfg['data']['load_weights_dir'] ='/'.join(subs)
    # Evaluating on DDP trained model
    if cfg['ddp']['ddp_enable'] == True:
        import torch.multiprocessing as mp

        mp.spawn(v_test_ddp, nprocs=cfg['ddp']['world_size'], args=(cfg,), join=True)

