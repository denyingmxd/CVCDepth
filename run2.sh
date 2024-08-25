#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

#python -W ignore train.py --config_file='./configs/ddp/baseline_ddp_384.yaml'


python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nusc_baseline_352_ddp_min_1.0_front_sp_con_0.001_sptp_con_0.05_flipv5_34_106.yaml' --nusc_type surrounddepth --port 12355 --post_process
python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nsuc_baseline_352_ddp_front_min_1.0_sp_con_pre_0.001_sptp_con_0.01_flipv5_34.yaml' --nusc_type surrounddepth --port 12355 --post_process
python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nsuc_baseline_352_ddp_front_min_1.5_sp_con_pre_0.001_sptp_con_0.01_flipv5_34.yaml' --nusc_type surrounddepth --port 12355 --post_process