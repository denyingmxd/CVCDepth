#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5,6,7

python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nsuc_baseline_352_ddp_front_min_1.5_sp_con_pre_0.001_sptp_con_0.1_flipv5.yaml' --nusc_type surrounddepth --port 12356
python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nsuc_baseline_352_ddp_front_min_1.5_sp_con_pre_0.001_sptp_con_0.1_flipv5_34.yaml' --nusc_type surrounddepth --port 12356 --post_process
python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nsuc_baseline_352_ddp_front_min_1.5_sp_con_pre_0.001_sptp_con_0.2_flipv5.yaml' --nusc_type surrounddepth --port 12356
python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nsuc_baseline_352_ddp_front_min_1.5_sp_con_pre_0.001_sptp_con_0.2_flipv5_34.yaml' --nusc_type surrounddepth --port 12356 --post_process