#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3


#
#
python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nusc_baseline_352_ddp_min_2_max_80.yaml' --nusc_type surrounddepth --port 12354
python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nusc_baseline_352_ddp_min_3_max_80.yaml' --nusc_type surrounddepth --port 12354
python -W ignore eval.py --config_file='./configs/ddp/nuscenes/nusc_baseline_352_ddp_min_3_max_80_again.yaml' --nusc_type surrounddepth --port 12354




