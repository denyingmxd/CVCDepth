# CVCDepth

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in:

**Towards Cross-View-Consistent Self-Supervised Surround Depth Estimation**

Laiyan Ding, Hualie Jiang, Jie Li, Yongquan Chen, Rui Huang

[IROS 2024 (arXiv pdf)](https://arxiv.org/abs/2407.04041)

Please note that this code may have issue with batch-size larger than 1. You may refer to https://github.com/denyingmxd/FSM_stable/tree/master for some help. If I remember correct, the new code should be more efficient and robust.


This code is for non-commercial use.

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{ding2024towards,
  title={Towards Cross-View-Consistent Self-Supervised Surround Depth Estimation},
  author={Laiyan Ding, Hualie Jiang, Jie Li, Yongquan Chen, Rui Huang},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
  organization={IEEE}
}
```

# Data Preparation

Please refer to the codebase of [VFDepth](https://github.com/42dot/VFDepth) and [SurroundDepth](https://github.com/weiyithu/SurroundDepth) for data preparation. Our code also borrows largely from [VFDepth](https://github.com/42dot/VFDepth).

#Trained Models

You may find the trained models for resnet 18 and resnet 34 models on ddad and nuscens here: 链接: https://pan.baidu.com/s/1tnMe0jWVpvjG1JnQHnS0xw 提取码: ly66

# Environment Setup

Please refer to requirement.txt for the required packages.

# Training

To train the model, please run the following command:

```bash
python train.py \
--config_file ./configs/ddp/baseline_ddp_384_front_sp_con_pre_0.001_sptp_con_0.2_flipv5.yaml
```

# Test

To test the model, please run the following command:

```bash
python eval.py \
--config_file ./configs/ddp/baseline_ddp_384_front_sp_con_pre_0.001_sptp_con_0.2_flipv5.yaml \
--weight_path ./results/baseline_ddp_384_front_sp_con_pre_0.001_sptp_con_0.2_flipv5/models/weights_19
```
# Vis
Also, we provide the code for visualization. Please run the following command:

```bash
python vis.py \
--config_file ./configs/ddp/baseline_ddp_384_front_sp_con_pre_0.001_sptp_con_0.2_flipv5.yaml \
--weight_path 19
```

# Some Notes
1. As for the evaluation on the nuscenes dataset, please refer to the codebase of [SurroundDepth](https://github.com/weiyithu/SurroundDepth), [VFDepth](https://github.com/42dot/VFDepth) only provides results and test txt files on 
day-only scenes. You can set the ---nusc_type surrounddepth flag to True in the eval.py file to evaluate on the nuscenes dataset.
2. For the mask on the nuscenes dataset, different methods like [SurroundDepth](https://github.com/weiyithu/SurroundDepth), [VFDepth](https://github.com/42dot/VFDepth) have different mask settings. You may check it yourself, but the performance with
different masks are similar.
3. A large part of the training time are from calculating the loss in a for loop, you can refer to [SurroundDepth](https://github.com/weiyithu/SurroundDepth) for a more efficient implementation.
4. If you have any further questions, please feel free to contact me at 117010053@link.cuhk.edu.cn or file an issue.