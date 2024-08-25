# Copyright (c) 2023 42dot. All rights reserved.
from external.dataset import get_transforms


def construct_dataset(cfg, mode, **kwargs):
    """
    This function constructs datasets.
    """
    # dataset arguments for the dataloader
    if mode == 'train':
        dataset_args = {
            'cameras': cfg['data']['cameras'],
            'back_context': cfg['data']['back_context'],
            'forward_context': cfg['data']['forward_context'],
            'data_transform': get_transforms('train', **kwargs),
            'depth_type': cfg['data']['depth_type'] if 'gt_depth' in cfg['data']['train_requirements'] else None,
            'with_input_depth': None,
            'scale_range': cfg['model']['fusion_level'] if 'fusion_level' in cfg['model'] else -1,
            'with_pose': 'gt_pose' in cfg['data']['train_requirements'],
            'with_mask': 'mask' in cfg['data']['train_requirements']
        }
        
    elif mode == 'val':
        dataset_args = {
            'cameras': cfg['data']['cameras'],
            'back_context': cfg['data']['back_context'],
            'forward_context': cfg['data']['forward_context'],
            'data_transform': get_transforms('train', **kwargs), # for aligning inputs without any augmentations
            'depth_type': cfg['data']['depth_type'] if 'gt_depth' in cfg['data']['val_requirements'] else None,
            'with_input_depth': None,
            'scale_range': cfg['model']['fusion_level'] if 'fusion_level' in cfg['model'] else -1,
            'with_pose': 'gt_pose' in cfg['data']['val_requirements'],
            'with_mask': 'mask' in cfg['data']['val_requirements']            
        }
        
    # DDAD dataset
    if cfg['data']['dataset'] == 'ddad':
        # from dataset.ddad_dataset_sf import DDADdatasetSF
        # dataset = DDADdatasetSF(
        #     cfg['data']['data_path'], mode,
        #     **dataset_args
        # )
        from dataset.ddad_dataset import DDADdataset
        dataset = DDADdataset(
            cfg, mode,
            **dataset_args
        )
    elif cfg['data']['dataset'] == 'nuscenes_org':
        from dataset.nusc_dataset_vf import NuScenesdataset
        dataset = NuScenesdataset(
            cfg['data']['data_path'], mode,
            **dataset_args
        )
    elif cfg['data']['dataset'] == 'nuscenes':
        from dataset.nusc_dataset import NUSCdataset
        dataset = NUSCdataset(
            cfg, mode,
            **dataset_args
        )
    else:
        raise ValueError('Unknown dataset: ' + cfg['data']['dataset'])
    return dataset

