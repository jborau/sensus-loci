import warnings
from copy import deepcopy
from os import path as osp
from pathlib import Path
from typing import Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmdet3d.structures.det3d_data_sample import SampleList

import mmcv

def inference_mono_3d(model, img, cam2img):
    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    mono_img_info = dict({'CAM_BACK': {'img_path': '/home/javier/datasets/DAIR-V2X/single-infrastructure-side-mmdet/training/image_2/000006.png', 'cam2img': cam2img}})

    data_ = dict(
        images=mono_img_info,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d)

    data_ = test_pipeline(data_)
    torch_image = torch.tensor(img, dtype=torch.uint8)
    torch_image = torch_image.permute(2, 0, 1)
    data_['inputs']['img'] = torch_image

    data = [data_]

    collate_data = pseudo_collate(data)
        
    # forward the model
    with torch.no_grad():
        result = model.test_step(collate_data)

    return result