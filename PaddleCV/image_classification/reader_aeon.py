# coding: utf8
import os
import functools
import numpy as np
import paddle
import io

import json
from aeon import DataLoader

# dane wchodza rgb
# co robi ten transpose 2,0,1
# bgr -> rgb
# hwc -> chw
# subtract mean div by stdev
# img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
# img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

def train_reader(settings):
    image_config = {
        "type": "image",
        "height": 224,
        "width": 224,
        "output_type": "float",
        "channel_major": True
    }

    label_config = {"type": "label", "binary": False}

    augmentation_config = {
        "type": "image",
        "flip_enable": True,
        "center": False,
        "crop_enable": True,
        "scale": [224.0/256.0, 224.0/256.0]
    }

    manifest_filename = "/mnt/drive/data/i1k/i1k-extracted/train-index_copied.csv"
    manifest_root =  "/mnt/drive/data/ILSVRC2012_china/"
    cache_dir = "/mnt/drive/.aeon-cache/"

    config = dict()
    config['decode_thread_count'] = 14
    config['manifest_filename'] = manifest_filename
    config['manifest_root'] = manifest_root
    config['cache_directory'] = cache_dir
    config['etl'] = [image_config, label_config]
    config['augmentation'] = [augmentation_config]
    config['batch_size'] = settings.batch_size

    #  print(json.dumps(config, indent=4))

    dl = DataLoader(config)
    return dl


def val_reader(settings):
    image_config = {
        "type": "image",
        "height": 224,
        "width": 224,
        "output_type": "float",
        "channel_major": True
    }

    label_config = {"type": "label", "binary": False}

    augmentation_config = {
        "type": "image",
        "flip_enable": False,
        "center": True,
        "crop_enable": True,
        "scale": [224.0/256.0, 224.0/256.0]
    }

    manifest_filename = "/mnt/drive/data/i1k/i1k-extracted/val-index_copied.csv"
    manifest_root =  "/mnt/drive/data/ILSVRC2012_china/"
    cache_dir = "/mnt/drive/.aeon-cache/"

    config = dict()
    config['decode_thread_count'] = 14
    config['manifest_filename'] = manifest_filename
    config['manifest_root'] = manifest_root
    config['cache_directory'] = cache_dir
    config['etl'] = [image_config, label_config]
    config['augmentation'] = [augmentation_config]
    config['batch_size'] = settings.batch_size

    #  print(json.dumps(config, indent=4))

    dl = DataLoader(config)
    return dl
