# coding: utf8
import os
import functools
import numpy as np
import paddle
import io

import json
from aeon import DataLoader

DATA_DIR = './data/ILSVRC2012'

def _reader_creator(settings,
                    file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR,
                    pass_id_as_seed=0):
    def reader():
        # !!!!!!! PAMIETAJ aby potwierdzic ze dane wchodza w nchw rgb

        # co robi ten transpose 2,0,1
        # bgr -> rgb
        # hwc -> chw
        # subtract mean div by stdev
        # img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        # img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

        image_config = {"type": "image", "height": 224, "width": 224, 
                "channel_major": True}

        label_config = {"type": "label", "binary": True}

        augmentation_config = {"type": "image", "flip_enable": False}

            #"/mnt/drive/data/i1k/i1k-extracted/val-index.csv",
        aeon_config = {
            "manifest_filename": "/mnt/drive/data/ILSVRC2012_china/val_list.txt"
            "cache_directory": "/mnt/drive/.aeon-cache/",
            "etl": (image_config, label_config),
            "augment": (augmentation_config),
            "batch_size": 50
        }

        dl = DataLoader(json.dumps(aeon_config))
        #  data = dl.next()
        #  batch = {k: v for k, v in data}
        #  images = batch['image']
        #  labels = batch['label']

        yield dl.next()


def val(settings, data_dir=DATA_DIR):
    return _reader_creator(
        settings, '', 'val', shuffle=False, data_dir=data_dir)


def test(settings, data_dir=DATA_DIR):
    return _reader_creator(
        settings, '', 'test', shuffle=False, data_dir=data_dir)
