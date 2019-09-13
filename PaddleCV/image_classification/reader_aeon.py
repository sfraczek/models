# coding: utf8
from aeon import DataLoader
import numpy as np
import os

VAL_LIST = "val-index.tsv"
TRAIN_LIST = "train-index.tsv"
MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]


def common_config(shape,
                  cache_dir,
                  data_dir,
                  thread_count,
                  random_seed,
                  thr_affinity_low_bound,
                  thr_affinity_hi_bound):
    image_config = {
        "type": "image",
        "height": shape[1],
        "width": shape[2],
        "channels": shape[0],
        "output_type": "float",
        "channel_major": True,
        "bgr_to_rgb": True
    }

    label_config = {"type": "label", "binary": False}

    config = dict()
    config['random_seed'] = random_seed
    config['decode_thread_count'] = thread_count
    config['thread_affinity_low_bound'] = thr_affinity_low_bound
    config['thread_affinity_high_bound'] = thr_affinity_hi_bound
    config['manifest_root'] = data_dir
    config['cache_directory'] = cache_dir
    config['etl'] = [image_config, label_config]
    config['iteration_mode'] = "ONCE"

    return config


def train_reader(settings, batch_size):
    shape = map(int, settings.image_shape.split(','))
    config = common_config(shape,
                           settings.cache_dir,
                           settings.data_dir,
                           settings.reader_thread_count,
                           settings.random_seed,
                           settings.thread_affinity_low_bound,
                           settings.thread_affinity_high_bound)

    augmentation_config = {
        "type": "image",
        "flip_enable": True,
        "center": False,
        "crop_enable": True,
        "horizontal_distortion": [3. / 4., 4. / 3.],
        "do_area_scale": True,
        "scale": [0.08, 1],
        "resize_short_size": 0,
        "interpolation_method": "LINEAR",
        "mean": MEAN,
        "stddev": STDDEV
    }

    config["shuffle_enable"] = True
    config["shuffle_manifest"] = True
    config['manifest_filename'] = os.path.join(settings.data_dir, TRAIN_LIST)
    config['augmentation'] = [augmentation_config]
    config['batch_size'] = batch_size

    dl = DataLoader(config)
    return dl


def val_reader(settings, batch_size):
    shape = map(int, settings.image_shape.split(','))
    config = common_config(shape,
                           settings.cache_dir,
                           settings.data_dir,
                           settings.reader_thread_count,
                           settings.random_seed,
                           settings.thread_affinity_low_bound,
                           settings.thread_affinity_high_bound)

    scale = float(shape[1]) / settings.resize_short_size
    augmentation_config = {
        "type": "image",
        "flip_enable": False,
        "center": True,
        "crop_enable": True,
        "scale": [scale, scale],
        "resize_short_size": settings.resize_short_size,
        "interpolation_method": "LINEAR",
        "mean": MEAN,
        "stddev": STDDEV
    }

    config["shuffle_enable"] = False
    config["shuffle_manifest"] = False
    config['manifest_filename'] = os.path.join(settings.data_dir, VAL_LIST)
    config['augmentation'] = [augmentation_config]
    config['batch_size'] = batch_size

    dl = DataLoader(config)
    return dl


def train(settings, batch_size, drop_last=False):
    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(
            "batch_size should be a positive integeral value, but got batch_size={}"
            .format(batch_size))
    reader = train_reader(settings, batch_size)
    if drop_last == True:
        max_iter = np.floor(settings.total_images / batch_size)
    else:
        max_iter = np.ceil(settings.total_images / batch_size)

    def func():
        batch = 0
        for tup in reader:
            if batch < max_iter:
                batch += 1
                # tup is (('image',...),('label',...)) where ... stand is data
                yield zip(tup[0][1], tup[1][1])

    return func


def val(settings, batch_size, drop_last=False):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(
            "batch_size should be a positive integer value, but got batch_size={}"
            .format(batch_size))
    reader = val_reader(settings, batch_size)
    if drop_last == True:
        max_iter = np.floor(settings.total_images / batch_size)
    else:
        max_iter = np.ceil(settings.total_images / batch_size)

    def func():
        batch = 0
        for tup in reader:
            if batch < max_iter:
                batch += 1
                # tup is (('image',...),('label',...)) where ... stand is data
                yield zip(tup[0][1], tup[1][1])

    return func
