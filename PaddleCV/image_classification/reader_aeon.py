# coding: utf8
import json
from aeon import DataLoader

RANDOM_SEED = 1  # setting to 0 should yields random random seed (non deterministic)
THREAD = 14
DATA_DIR = "./data/ILSVRC2012"
VAL_LIST_DIR = "data/ILSVRC2012/val-index.tsv"
TRAIN_LIST_DIR = "data/ILSVRC2012/train-index.tsv"
CACHE_DIR = "/mnt/drive/.aeon-cache/"
MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]


def common_config():
    image_config = {
        "type": "image",
        "height": 224,
        "width": 224,
        "channels": 3,
        "mean": MEAN,
        "stddev": STDDEV,
        "output_type": "float",
        "channel_major": True,
        "bgr_to_rgb": True
    }

    label_config = {"type": "label", "binary": False}

    config = dict()
    config['random_seed'] = RANDOM_SEED
    config['decode_thread_count'] = THREAD
    config['manifest_root'] = DATA_DIR
    config['cache_directory'] = CACHE_DIR
    config['etl'] = [image_config, label_config]
    config['iteration_mode'] = "ONCE"

    return config


def train_reader(settings, batch_size):
    config = common_config()

    augmentation_config = {
        "random_seed": 1,
        "type": "image",
        "flip_enable": True,
        "center": False,
        "crop_enable": True,
        "horizontal_distortion": [3. / 4., 4. / 3.],
        "do_area_scale": True,
        "scale": [0.08, 1],
        "resize_short_size": 0,
    }

    config["shuffle_enable"] = True
    config["shuffle_manifest"] = True
    config['manifest_filename'] = TRAIN_LIST_DIR
    config['augmentation'] = [augmentation_config]
    config['batch_size'] = batch_size

    #  print(json.dumps(config, indent=4))

    dl = DataLoader(config)
    return dl


def val_reader(settings, batch_size):
    config = common_config()

    augmentation_config = {
        "random_seed": 1,
        "type": "image",
        "flip_enable": False,
        "center": True,
        "crop_enable": True,
        "scale": [244.0 / 256.0, 244.0 / 256.0],
        "resize_short_size": settings.resize_short_size,
    }

    config["shuffle_enable"] = False
    config["shuffle_manifest"] = False
    config['manifest_filename'] = VAL_LIST_DIR
    config['augmentation'] = [augmentation_config]
    config['batch_size'] = batch_size

    #  print(json.dumps(config, indent=4))

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

    def func():
        for tup in reader:
            #  batch = {k: v for k, v in tup}
            #  images = batch['image']
            #  labels = batch['label']
            #  yield zip(images, labels)
            if len(tup[0][1]) == batch_size:
                yield zip(tup[0][1], tup[1][1])
        if drop_last == False and len(tup[0][1]) != 0:
            yield zip(tup[0][1], tup[1][1])

    return func


def val(settings, batch_size, drop_last=False):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(
            "batch_size should be a positive integeral value, but got batch_size={}"
            .format(batch_size))
    reader = val_reader(settings, batch_size)

    def func():
        for tup in reader:
            #  batch = {k: v for k, v in tup}
            #  images = batch['image']
            #  labels = batch['label']
            #  yield zip(images, labels)
            if len(tup[0][1]) == batch_size:
                yield zip(tup[0][1], tup[1][1])
        if drop_last == False and len(tup[0][1]) != 0:
            yield zip(tup[0][1], tup[1][1])

    return func
