# coding: utf8
import os
# import json

from aeon import DataLoader

RANDOM_SEED = 1  # setting to 0 should yields random random seed (non deterministic)
THREAD = 12
VAL_LIST_FILENAME = "val-index.tsv"
TRAIN_LIST_FILENAME = "train-index.tsv"
CACHE_DIR = ".aeon-cache/"

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
        "channels": 3,
        "output_type": "float",
        "channel_major": True,
        "bgr_to_rgb": True
    }

    label_config = {"type": "label", "binary": False}

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

    manifest_root = settings.data_dir
    manifest_filename = os.path.join(manifest_root, TRAIN_LIST_FILENAME)
    cache_dir = CACHE_DIR
    config = dict()
    config["shuffle_enable"] = True
    config["shuffle_manifest"] = True
    config['decode_thread_count'] = THREAD
    config['manifest_filename'] = manifest_filename
    config['manifest_root'] = manifest_root
    config['cache_directory'] = cache_dir
    config['etl'] = [image_config, label_config]
    config['augmentation'] = [augmentation_config]
    config['batch_size'] = settings.batch_size
    config['iteration_mode'] = "INFINITE"

    #  print(json.dumps(config, indent=4))

    dl = DataLoader(config)
    return dl


def val_reader(settings):
    image_config = {
        "type": "image",
        "height": 224,
        "width": 224,
        "channels": 3,
        "output_type": "float",
        "channel_major": True,
        "bgr_to_rgb": True
    }

    label_config = {"type": "label", "binary": False}

    augmentation_config = {
        "random_seed": 1,
        "type": "image",
        "flip_enable": False,
        "center": True,
        "crop_enable": True,
        "scale": [244.0 / 256.0, 244.0 / 256.0],
        "resize_short_size": settings.resize_short_size,
    }

    manifest_root = settings.data_dir
    manifest_filename = os.path.join(manifest_root, VAL_LIST_FILENAME)
    cache_dir = CACHE_DIR

    config = dict()
    config['decode_thread_count'] = THREAD
    config['manifest_filename'] = manifest_filename
    config['manifest_root'] = manifest_root
    config['cache_directory'] = cache_dir
    config['etl'] = [image_config, label_config]
    config['augmentation'] = [augmentation_config]
    config['batch_size'] = settings.batch_size

    #  print(json.dumps(config, indent=4))

    dl = DataLoader(config)
    return dl
