from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import math
import numpy as np
import argparse
import functools

import cv2

import distutils.util
import six

#-------------------------------------------------------------------------------
#           SCRIPT ARGUMENT UTILITIES
#-------------------------------------------------------------------------------

def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-------------  Configuration Arguments -------------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%25s : %s" % (arg, value))
    print("----------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)

#-------------------------------------------------------------------------------
#           SCRIPT PARAMETERS
#-------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('in_img_shape',      str,  '333,500,3',   'Input image shape.')
add_arg('out_img_shape',     str,  '3,224,224',   'Output image shape.')
add_arg('resize_short_size', int,  256,           'Set the size of shorther image edge.')
add_arg('lower_scale',       float,     0.08,      "Set the lower_scale in ramdom_crop")
add_arg('lower_ratio',       float,     3./4.,      "Set the lower_ratio in ramdom_crop")
add_arg('upper_ratio',       float,     4./3.,      "Set the upper_ratio in ramdom_crop")
add_arg('random_seed',       int,  1234,          'Random seed. Choose 0 for non-deterministic.')
add_arg('train_mode',        bool, True,          'Whether to use training set of transformations.')
add_arg('eval_mode',         bool, True,         'Whether to use validation set of transformations.')
add_arg('in_img_path',       str,  None,          'Path to input img file to use.')
add_arg('rotation_angle',     int, 5,              'The angle the image will be rotated.')
add_arg('random_crop_aspect_ratio', float, 0.85,   'The aspect ratio for random cropping.')
add_arg('area_scale',           float, 0.2,        'The area scale coefficient.')
add_arg('crop_y_offset',        int,   None,         'The y_offset for random croping.')
add_arg('crop_x_offset',        int,   None,         'The x_offset for random croping.')
add_arg('flip',                 bool,  True,       'Whether to do image horizontall flipping.')
add_arg('interpolation',        str,  'LINEAR',    'Interpolation method to use in resizing.')
add_arg('output_filename',      str,  'augment_output_linear',    'Output file name.')

#-------------------------------------------------------------------------------
#           IMAGE PROCESSING HELPERS
#-------------------------------------------------------------------------------

def rotate_image(img, settings):
    """ rotate_image """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = np.random.randint(-10, 11)
    if settings.angle:
        angle = settings.angle
    print('[rotate_image] angle: {}'.format(angle))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def random_crop(img, settings, scale=None, ratio=None):
    """ random_crop """
    lower_scale = settings.lower_scale
    lower_ratio = settings.lower_ratio
    upper_ratio = settings.upper_ratio
    scale = [lower_scale, 1.0] if scale is None else scale
    ratio = [lower_ratio, upper_ratio] if ratio is None else ratio

    aspect_ratio = np.random.uniform(*ratio)
    if settings.random_crop_aspect_ratio:
        aspect_ratio = settings.random_crop_aspect_ratio
    print('[random_crop] aspect_ratio: {}'.format(aspect_ratio))
    aspect_ratio = math.sqrt(aspect_ratio)
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    print('[random_crop] input img shape: {}'.format(img.shape))

    bound = min((float(img.shape[0]) / img.shape[1]) / (h**2),
                (float(img.shape[1]) / img.shape[0]) / (w**2))

    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    area_scale = np.random.uniform(scale_min, scale_max)
    if settings.area_scale:
        area_scale = settings.area_scale
    print('[random_crop] area_scale: {}'.format(area_scale))
    target_area = img.shape[0] * img.shape[1] * area_scale
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    y_offset = np.random.randint(0, img.shape[0] - h + 1)
    x_offset = np.random.randint(0, img.shape[1] - w + 1)
    if settings.crop_y_offset:
        y_offset = settings.crop_y_offset
    if settings.crop_x_offset:
        x_offset = settings.crop_x_offset

    print('[random_crop] w: {}'.format(w))
    print('[random_crop] h: {}'.format(h))
    print('[random_crop] y_offset: {}'.format(y_offset))
    print('[random_crop] x_offset: {}'.format(x_offset))

    img = img[y_offset:y_offset + h, x_offset:x_offset + w, :]
    return img


def distort_color(img):
    return img

def resize_short(img, target_size, settings):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    print('[resize_short] resized_width: {}'.format(resized_width))
    print('[resize_short] resized_height: {}'.format(resized_height))
    resized = cv2.resize(
        img,
        (resized_width, resized_height),
        interpolation=_get_interpolation_method(settings.interpolation)
        )
    return resized


def crop_image(img, target_size, center, settings):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
        if settings.crop_y_offset:
            h_start = settings.crop_y_offset
        if settings.random_crop_x_offset:
            w_start = settings.crop_x_offset

    print('[crop] w: {}'.format(size))
    print('[crop] h: {}'.format(size))
    print('[crop] x_offset: {}'.format(w_start))
    print('[crop] y_offset: {}'.format(h_start))
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def _bgr2rgb2CHW(img):
    return img[:, :, ::-1].transpose((2, 0, 1))


def _standardize(img, chw=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = img.astype('float32') / 255
    if chw:
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
    else:
        img_mean = np.array(mean).reshape((1, 1, 3))
        img_std = np.array(std).reshape((1, 1, 3))
    img -= img_mean
    img /= img_std
    return img

_interpolation_method_map = {
    'NEAREST':cv2.INTER_NEAREST,
    'LINEAR':cv2.INTER_LINEAR,
    'CUBIC':cv2.INTER_CUBIC,
    'AREA':cv2.INTER_AREA,
    'LANCZOS4':cv2.INTER_LANCZOS4,
}

def _get_interpolation_method(method):
    return _interpolation_method_map[method.upper()]

#-------------------------------------------------------------------------------
#           HIGH LEVEL FUNCTIONALITY
#-------------------------------------------------------------------------------

def train(settings,
          img,
          rotate=False,
          color_jitter=False,
          crop_size=224):
    if rotate:
        img = rotate_image(img)
    if crop_size > 0:
       img = random_crop(img, settings)
       img = cv2.resize(img, (crop_size, crop_size),
                        interpolation=_get_interpolation_method(settings.interpolation)
                        )
    if color_jitter:
        img = distort_color(img)
    # flip horizontally
    do_flip = np.random.randint(0, 2)
    if settings.flip:
        do_flip = settings.flip
    if bool(do_flip):
        img = img[:, ::-1, :]

    img = _bgr2rgb2CHW(img)
    img = _standardize(img)
    return img


def val(settings,
        img,
        crop_size=224):
    if crop_size > 0:
        target_size = settings.resize_short_size
        img = resize_short(img, target_size, settings)
        print('[val] resized image shape: {}'.format(img.shape))
        img = crop_image(img, crop_size, center=True, settings=settings)
        print('[val] center cropped image shape: {}'.format(img.shape))

    img = _bgr2rgb2CHW(img)
    return _standardize(img)


def main():
    args = parser.parse_args()
    print_arguments(args)

    np.random.seed(args.random_seed)

    if args.in_img_path:
        input_img = cv2.imread(args.in_img_path)
    else:
        in_img_shape = [int(x) for x in args.in_img_shape.split(',')]
        input_img = np.random.randint(0, 256, in_img_shape, dtype=np.uint8)
        cv2.imwrite('augment_input_img.jpg', input_img)

    crop_size = int(args.out_img_shape.split(',')[2])

    if args.train_mode:
        print('Running train augmentation set.')
        output_img = train(args, input_img, crop_size=crop_size)
        output_file = args.output_filename + '_train.bin'
        print('Writing result to file: {}'.format(output_file))
        with open(output_file, 'wb') as f:
            f.write(output_img.tobytes())

        # import ipdb; ipdb.set_trace()
        print('output img shape: {}'.format(output_img.shape))

        # OpenCV expect HWC order
        if output_img.shape[2] == 3 :
            cv2.imwrite(args.output_filename + '_train.jpg', output_img)

    if args.eval_mode:
        print('Running val augmentation set.')
        output_img = val(args, input_img, crop_size)
        output_file = args.output_filename + '_eval.bin'
        print('Writing result to file: {}'.format(output_file))
        with open(output_file, 'wb') as f:
            f.write(output_img.tobytes())

        # OpenCV expect HWC order
        if output_img.shape[2] == 3:
            cv2.imwrite(args.output_filename + '_eval.jpg', output_img)

if __name__ == '__main__':
    main()
