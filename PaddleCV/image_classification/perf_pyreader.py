import argparse
import functools
import numpy as np
import reader_cv2 as reader
import sys
import time
from utils.utility import add_arguments, print_arguments

import paddle
import paddle.fluid as fluid

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('mode',                 str,    'train', "Choose training or evaluation", choices=['train','eval'])
add_arg('total_images',         int,    1281167, "Training image number.")
add_arg('image_shape',          str,    '3,224,224', "input image size")
add_arg('data_dir',             str,    '/root/data/ILSVRC2012/', "The ImageNet dataset root dir.")
add_arg('batch_size',           int,    256, "Minibatch size.")
add_arg('num_epochs',           int,    120, "number of epochs.")
add_arg('lower_scale',          float,  0.08, "Set the lower_scale in ramdom_crop")
add_arg('lower_ratio',          float,  3. / 4., "Set the lower_ratio in ramdom_crop")
add_arg('upper_ratio',          float,  4. / 3., "Set the upper_ratio in ramdom_crop")
add_arg('resize_short_size',    int,    256, "Set the resize_short_size")
add_arg('reader_thread_count',  int,    12, "How many threads to allocate for reader")
add_arg('random_seed',          int,    0, "Random seed. Choose 0 for non-deterministic.")
add_arg('drop_last',            bool,   True, "Skip last batch if not full")
add_arg('augment',              bool,   True, "Whether or not to perform image augmentation.")
add_arg('use_mixup',            bool,   False, "Whether to use mixup or not")
add_arg('dummy_data',           bool,   False, "Use dummy data reader.")


def perf(args):
    if args.mode == 'train':
        the_reader = paddle.batch(reader.train(settings=args, data_dir=args.data_dir),
                                  batch_size=args.batch_size,
                                  drop_last=args.drop_last)
    elif args.mode == 'eval':
        the_reader = paddle.batch(reader.val(settings=args, data_dir=args.data_dir),
                                  batch_size=args.batch_size)

    if args.drop_last == True:
        max_iter = np.floor(args.total_images / args.batch_size)
    else:
        max_iter = np.ceil(args.total_images / args.batch_size)

    latency = np.ones((int)(args.num_epochs * max_iter), dtype=float)
    fps = np.ones((int)(args.num_epochs * max_iter), dtype=float)
    index = 0
    for pass_id in range(args.num_epochs):
        print("Pass: {0}".format(pass_id))
        t1 = time.time()
        for batch_id, data in enumerate(the_reader(), 1):
            t2 = time.time()
            latency[index] = t2 - t1
            t1 = t2
            fps[index] = args.batch_size / latency[index]
            if batch_id % 100 == 0:
                print("Iteration {0}: latency: {1} s, fps: {2} img/s".format(
                    batch_id, latency[index], fps[index]))
                sys.stdout.flush()
            index += 1

    print("Mean latency: {0}, mean fps: {1}".format(
        latency.mean(), fps.mean()))
    print("Finished")


def main():
    args = parser.parse_args()
    print_arguments(args)
    perf(args)


if __name__ == '__main__':
    main()
