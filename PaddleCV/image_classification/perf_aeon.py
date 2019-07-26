import time
import argparse
import functools
from utils.utility import add_arguments, print_arguments
import reader_aeon as reader
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('total_images', int, 1281167, "Training image number.")
add_arg('data_dir', str, "./data/ILSVRC2012/", "The ImageNet dataset root dir.")
add_arg('batch_size', int, 256, "Minibatch size.")
add_arg('num_epochs', int, 120, "number of epochs.")
add_arg('lower_scale', float, 0.08, "Set the lower_scale in ramdom_crop")
add_arg('lower_ratio', float, 3. / 4., "Set the lower_ratio in ramdom_crop")
add_arg('upper_ratio', float, 4. / 3., "Set the upper_ratio in ramdom_crop")
add_arg('resize_short_size', int, 256, "Set the resize_short_size")
add_arg('cache_dir', str, "", "Place where aeon will store cache")
add_arg('reader_thread_count', int, 12,
        "How many threads to allocate for reader")
add_arg('random_seed', int, 0, "Random seed. Choose 0 for non-deterministic.")
add_arg('drop_last', bool, True, "Skip last not batch if not full")


def perf(args):
    train_reader = reader.train(
        settings=args, batch_size=args.batch_size, drop_last=args.drop_last)

    if (args.drop_last):
        max_iter = np.floor(args.total_images / args.batch_size)
    else:
        max_iter = np.ceil(args.total_images / args.batch_size)

    latency = np.ones((int)(args.num_epochs * max_iter), dtype=float)
    fps = np.ones((int)(args.num_epochs * max_iter), dtype=float)
    for pass_id in range(args.num_epochs):
        print("Pass: {0}".format(pass_id))
        t1 = time.time()
        for batch_id, data in enumerate(train_reader()):
            t2 = time.time()
            latency[batch_id] = t2 - t1
            t1 = t2
            fps[batch_id] = args.batch_size / latency[batch_id]
            if batch_id % 100 == 0:
                print("Iteration {0}: latency: {1} s, fps: {2} img/s".format(
                    batch_id, latency[batch_id], fps[batch_id]))
    print("Mean latency: {0}, mean fps: {1}".format(
        latency.mean(), fps.mean()))
    print("Finished")


def main():
    args = parser.parse_args()
    print_arguments(args)
    perf(args)


if __name__ == '__main__':
    main()
