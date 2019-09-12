from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import reader_aeon as aeon_reader
import reader_cv2 as pd_reader
import argparse
import functools
import models
from utils.learning_rate import cosine_decay
from utils.utility import add_arguments, print_arguments
import math

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',           int,  1,                   "Minibatch size.")
add_arg('use_gpu',              bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',            int,  1000,                "Class number.")
add_arg('image_shape',          str,  "3,224,224",         "Input image size")
add_arg('total_images',         int,   50000,              "Validation set size.")
add_arg('with_mem_opt',         bool, True,                "Whether to use memory optimization or not.")
add_arg('pretrained_model',     str,  "/root/data/ILSVRC2012/ResNet50_pretrained/", "Whether to use pretrained model.")
add_arg('model',                str,  "ResNet50",          "Set the network to use.")
add_arg('resize_short_size',    int,  256,                 "Set resize short size")
add_arg('iterations',           int,  100,                 "Quit after this many iterations")
add_arg('data_dir',             str, "/root/data/ILSVRC2012/", "The ImageNet dataset root dir.")
add_arg('cache_dir',            str, "",                   "Place where aeon will store cache")
add_arg('reader_thread_count',  int, 12,                   "How many threads to allocate for reader")
add_arg('random_seed',          int, 1,                    "Random seed. Choose 0 for non-deterministic.")
# yapf: enable

def compare_imgs(ref_data, out_data):
    max_err = 0
    for ref, out in zip(ref_data, out_data):
        if not np.allclose(ref, out):
            # ref_l1 = np.linalg.norm(ref.flatten(), ord=1)
            # out_l1 = np.linalg.norm(out.flatten(), ord=1)
            # rtol=1e-05
            # atol=1e-08
            # diff = abs(ref_l1 - out_l1)
            # rel_err = diff / (atol + rtol * abs(ref_l1))
            max_err = np.max(np.abs(ref - out))
            # print("[L1] ref: {}, out: {}, diff: {}, rel_err: {}"
            #       .format("%.6f"%ref_l1, "%.6f"%out_l1, "%.6f"%diff, "%.5f"%rel_err))
    return max_err

def run_infer(exe, batch_data, program, fetch_list, feeder):
    t1 = time.time()
    loss, acc1, acc5 = exe.run(
        program, fetch_list=fetch_list, feed=feeder.feed(batch_data))
    t2 = time.time()
    period = t2 - t1

    return (period, loss, acc1, acc5)

def eval(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(
        args.model, model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()

    out = model.net(input=image, class_dim=class_dim)
    cost, pred = fluid.layers.softmax_with_cross_entropy(
        out, label, return_softmax=True)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]
    if with_memory_optimization:
        fluid.memory_optimize(
            fluid.default_main_program(), skip_opt_set=set(fetch_list))

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fluid.io.load_persistables(exe, pretrained_model)

    pd_val_reader = paddle.batch(pd_reader.val(settings=args, data_dir=args.data_dir),
                                batch_size=args.batch_size)
    aeon_val_reader = aeon_reader.val(settings=args, batch_size=args.batch_size)()
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img_mean = np.array(img_mean).reshape((3, 1, 1))
    img_std = np.array(img_std).reshape((3, 1, 1))

    for batch_id, pd_data in enumerate(pd_val_reader()):
        aeon_data = next(aeon_val_reader)

        pd_images = [s[0] for s in pd_data]
        aeon_images = [s[0] for s in aeon_data]

        max_in_img_err = compare_imgs(pd_images, aeon_images)

        pd_results = run_infer(exe, pd_data, test_program, fetch_list, feeder)
        aeon_results = run_infer(exe, aeon_data, test_program, fetch_list, feeder)

        loss_err = np.max(abs(pd_results[1] - aeon_results[1]))
        acc1_err = np.max(abs(pd_results[2] - aeon_results[2]))
        acc5_err = np.max(abs(pd_results[3] - aeon_results[3]))

        print("Testbatch {},\tmax input img err: {},\tloss_err: {},\tacc1_err: {},\tacc5_err: {}"
              .format(batch_id, "%.6f"%max_in_img_err, "%.6f"%loss_err, "%.6f"%acc1_err,
                      "%.6f"%acc5_err))
        sys.stdout.flush()

        if batch_id == args.iterations:
            break

def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()

