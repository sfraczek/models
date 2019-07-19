from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import reader_aeon as reader
import argparse
import functools
import models
from utils.learning_rate import cosine_decay
from utils.utility import add_arguments, print_arguments
import math

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  256,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('with_mem_opt',     bool, True,                "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('model',            str,  "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('resize_short_size', int, 256,                "Set resize short size")
add_arg('iterations',       int, 50000,                  "Quit after this many iterations")
# yapf: enable


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

    if model_name == "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
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

    val_reader = reader.val_reader(settings=args)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    test_info = [[], [], []]
    cnt = 0
    batch_id = 0
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img_mean = np.array(img_mean).reshape((3, 1, 1))
    img_std = np.array(img_std).reshape((3, 1, 1))
    while True:
        batch_id += 1
        data = val_reader.next()
        batch = {k: v for k, v in data}
        images = batch['image']
        labels = batch['label']

        images /= 255
        images -= img_mean
        images /= img_std

        t1 = time.time()
        feed_data = zip(images, labels)
        loss, acc1, acc5 = exe.run(
            test_program, fetch_list=fetch_list, feed=feeder.feed(feed_data))
        #  with open('objs_aeon.txt', 'w') as f:
        #      print(len(feed_data))
        #      np.set_printoptions(threshold=np.inf)
        #      print(feed_data[0])
        #      return
        t2 = time.time()
        period = t2 - t1
        loss = np.mean(loss)
        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)
        test_info[0].append(loss * len(data))
        test_info[1].append(acc1 * len(data))
        test_info[2].append(acc5 * len(data))
        cnt += len(data)
        if batch_id % 10 == 0:
            print("Testbatch {0},loss {1}, "
                  "acc1 {2},acc5 {3},time {4}".format(batch_id, \
                  "%.5f"%loss,"%.5f"%acc1, "%.5f"%acc5, \
                  "%2.2f sec" % period))
            sys.stdout.flush()

        if batch_id == args.iterations:
            break

    test_loss = np.sum(test_info[0]) / cnt
    test_acc1 = np.sum(test_info[1]) / cnt
    test_acc5 = np.sum(test_info[2]) / cnt

    print("Test_loss {0}, test_acc1 {1}, test_acc5 {2}".format(
        "%.5f" % test_loss, "%.5f" % test_acc1, "%.5f" % test_acc5))
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()