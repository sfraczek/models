import os
import sys
import numpy as np
import argparse
import functools
import time

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from utility import add_arguments, print_arguments
from se_resnext import SE_ResNeXt
import reader

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  1,     "Minibatch size.")
add_arg('use_gpu',          bool, True,  "Whether to use GPU or not.")
add_arg('test_list',        str,  '',    "The testing data lists.")
add_arg('num_layers',       int,  50,    "How many layers for SE-ResNeXt model.")
add_arg('model_dir',        str,  '',    "The model path.")
add_arg('iterations',       int,  0,     "The number of iterations. Zero or less means whole test set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('skip_batch_num',   int,  0,     "The first num of minibatch num to skip, for better performance test.")
add_arg('use_mkldnn',       bool, False, "If set, use mkldnn library for speed up.")
add_arg('profile',          bool, True,  "If set, do profiling.")
# yapf: enable


def infer(args):
    class_dim = 1000
    image_shape = [3, 224, 224]
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    out = SE_ResNeXt(
        input=image,
        class_dim=class_dim,
        layers=args.num_layers,
        use_mkldnn=args.use_mkldnn)
    out = fluid.layers.softmax(input=out)

    inference_program = fluid.default_main_program().clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    if not os.path.exists(args.model_dir):
        raise ValueError("The model path [%s] does not exist." %
                         (args.model_dir))
    if not os.path.exists(args.test_list):
        raise ValueError("The test lists [%s] does not exist." %
                         (args.test_list))

    def if_exist(var):
        return os.path.exists(os.path.join(args.model_dir, var.name))

    fluid.io.load_vars(exe, args.model_dir, predicate=if_exist)

    test_reader = paddle.batch(
        reader.infer(
            args.test_list, cycle=args.iterations > 0),
        batch_size=args.batch_size)

    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    fetch_list = [out]

    TOPK = 1
    iters = 0
    batch_times = []

    for data in test_reader():
        if iters == args.skip_batch_num:
            profiler.reset_profiler()
        elif iters < args.skip_batch_num:
            print("Warm-up iteration")
        if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
            break
        start = time.time()
        result = exe.run(inference_program,
                         feed=feeder.feed(data),
                         fetch_list=fetch_list)
        batch_time = time.time() - start
        fps = args.batch_size / batch_time
        batch_times.append(batch_time)
        result = result[0]
        pred_label = np.argsort(result)[::-1][0][0]
        print("Test {0}-score {1}, class: {2}, latency: {3}, fps: {4}".format(
            iters, result[0][pred_label], pred_label, batch_time, fps))
        sys.stdout.flush()
        iters += 1

    latencies = batch_times[args.skip_batch_num:]
    latency_avg = np.average(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    fpses = np.divide(args.batch_size, latencies)
    fps_avg = np.average(fpses)
    fps_pc99 = np.percentile(fpses, 1)

    # Benchmark output
    print('\nTotal examples (incl. warm-up): %d' % (iters * args.batch_size))
    print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                             latency_pc99))
    print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg, fps_pc99))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                infer(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                infer(args)
    else:
        infer(args)
