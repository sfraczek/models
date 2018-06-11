#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from se_resnext import SE_ResNeXt
from mobilenet import mobile_net
from inception_v4 import inception_v4
import reader
import argparse
import functools
import paddle.fluid.layers.ops as ops
from utility import add_arguments, print_arguments
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import math

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('batch_size',       int,  256,           "Minibatch size.")
add_arg('num_layers',       int,  50,            "How many layers for SE-ResNeXt model.")
add_arg('with_mem_opt',     bool, True,          "Whether to use memory optimization or not.")
add_arg('parallel_exe',     bool, True,          "Whether to use ParallelExecutor to train or not.")
add_arg('init_model',       str, None,           "Whether to use initialized model.")
add_arg('pretrained_model', str, None,           "Whether to use pretrained model.")
add_arg('lr_strategy',      str, "cosine_decay", "Set the learning rate decay strategy.")
add_arg('model',            str, "se_resnext",   "Set the network to use.", choices=["se_resnext", "mobilenet_ssd"])
add_arg('use_mkldnn',       bool, False,         "If set, use MKLDNN library.")
add_arg('use_gpu',          bool, True,          "Whether to use GPU or not.")
add_arg('iterations',       int,  0,             "The number of iterations. Zero or less means whole training set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('skip_test',        bool, True,          "Whether to skip test phase.")
add_arg('pass_num',         int,  120,           "The number of passes.")
add_arg('profile',          bool, False,         "If set, do profiling.")
add_arg('skip_batch_num',   int,  0,             "The number of first minibatches to skip as warm-up for better performance test.")
# yapf: enable


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        epoch = ops.floor(global_step / step_each_epoch)
        decayed_lr = learning_rate * \
                     (ops.cos(epoch * (math.pi / epochs)) + 1)/2
    return decayed_lr


def train_parallel_do(args,
                      learning_rate,
                      batch_size,
                      num_passes,
                      init_model=None,
                      pretrained_model=None,
                      model_save_dir='model',
                      parallel=True,
                      use_nccl=True,
                      lr_strategy=None,
                      layers=50):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=use_nccl)

        with pd.do():
            image_ = pd.read_input(image)
            label_ = pd.read_input(label)

            if args.model is 'se_resnext':
                out = SE_ResNeXt(
                    input=image_,
                    class_dim=class_dim,
                    layers=layers,
                    use_mkldnn=args.use_mkldnn)
            elif args.model is 'mobile_net':
                out = mobile_net(img=image_, class_dim=class_dim)
            else:
                out = inception_v4(img=image_, class_dim=class_dim)

            cost = fluid.layers.cross_entropy(input=out, label=label_)
            avg_cost = fluid.layers.mean(x=cost)
            acc_top1 = fluid.layers.accuracy(input=out, label=label_, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label_, k=5)
            pd.write_output(avg_cost)
            pd.write_output(acc_top1)
            pd.write_output(acc_top5)

        avg_cost, acc_top1, acc_top5 = pd()
        avg_cost = fluid.layers.mean(x=avg_cost)
        acc_top1 = fluid.layers.mean(x=acc_top1)
        acc_top5 = fluid.layers.mean(x=acc_top5)
    else:
        if args.model is 'se_resnext':
            out = SE_ResNeXt(
                input=image,
                class_dim=class_dim,
                layers=layers,
                use_mkldnn=args.use_mkldnn)
        elif args.model is 'mobile_net':
            out = mobile_net(img=image, class_dim=class_dim)
        else:
            out = inception_v4(img=image, class_dim=class_dim)

        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    inference_program = fluid.default_main_program().clone(for_test=True)

    if "piecewise_decay" in lr_strategy:
        bd = lr_strategy["piecewise_decay"]["bd"]
        lr = lr_strategy["piecewise_decay"]["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    elif "cosine_decay" in lr_strategy:
        step_each_epoch = lr_strategy["cosine_decay"]["step_each_epoch"]
        epochs = lr_strategy["cosine_decay"]["epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=learning_rate,
                step_each_epoch=step_each_epoch,
                epochs=epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    opts = optimizer.minimize(avg_cost)
    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        fluid.io.load_persistables(exe, init_model)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = paddle.batch(
        reader.train(cycle=args.iterations > 0), batch_size=batch_size)
    test_reader = paddle.batch(reader.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    for pass_id in range(num_passes):
        train_info = [[], [], []]
        test_info = [[], [], []]
        iters = 0
        batch_times = []

        for batch_id, data in enumerate(train_reader()):
            iters = batch_id
            if batch_id == args.skip_batch_num:
                profiler.reset_profiler()
            elif batch_id < args.skip_batch_num:
                print("Warm-up iteration")
            if args.iterations > 0 and batch_id == args.iterations + args.skip_batch_num:
                break
            t1 = time.time()
            loss, acc1, acc5 = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost, acc_top1, acc_top5])
            t2 = time.time()
            period = t2 - t1
            batch_time = period
            fps = args.batch_size / batch_time
            batch_times.append(batch_time)
            train_info[0].append(loss[0])
            train_info[1].append(acc1[0])
            train_info[2].append(acc5[0])
            print(
                "Train pass {0}, trainbatch {1}, loss {2}, acc1 {3}, acc5 {4} time {5}, latency: {6}, fps: {7}"
                .format(pass_id, batch_id, loss[0], acc1[0], acc5[0],
                        "%2.2f sec" % period, batch_time, fps))
            sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()

        if not args.skip_test:
            for batch_id, data in enumerate(test_reader()):
                t1 = time.time()
                loss, acc1, acc5 = exe.run(
                    inference_program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                t2 = time.time()
                period = t2 - t1
                test_info[0].append(loss[0])
                test_info[1].append(acc1[0])
                test_info[2].append(acc5[0])
                print(
                    "Test pass {0}, testbatch {1}, loss {2}, acc1 {3}, acc5 {4}, time {5}"
                    .format(pass_id, batch_id, loss[0], acc1[0], acc5[0],
                            "%2.2f sec" % period))
                sys.stdout.flush()

            test_loss = np.array(test_info[0]).mean()
            test_acc1 = np.array(test_info[1]).mean()
            test_acc5 = np.array(test_info[2]).mean()

            print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, " \
                "test_loss {4}, test_acc1 {5}, test_acc5 {6}"
                .format(pass_id, train_loss, train_acc1, train_acc5, test_loss,
                        test_acc1, test_acc5))
            sys.stdout.flush()

            model_path = os.path.join(model_save_dir + '/' + args.model,
                                      str(pass_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            fluid.io.save_persistables(exe, model_path)

        latencies = batch_times[args.skip_batch_num:]
        latency_avg = np.average(latencies)
        latency_pc99 = np.percentile(latencies, 99)
        fpses = np.divide(args.batch_size, latencies)
        fps_avg = np.average(fpses)
        fps_pc99 = np.percentile(fpses, 1)

        # Benchmark output
        print('\nTotal examples (incl. warm-up): %d' %
              (iters * args.batch_size))
        print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                                 latency_pc99))
        print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg,
                                                                 fps_pc99))


def train_parallel_exe(args,
                       learning_rate,
                       batch_size,
                       num_passes,
                       init_model=None,
                       pretrained_model=None,
                       model_save_dir='model',
                       parallel=True,
                       use_nccl=True,
                       lr_strategy=None,
                       layers=50):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    if args.model is 'se_resnext':
        out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
    elif args.model is 'mobile_net':
        out = mobile_net(img=image, class_dim=class_dim)
    else:
        out = inception_v4(img=image, class_dim=class_dim)

    cost = fluid.layers.cross_entropy(input=out, label=label)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    avg_cost = fluid.layers.mean(x=cost)

    test_program = fluid.default_main_program().clone(for_test=True)

    if "piecewise_decay" in lr_strategy:
        bd = lr_strategy["piecewise_decay"]["bd"]
        lr = lr_strategy["piecewise_decay"]["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    elif "cosine_decay" in lr_strategy:
        step_each_epoch = lr_strategy["cosine_decay"]["step_each_epoch"]
        epochs = lr_strategy["cosine_decay"]["epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=learning_rate,
                step_each_epoch=step_each_epoch,
                epochs=epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    opts = optimizer.minimize(avg_cost)

    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        fluid.io.load_persistables(exe, init_model)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = paddle.batch(
        reader.train(cycle=args.iterations > 0), batch_size=batch_size)
    test_reader = paddle.batch(reader.test(), batch_size=batch_size)

    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    use_cuda = args.use_gpu
    train_exe = fluid.ParallelExecutor(
        use_cuda=use_cuda, loss_name=avg_cost.name)
    test_exe = fluid.ParallelExecutor(
        use_cuda=use_cuda, main_program=test_program, share_vars_from=train_exe)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    for pass_id in range(num_passes):
        train_info = [[], [], []]
        test_info = [[], [], []]
        batch_times = []
        iters = 0

        for batch_id, data in enumerate(train_reader()):
            iters = batch_id
            if batch_id == args.skip_batch_num:
                profiler.reset_profiler()
            elif batch_id < args.skip_batch_num:
                print("Warm-up iteration")
            if args.iterations > 0 and batch_id == args.iterations + args.skip_batch_num:
                break
            t1 = time.time()
            loss, acc1, acc5 = train_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            batch_time = period
            fps = args.batch_size / batch_time
            batch_times.append(batch_time)
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            print(
                "Train pass {0}, trainbatch {1}, loss {2}, acc1 {3}, acc5 {4}, time {5}, latency: {6}, fps: {7}"
                .format(pass_id, batch_id, loss, acc1, acc5, "%2.2f sec" %
                        period, batch_time, fps))

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()

        if not args.skip_test:
            for batch_id, data in enumerate(test_reader()):
                t1 = time.time()
                loss, acc1, acc5 = test_exe.run(fetch_list,
                                                feed=feeder.feed(data))
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(np.array(loss))
                acc1 = np.mean(np.array(acc1))
                acc5 = np.mean(np.array(acc5))
                test_info[0].append(loss)
                test_info[1].append(acc1)
                test_info[2].append(acc5)
                print(
                    "Test pass {0}, testbatch {1}, loss {2}, acc1 {3}, acc5 {4}, time {5}"
                    .format(pass_id, batch_id, loss, acc1, acc5, "%2.2f sec" %
                            period))
                sys.stdout.flush()

            test_loss = np.array(test_info[0]).mean()
            test_acc1 = np.array(test_info[1]).mean()
            test_acc5 = np.array(test_info[2]).mean()

            print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, " \
                "test_loss {4}, test_acc1 {5}, test_acc5 {6}"
                .format(pass_id, train_loss, train_acc1, train_acc5, test_loss,
                        test_acc1, test_acc5))
            sys.stdout.flush()

            model_path = os.path.join(model_save_dir + '/' + args.model,
                                      str(pass_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            fluid.io.save_persistables(exe, model_path)

        latencies = batch_times[args.skip_batch_num:]
        latency_avg = np.average(latencies)
        latency_pc99 = np.percentile(latencies, 99)
        fpses = np.divide(args.batch_size, latencies)
        fps_avg = np.average(fpses)
        fps_pc99 = np.percentile(fpses, 1)

        # Benchmark output
        print('\nTotal examples (incl. warm-up): %d' %
              (iters * args.batch_size))
        print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                                 latency_pc99))
        print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg,
                                                                 fps_pc99))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    total_images = 1281167
    batch_size = args.batch_size
    step = int(total_images / batch_size + 1)
    num_epochs = args.pass_num

    learning_rate_mode = args.lr_strategy
    lr_strategy = {}
    if learning_rate_mode == "piecewise_decay":
        epoch_points = [30, 60, 90]
        bd = [e * step for e in epoch_points]
        lr = [0.1, 0.01, 0.001, 0.0001]
        lr_strategy[learning_rate_mode] = {"bd": bd, "lr": lr}
    elif learning_rate_mode == "cosine_decay":
        lr_strategy[learning_rate_mode] = {
            "step_each_epoch": step,
            "epochs": num_epochs
        }
    else:
        lr_strategy = None

    use_nccl = args.use_gpu
    # layers: 50, 152
    layers = args.num_layers
    method = train_parallel_exe if args.parallel_exe else train_parallel_do
    init_model = args.init_model if args.init_model else None
    pretrained_model = args.pretrained_model if args.pretrained_model else None
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                method(
                    args,
                    learning_rate=0.1,
                    batch_size=batch_size,
                    num_passes=num_epochs,
                    init_model=init_model,
                    pretrained_model=pretrained_model,
                    parallel=args.parallel_exe,
                    use_nccl=use_nccl,
                    lr_strategy=lr_strategy,
                    layers=layers)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                method(
                    args,
                    learning_rate=0.1,
                    batch_size=batch_size,
                    num_passes=num_epochs,
                    init_model=init_model,
                    pretrained_model=pretrained_model,
                    parallel=args.parallel_exe,
                    use_nccl=use_nccl,
                    lr_strategy=lr_strategy,
                    layers=layers)
    else:
        method(
            args,
            learning_rate=0.1,
            batch_size=batch_size,
            num_passes=num_epochs,
            init_model=init_model,
            pretrained_model=pretrained_model,
            parallel=args.parallel_exe,
            use_nccl=use_nccl,
            lr_strategy=lr_strategy,
            layers=layers)
