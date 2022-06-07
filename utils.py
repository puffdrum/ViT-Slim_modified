# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import math
import copy
import numpy as np


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def read_list_2d_int(filename):
    file1 = open(filename+".txt", "r")
    list_row =file1.readlines()
    list_source = []
    for i in range(len(list_row)):
        column_list = list_row[i].strip().split("\t")
        list_source.append(column_list)
    for i in range(len(list_source)):
        for j in range(len(list_source[i])):
            list_source[i][j]=int(list_source[i][j])
    file1.close()
    return list_source


def save_list_2d_int(list1, filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))
            file2.write('\t')
        file2.write('\n')
    file2.close()


def simulated_annealing_sparse_layerwise(pre_pis, cur_pis, w, cur_epoch, total_epochs):
    """
    This function updates layerwise regulation coefficient
    """
    pre_sum = 0
    cur_sum = 0
    for i in range(len(w)):
        pre_sum+=pre_pis[i]
        cur_sum+=cur_pis[i]
    
    pre_pis_layerwise = []
    cur_pis_layerwise = []
    for i in range(len(w)):
        pre_pis_layerwise.append(pre_pis[i]/pre_sum)
        cur_pis_layerwise.append(cur_pis[i]/cur_sum)
    
    T = cur_epoch/(total_epochs+1)
    for i in range(len(w)):
        w_new = w[i] + np.random.uniform(low=-0.000002, high=0.000002)*T
        if (0.00001<=w_new and w_new<=0.0007):
            delta_pi = cur_pis_layerwise[i] - pre_pis_layerwise[i]
            if delta_pi>0:
                w[i] = w_new
            elif pre_pis[i]*0.0005+delta_pi <= 0:
                pChange = np.random.uniform(low=0, high=1)
                if pChange>0.5:
                    w[i] = w_new
            else:
                pAccept = math.exp(138.63*(delta_pi)*total_epochs/T)
                pChange = np.random.uniform(low=0, high=1)
                if pChange < pAccept:
                    w[i] = w_new
    return w


def simulated_annealing_sparse_channelwise(pre_pis, cur_pis, w, cur_epoch, total_epochs):
    """
    This function updates channelwise regulation coefficient
    """
    pre_sum = 0
    cur_sum = 0
    for i in range(len(w)):
        pre_sum+=pre_pis[i]
        cur_sum+=cur_pis[i]
    
    pre_pis_channelwise = []
    cur_pis_channelwise = []
    for i in range(len(w)):
        pre_pis_channelwise.append(pre_pis[i]/pre_sum)
        cur_pis_channelwise.append(cur_pis[i]/cur_sum)
    
    T = cur_epoch/(total_epochs+1)
    for i in range(len(w)):
        w_new = w[i] + np.random.uniform(low=-0.000002, high=0.000002)*T
        if (0.00001<=w_new and w_new<=0.0007):
            delta_pi = cur_pis_channelwise[i] - pre_pis_channelwise[i]
            if delta_pi>0:
                w[i] = w_new
            elif pre_pis[i]*0.0005+delta_pi <= 0:
                pChange = np.random.uniform(low=0, high=1)
                if pChange>0.5:
                    w[i] = w_new
            else:
                pAccept = math.exp(138.63*(delta_pi)*total_epochs/T)
                pChange = np.random.uniform(low=0, high=1)
                if pChange < pAccept:
                    w[i] = w_new
    return w

def update_bp_methods(zetas_attn, zetas_mlp, zetas_patch, bp_methods, ft, rt):
    """
    This function updates back-propagation methods for zetas
    """
    for layer in range(len(zetas_attn)):
        for index in range(len(zetas_attn[layer])):
            if bp_methods[0][layer][index] == 0:
                if zetas_attn[layer][index] <= ft:
                    bp_methods[0][layer][index] = -1
                if zetas_attn[layer][index] > rt:
                    bp_methods[0][layer][index] = 1
    for layer in range(len(zetas_mlp)):
        for index in range(len(zetas_mlp[layer])):
            if bp_methods[1][layer][index] == 0:
                if zetas_mlp[layer][index] <= ft:
                    bp_methods[1][layer][index] = -1
                if zetas_mlp[layer][index] > rt:
                    bp_methods[1][layer][index] = 1
    for layer in range(len(zetas_patch)):
        for index in range(len(zetas_patch[layer])):
            if bp_methods[2][layer][index] == 0:
                if zetas_patch[layer][index] <= ft:
                    bp_methods[2][layer][index] = -1
                if zetas_patch[layer][index] > rt:
                    bp_methods[2][layer][index] = 1
    
    return bp_methods

def generate_criterion_w(bp_methods, w, value):
    """
    This function generates regulation coeeficients based on back-propagation methods for zetas
    """
    w_train = []
    w_train_attn = [w[0] if method == value else 0 for methods in bp_methods[0] for method in methods]
    w_train_mlp = [w[1] if method == value else 0 for methods in bp_methods[1] for method in methods]
    w_train_patch = [w[2] if method == value else 0 for methods in bp_methods[2] for method in methods]
    w_train.extend(w_train_attn)
    w_train.extend(w_train_mlp)
    w_train.extend(w_train_patch)

    return w_train

def update_grad_masks(grad_masks, bp_methods):
    """
    This function updates grad_masks for zetas
    """
    for layers in range(len(bp_methods)):
        for layer in range(len(bp_methods[layers])):
            for channel in range(len(bp_methods[layers][layer])):
                if bp_methods[layers][layer][channel] != 0:
                    grad_masks[layers][layer][channel] = 0

    return grad_masks




