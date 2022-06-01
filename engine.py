# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode = True, use_amp=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                if hasattr(criterion, 'w1'):
                    loss = criterion(samples, outputs, targets, model)
                elif hasattr(criterion, 'w'):
                    loss = criterion(samples, outputs, targets, model)
                else:
                    loss = criterion(samples, outputs, targets)
        else:
            outputs = model(samples)
            if hasattr(criterion, 'w1'):
                loss = criterion(samples, outputs, targets, model)
            elif hasattr(criterion, 'w'):
                    loss = criterion(samples, outputs, targets, model)
            else:
                loss = criterion(samples, outputs, targets)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_partialbp(model: torch.nn.Module, criterion, criterion_regzero, criterion_regone, grad_masks,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer0: torch.optim.Optimizer, optimizer1: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode = True, use_amp=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                if hasattr(criterion, 'w1'):
                    loss = criterion(samples, outputs, targets, model)
                elif hasattr(criterion, 'w'):
                    loss = criterion(samples, outputs, targets, model)
                else:
                    loss = criterion(samples, outputs, targets)
                loss_regzero = criterion_regzero(model)
                loss_regone = criterion_regone(model)
        else:
            outputs = model(samples)
            if hasattr(criterion, 'w1'):
                loss = criterion(samples, outputs, targets, model)
            elif hasattr(criterion, 'w'):
                loss = criterion(samples, outputs, targets, model)
            else:
                loss = criterion(samples, outputs, targets)
            loss_regzero = criterion_regzero(model)
            loss_regone = criterion_regone(model)


        loss_value = loss.item()
        loss_regzero_value = loss_regzero.item()
        loss_regone_value = loss_regone.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if not math.isfinite(loss_regzero_value):
            print("Regzero loss is {}, stopping training".format(loss_regzero_value))
            sys.exit(1)
        if not math.isfinite(loss_regone_value):
            print("Regone loss is {}, stopping training".format(loss_regone_value))
            sys.exit(1)

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            is_second_order = hasattr(optimizer0, 'is_second_order') and optimizer0.is_second_order
            is_second_order = hasattr(optimizer1, 'is_second_order') and optimizer1.is_second_order
            hooks_attn, hooks_mlp, hooks_patch = model.add_hooks_zetas(grad_masks, device)
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
            model.remove_hooks_zetas(hooks_attn, hooks_mlp, hooks_patch)
            loss_scaler(loss_regzero, optimizer0, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
            loss_scaler(loss_regone, optimizer1, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
        else:
            model.zero_grad()
            loss.backward()
            loss_regzero.backward()
            loss_regone.backward()
            hooks_attn, hooks_mlp, hooks_patch = model.module.add_hooks_zetas(grad_masks, device)
            optimizer.step()
            model.module.remove_hooks_zetas(hooks_attn, hooks_mlp, hooks_patch)
            optimizer0.step()
            optimizer1.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value+loss_regzero_value+loss_regone_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
