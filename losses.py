# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

#class SearchingDistillationLoss(torch.nn.Module):
#    def __init__(self, base_criterion, device, attn_w=0.0001, mlp_w=0.0001, patch_w=0.0001):
#        super().__init__()
#        self.base_criterion = base_criterion
#        self.w1 = attn_w
#        self.w2 = mlp_w
#        self.w3 = patch_w
#        self.device = device
#
#    def forward(self, inputs, outputs, labels, model):
#        base_loss = self.base_criterion(inputs, outputs, labels)
#        sparsity_loss_attn, sparsity_loss_mlp, sparsity_loss_patch = model.module.get_sparsity_loss(self.device)
#        return  base_loss + self.w1*sparsity_loss_attn + self.w2*sparsity_loss_mlp + self.w3*sparsity_loss_patch#


class SearchingDistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion, device, attn_w=0.0001, mlp_w=0.0001, patch_w=0.0001):
        super().__init__()
        self.base_criterion = base_criterion
        self.w1 = attn_w
        self.w2 = mlp_w
        self.w3 = patch_w
        self.device = device

    def forward(self, inputs, outputs, labels, model):
        base_loss = self.base_criterion(inputs, outputs, labels)
        sparsity_loss_attn, sparsity_loss_mlp, sparsity_loss_patch = model.module.get_sparsity_loss(self.device)
        # discreteness_loss_attn, discreteness_loss_mlp, discreteness_loss_patch = model.module.get_discreteness_loss(self.device)
        # return  base_loss + self.w1*(sparsity_loss_attn + discreteness_loss_attn) + self.w2*(sparsity_loss_mlp + discreteness_loss_mlp) + self.w3*(sparsity_loss_patch + discreteness_loss_patch)
        return  base_loss + self.w1*sparsity_loss_attn + self.w2*sparsity_loss_mlp + self.w3*sparsity_loss_patch


class SearchingDistillationLossLayerWise(torch.nn.Module):
    def __init__(self, base_criterion, device, w):
        super().__init__()
        self.base_criterion = base_criterion
        self.w = w
        self.device = device

    def forward(self, inputs, outputs, labels, model):
        base_loss = self.base_criterion(inputs, outputs, labels)
        sparsity_loss_layerwise = model.module.get_sparsity_loss_layerwise(self.device, self.w)
        
        return  base_loss + sparsity_loss_layerwise


class SearchingDistillationLossChannelWise(torch.nn.Module):
    def __init__(self, base_criterion, device, w):
        super().__init__()
        self.base_criterion = base_criterion
        self.w = w
        self.device = device

    def forward(self, inputs, outputs, labels, model):
        base_loss = self.base_criterion(inputs, outputs, labels)
        sparsity_loss_channelwise = model.module.get_sparsity_loss_channelwise(self.device, self.w)
        
        return  base_loss + sparsity_loss_channelwise


class LossRegZero(torch.nn.Module):
    def __init__(self, device, w):
        super().__init__()
        self.w = w
        self.device = device

    def forward(self, model):
        loss_regularization_zero = model.module.get_sparsity_loss_channelwise(self.device, self.w)
        
        return  loss_regularization_zero

class LossRegOne(torch.nn.Module):
    def __init__(self, device, w):
        super().__init__()
        self.w = w
        self.device = device

    def forward(self, model):
        loss_regularization_one = model.module.get_sparsity_loss_channelwise_one(self.device, self.w)
        
        return  loss_regularization_one

