# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from ..builder import LOSSES
from .utils import weight_reduce_loss


# ##
# # version 1: use torch.autograd
# class LabelSmoothSoftmaxCEV1(nn.Module):
#     '''
#     This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
#     '''

#     def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
#         super(LabelSmoothSoftmaxCEV1, self).__init__()
#         self.lb_smooth = lb_smooth
#         self.reduction = reduction
#         self.lb_ignore = ignore_index
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, logits, label):
#         '''
#         Same usage method as nn.CrossEntropyLoss:
#             >>> criteria = LabelSmoothSoftmaxCEV1()
#             >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
#             >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
#             >>> loss = criteria(logits, lbs)
#         '''
#         # overcome ignored label
#         logits = logits.float() # use fp32 to avoid nan
#         with torch.no_grad():
#             num_classes = logits.size(1)
#             label = label.clone().detach()
#             ignore = label.eq(self.lb_ignore)
#             n_valid = ignore.eq(0).sum()
#             label[ignore] = 0
#             lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
#             lb_one_hot = torch.empty_like(logits).fill_(
#                 lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

#         logs = self.log_softmax(logits)
#         loss = -torch.sum(logs * lb_one_hot, dim=1)
#         loss[ignore] = 0
#         if self.reduction == 'mean':
#             loss = loss.sum() / n_valid
#         if self.reduction == 'sum':
#             loss = loss.sum()

#         return loss



##
# version 2: user derived grad computation
class LSRCrossEntropyFunctionV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, lb_smooth, lb_ignore):
        # prepare label
        num_classes = logits.size(1)
        lb_pos, lb_neg = 1. - lb_smooth, lb_smooth / num_classes
        label = label.clone().detach()
        ignore = label.eq(lb_ignore)
        n_valid = ignore.eq(0).sum()
        label[ignore] = 0
        lb_one_hot = torch.empty_like(logits).fill_(
            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = [a, torch.arange(logits.size(1)), *b]
        lb_one_hot[mask] = 0
        coeff = (num_classes - 1) * lb_neg + lb_pos

        ctx.variables = coeff, mask, logits, lb_one_hot

        loss = torch.log_softmax(logits, dim=1).neg_().mul_(lb_one_hot).sum(dim=1)
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        coeff, mask, logits, lb_one_hot = ctx.variables

        scores = torch.softmax(logits, dim=1).mul_(coeff)
        grad = scores.sub_(lb_one_hot).mul_(grad_output.unsqueeze(1))
        grad[mask] = 0
        return grad, None, None, None


class LabelSmoothSoftmaxCEV2(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV2, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, labels):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV2()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        losses = LSRCrossEntropyFunctionV2.apply(
                logits, labels, self.lb_smooth, self.lb_ignore)
        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            n_valid = (labels != self.lb_ignore).sum()
            losses = losses.sum() / n_valid
        return losses

# ##
# # version 3: implement wit cpp/cuda to save memory and accelerate
# import lsr_cpp
# class LSRCrossEntropyFunctionV3(torch.autograd.Function):
#     '''
#     use cpp/cuda to accelerate and shrink memory usage
#     '''
#     @staticmethod
#     @amp.custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, logits, labels, lb_smooth, lb_ignore):
#         losses = lsr_cpp.lsr_forward(logits, labels, lb_ignore, lb_smooth)

#         ctx.variables = logits, labels, lb_ignore, lb_smooth
#         return losses

#     @staticmethod
#     @amp.custom_bwd
#     def backward(ctx, grad_output):
#         logits, labels, lb_ignore, lb_smooth = ctx.variables

#         grad = lsr_cpp.lsr_backward(logits, labels, lb_ignore, lb_smooth)
#         grad.mul_(grad_output.unsqueeze(1))
#         return grad, None, None, None


# class LabelSmoothSoftmaxCEV3(nn.Module):

#     def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
#         super(LabelSmoothSoftmaxCEV3, self).__init__()
#         self.lb_smooth = lb_smooth
#         self.reduction = reduction
#         self.lb_ignore = ignore_index

#     def forward(self, logits, labels):
#         '''
#         Same usage method as nn.CrossEntropyLoss:
#             >>> criteria = LabelSmoothSoftmaxCEV3()
#             >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
#             >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
#             >>> loss = criteria(logits, lbs)
#         '''
#         losses = LSRCrossEntropyFunctionV3.apply(
#                 logits, labels, self.lb_smooth, self.lb_ignore)
#         if self.reduction == 'sum':
#             losses = losses.sum()
#         elif self.reduction == 'mean':
#             n_valid = (labels != self.lb_ignore).sum()
#             losses = losses.sum() / n_valid
#         return losses


@LOSSES.register_module()
class LabelSmoothLoss(nn.Module):

    def __init__(self,
                 lb_smooth=0.1,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=-100,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(LabelSmoothLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.reduction = reduction
        self.cls_criterion = LabelSmoothSoftmaxCEV2(lb_smooth=lb_smooth, reduction='none', ignore_index=ignore_index)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        # element-wise losses
        loss = self.cls_criterion(cls_score, label)

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
            
        loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)    
        loss_cls = self.loss_weight * loss

        return loss_cls
