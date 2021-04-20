import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_one(x):
    """
    Normalize input tensor x base on batch dimension (dim=0) to -1~1
    """
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = (x-0.5)*2
    return x


def normalize(x, axis=None):
    if len(np.shape(x)) == 1:
        return x/(np.linalg.norm(x))
    else:
        return x/np.expand_dims(np.linalg.norm(x, axis=axis), axis=axis)


def angular_error(map1, map2):
    if map1.shape[0] == 3 and map2.shape[0] == 3:
        map1 = np.moveaxis(map1, 0, -1)
        map2 = np.moveaxis(map2, 0, -1)
    err = np.sum(map1 * map2, -1)
    ang_err = np.arccos(np.clip(err, -1, 1)) / np.pi * 180
    return ang_err


def angular_loss(out_nor, gt_nor):
    err_map = torch.clamp(torch.sum(out_nor * gt_nor, dim=1), min=-1, max=1)
    ang_err_map = torch.acos(err_map) / np.pi * 180
    return torch.mean(ang_err_map)


def cosine_loss(out_nor, gt_nor):
    err_map = 1 - torch.sum(out_nor * gt_nor, dim=1)
    loss = torch.mean(err_map)
    return loss


def coarse_fine_loss(output, gt_nor, gt_alb, alb_k=1, coarse_k=1):
    alb_k = alb_k/(alb_k+1)
    nor_k = 1/(alb_k+1)
    coarse_k = coarse_k / (coarse_k + 1)
    fine_k = 1 / (coarse_k + 1)

    out_nor, out_alb, c_nor, c_alb = output

    coarse_loss = alb_k*F.mse_loss(c_alb, gt_alb) + nor_k*F.mse_loss(c_nor, gt_nor)

    nor_loss = F.mse_loss(out_nor, gt_nor)
    alb_loss = F.mse_loss(out_alb, gt_alb)
    fine_loss = alb_k*alb_loss + nor_k*nor_loss
    return fine_k * fine_loss + coarse_k * coarse_loss, nor_loss, alb_loss


def small_large_loss(output, gt_nor_alb, alb_k=1, small_k=0.5, grad_k=0.5):
    alb_k = alb_k/(alb_k+1)
    nor_k = 1/(alb_k+1)
    small_k = small_k / (small_k + 1)
    large_k = 1 / (small_k + 1)

    large_nor, large_alb, small_nor, small_alb = output
    small_gt_normal, small_gt_albedo, large_gt_normal, large_gt_albedo = gt_nor_alb

    if small_k < 1e-3:
        small_nor_loss = 0
        small_alb_loss = 0
        small_loss = 0
    else:
        small_nor_loss = cosine_loss(small_nor, small_gt_normal)
        small_alb_loss = F.mse_loss(small_alb, small_gt_albedo)
        small_loss = alb_k*small_alb_loss + nor_k*small_nor_loss

    large_nor_loss = cosine_loss(large_nor, large_gt_normal)
    large_alb_loss = F.mse_loss(large_alb, large_gt_albedo)
    large_loss = alb_k*large_alb_loss + nor_k*large_nor_loss

    if small_k < 1e-3:
        grad_small_loss = 0
    else:
        grad_small_nor_loss = GradientLoss(small_nor, small_gt_normal)
        grad_small_alb_loss = GradientLoss(small_alb, small_gt_albedo)
        grad_small_loss = alb_k*grad_small_alb_loss + nor_k*grad_small_nor_loss

    grad_large_nor_loss = GradientLoss(large_nor, large_gt_normal)
    grad_large_alb_loss = GradientLoss(large_alb, large_gt_albedo)
    grad_large_loss = alb_k*grad_large_alb_loss + nor_k*grad_large_nor_loss

    total_loss = (large_k * large_loss + small_k * small_loss) + \
                 grad_k*(large_k * grad_large_loss + small_k * grad_small_loss)

    return total_loss, large_nor_loss, large_alb_loss, small_nor_loss, small_alb_loss


def GradientLoss(prediction_n, gt_n):
    # horizontal difference
    h_gradient = prediction_n[:,:,:,0:-2] - prediction_n[:,:,:,2:]
    h_gradient_gt = gt_n[:,:,:,0:-2] -  gt_n[:,:,:,2:]
    h_gradient_loss = torch.abs(h_gradient - h_gradient_gt)

    # Vertical difference
    v_gradient = prediction_n[:,:,0:-2,:] - prediction_n[:,:,2:,:]
    v_gradient_gt = gt_n[:,:,0:-2,:] - gt_n[:,:,2:,:]
    v_gradient_loss = torch.abs(v_gradient - v_gradient_gt)

    gradient_loss = (torch.mean(h_gradient_loss) + torch.mean(v_gradient_loss))/2

    return gradient_loss


def batchwise_scale_compute(gt, prediction):
    """
    Compute scale factor using least square.
    Input shape : (batch, channel, h, w)
    Output shapd : (batch, 1, 1, 1),   witch each value represent the batchwise scale factor
    """
    with torch.no_grad():
        scales = torch.empty(size=(gt.size(0), 1, 1, 1), dtype=gt.dtype, requires_grad=False, device=gt.device)
        for batch_id in range(gt.size(0)):
            gt_batch = gt[batch_id].flatten().unsqueeze(1)
            pred_batch = prediction[batch_id].flatten().unsqueeze(1)
            scale_batch, _ = torch.lstsq(gt_batch.to('cpu'), pred_batch.to('cpu'))
            scales[batch_id, 0, 0, 0] = scale_batch[0, 0].clone().detach().to(gt.device)
    return scales


