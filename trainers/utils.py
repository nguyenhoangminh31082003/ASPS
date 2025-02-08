import torch


def compute_dice(seg, gt):
    intersection = torch.sum(seg * gt)
    union = torch.sum(seg) + torch.sum(gt)
    return (2.0 * intersection + 1e-10) / (union + 1e-10)


def compute_iou(seg, gt):
    intersection = torch.sum(seg * gt)
    union = torch.sum(seg) + torch.sum(gt) - intersection
    return (intersection + 1e-10) / (union + 1e-10)
