import torch
import torch.nn as nn
import torch.nn.functional as F

def global_loss(pred, ssc_target):
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(1, n_classes):
        p = pred[:, i, :, :, :]
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.mse_loss(precision, torch.ones_like(precision))
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.mse_loss(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.zeros_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count

def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    torch.use_deterministic_algorithms(False)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, target.long())
    return loss


def CE_semantic_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    pred = F.softmax(pred, dim=1)
    mask_geometry = (target == 0) * 255
    sematic_target = target.clone() + mask_geometry

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, sematic_target.long())
    return loss


