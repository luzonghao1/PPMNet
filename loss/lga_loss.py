import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


def LGA_loss(pred, ssc_target):
    bce_loss = nn.BCELoss(reduction='none')
    '''kernel = torch.tensor([[[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           [[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]],
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           ]).cuda().float()'''
    kernel = torch.tensor([[[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           ]).cuda().float()
    kernel = torch.reshape(kernel, (1, 1, 3, 3, 3))
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    '''mask = torch.logical_and((ssc_target != 255), (ssc_target != 0))'''
    mask = (ssc_target != 255)
    loss = 0
    count = 0
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        completion_target = torch.ones_like(ssc_target)
        completion_target[ssc_target != i] = 0

        if torch.sum(completion_target) > 0:
            mask_semantic = (completion_target.clone()).float()
            mask_semantic = torch.reshape(mask_semantic, (1, 1, 256, 256, 32))
            '''lga_weight = (torch.ones_like(mask_semantic) * 26 - F.conv3d(mask_semantic, kernel, stride=1,
                                                                        padding=1, bias=None)).squeeze(dim=1)'''
            lga_weight = (mask_semantic * 26 - F.conv3d(mask_semantic, kernel, stride=1,
                                                                         padding=1, bias=None)).abs().squeeze(dim=1)
            M_LGA = (lga_weight + 1.0)[mask].reshape(1, -1) #0.5 *

            # Get probability of class i
            p = (pred[:, i, :, :, :])[mask].reshape(1, -1)
            count += 1.0
            lga_loss = bce_loss(p, completion_target[mask].reshape(1, -1).float()) * M_LGA
            loss += lga_loss.mean()
    return loss / count


def sem_BCE_loss(pred, ssc_target):
    bce_loss = nn.BCELoss(reduction='none')
    kernel = torch.tensor([[[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           ]).cuda().float()
    kernel = torch.reshape(kernel, (1, 1, 3, 3, 3))

    completion_target = torch.ones_like(ssc_target)
    completion_target[ssc_target != 0] = 0

    mask = (completion_target.clone()).float()
    mask = torch.reshape(mask, (1, 1, 256, 256, 32))

    mask_ignore = ssc_target != 255
    #lga_weight = (torch.ones_like(mask) * 26 - F.conv3d(mask, kernel, stride=1, padding=1, bias=None)).squeeze(dim=1)
    lga_weight = (mask * 26 - F.conv3d(mask, kernel, stride=1, padding=1, bias=None)).abs().squeeze(dim=1)
    M_LGA = ( lga_weight + 1.0)[mask_ignore].reshape(1, -1) # 0.5 *

    loss = bce_loss((F.softmax(pred, dim=1)[:, 0, :, :, :])[mask_ignore].reshape(1, -1),
                    completion_target[mask_ignore].reshape(1, -1).float()) * M_LGA
    return loss.mean()


def CE_LGA_loss(pred, ssc_target, class_weight):
    #ce_loss = nn.CrossEntropyLoss(weight=class_weight, ignore_index=255, reduction='none')
    ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    '''kernel = torch.tensor([[[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           [[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]],
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           ]).cuda().float()'''
    kernel = torch.tensor([[[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           ]).cuda().float()
    kernel = torch.reshape(kernel, (1, 1, 5, 5, 5))
    '''mask = torch.logical_and((ssc_target != 255), (ssc_target != 0))'''
    mask = (ssc_target != 255)
    loss = 0
    n_classes = pred.shape[1]

    ce_M_LGA = torch.zeros_like(ssc_target).float()
    for i in range(0, n_classes):

        completion_target = torch.ones_like(ssc_target)
        completion_target[ssc_target != i] = 0

        if torch.sum(completion_target) > 0:

            mask_semantic = (completion_target.clone()).float()
            mask_semantic = torch.reshape(mask_semantic, (1, 1, 256, 256, 32))
            M_LGA = (mask_semantic * 124 - F.conv3d(mask_semantic, kernel, stride=1,
                    padding=2, bias=None)).abs().squeeze(dim=1)
            ce_M_LGA += M_LGA
        #loss = ce_loss(pred, ssc_target.long()) * (ce_M_LGA + 1.0)
        #loss = ce_loss(pred, ssc_target.long()) * torch.log((ce_M_LGA // 2) + 1.1) #F.softmax(pred, dim=1) 0.5 *
        loss = ce_loss(pred, ssc_target.long()) * torch.log((ce_M_LGA // 2) + 2.7182)
    return loss.mean()

def CE_LGA_class(pred, target, class_weight):
    #ce_loss = nn.CrossEntropyLoss(weight=class_weight, ignore_index=255, reduction='none')
    ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    '''kernel = torch.tensor([[[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           [[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]],
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           ]).cuda().float()'''
    kernel = torch.tensor([[[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           ]).cuda().float()
    kernel = torch.reshape(kernel, (1, 1, 5, 5, 5))

    '''mask = torch.logical_and((ssc_target != 255), (ssc_target != 0))'''
    mask_geometry = (target == 0) * 255
    ssc_target = target.clone() + mask_geometry

    loss = 0
    n_classes = pred.shape[1]

    ce_M_LGA = torch.zeros_like(ssc_target).float()
    for i in range(1, n_classes):

        completion_target = torch.ones_like(ssc_target)
        completion_target[ssc_target != i] = 0

        if torch.sum(completion_target) > 0:

            mask_semantic = (completion_target.clone()).float()
            mask_semantic = torch.reshape(mask_semantic, (1, 1, 256, 256, 32))
            M_LGA = (mask_semantic * 124 - F.conv3d(mask_semantic, kernel, stride=1,
                    padding=2, bias=None)).abs().squeeze(dim=1)
            ce_M_LGA += M_LGA
        #loss = ce_loss(pred, ssc_target.long()) * (ce_M_LGA + 1.0)
        #loss = ce_loss(pred, ssc_target.long()) * torch.log((ce_M_LGA // 2) + 1.1) #F.softmax(pred, dim=1) 0.5 *
        loss = ce_loss(pred, ssc_target.long()) * torch.log((ce_M_LGA // 2) + 2.7182)
    return loss.mean()

def CE_LGA2D_loss(pred, proj_label, seg_class_weight):
    #loss_func = torch.nn.CrossEntropyLoss(weight=seg_class_weight, ignore_index=0)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)

    '''kernel = torch.tensor([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]]).cuda().float()'''
    kernel = torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1]]).cuda().float()
    kernel = torch.reshape(kernel, (1, 1, 7, 7))

    loss = 0
    n_classes = pred.shape[1]
    ce_M_LGA = torch.zeros_like(proj_label).float()
    for i in range(1, n_classes):

        completion_target = torch.ones_like(proj_label)
        completion_target[proj_label != i] = 0
        if torch.sum(completion_target) > 0:

            mask_semantic = (completion_target.clone()).float()
            mask_semantic = torch.reshape(mask_semantic, (1, 1, 370, 1220))
            M_LGA = (mask_semantic * 48 - F.conv2d(mask_semantic, kernel, stride=1,
                    padding=3, bias=None)).abs().squeeze(dim=1)

            ce_M_LGA += M_LGA

        #loss = loss_func(pred, proj_label.long()) * (ce_M_LGA + 1.0) #1/8 0.125 *
        #loss = loss_func(pred, proj_label.long()) * torch.log((ce_M_LGA // 2) + 1.1)  # 1/8 0.125 *
        loss = loss_func(pred, proj_label.long()) * torch.log((ce_M_LGA // 2) + 2.7182)
    return loss.mean()