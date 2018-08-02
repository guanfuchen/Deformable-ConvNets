# coding=utf-8
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import numpy as np
from dataset import *


# 加载gt的roidb
def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False):
    """ load ground truth roidb """
    # 加载GT roidb，数据集命名名字，dataset_name，比如为PascalVOC，然后根据具体的图像集子集，根目录和数据集路径以及最后的路径
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    # 从对应的数据集中获取roidb
    roidb = imdb.gt_roidb()
    if flip:
        # 将roidb增广相应的flip
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                        proposal='rpn', append_gt=True, flip=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)

    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, append_gt)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


# 多个数据集的roidbs需要将这些rois合并为一个roi
def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


# 过滤一些没有有用的rois的roidb实体
def filter_roidb(roidb, config):
    """ remove roidb entries without usable rois """

    # roidb是否有效，图像至少有一个fg或者bg roi才是有效的
    def is_valid(entry):
        """ valid images have at least 1 fg or bg roi """
        overlaps = entry['max_overlaps']
        # 前景判断标准和背景判断标准，如果通过这个标准，那么该roi有效
        fg_inds = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
        bg_inds = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    # 如果rois不是有效的，那么过滤该roi
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    # 打印过滤以后的roidb实体
    print 'filtered %d roidb entries: %d -> %d' % (num - num_after, num, num_after)

    return filtered_roidb


def load_gt_segdb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False):
    """ load ground truth segdb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    segdb = imdb.gt_segdb()
    if flip:
        segdb = imdb.append_flipped_images_for_segmentation(segdb)
    return segdb


def merge_segdb(segdbs):
    """ segdb are list, concat them together """
    segdb = segdbs[0]
    for r in segdbs[1:]:
        segdb.extend(r)
    return segdb
