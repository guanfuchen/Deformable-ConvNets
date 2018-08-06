# coding=utf-8
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import _init_paths

import cv2
import time
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    # 训练回调频率
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import shutil
import numpy as np
import mxnet as mx

from symbols import *
from core import callback, metric
from core.loader import AnchorLoader
from core.module import MutableModule
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler


# RFCN网络端到端训练
def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    # 创建logger和对应的输出路径
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=True)
    # 特征symbol，从网络sym中获取rpn_cls_score_output
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    # 使能多GPU训练，每一张卡训练一个batch
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    # 加载数据集同时准备训练的imdb，使用+分割不同的图像数据集，比如2007_trainval+2012_trainval
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    # load gt roidb加载gt roidb，根据数据集类型，图像集具体子类，数据集根目录和数据集路径，同时配置相关TRAIN为FLIP来增广数据
    roidbs = [load_gt_roidb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                            flip=config.TRAIN.FLIP)
              for image_set in image_sets]
    # 合并不同的roidb
    roidb = merge_roidb(roidbs)
    # 根据配置文件中对应的过滤规则来滤出roi
    roidb = filter_roidb(roidb, config)
    # load training data
    # 加载训练数据，anchor Loader为对应分类和回归的锚点加载，通过对应的roidb，查找对应的正负样本的锚点，该生成器需要参数锚点尺度，ratios和对应的feature的stride
    train_data = AnchorLoader(feat_sym, roidb, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE, ctx=ctx,
                              feat_stride=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
                              anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING)

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))
    print('providing maximum shape', max_data_shape, max_label_shape)

    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    # 加载并且初始化参数，如果训练中是继续上次的训练，也就是RESUME这一flag设置为True
    if config.TRAIN.RESUME:
        print('continue training from ', begin_epoch)
        # 从前缀和being_epoch中加载RESUME的arg参数和aux参数，同时需要转换为GPU NDArray
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        sym_instance.init_weight(config, arg_params, aux_params)

    # check parameter shapes
    # 检查相关参数的shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    # 创造求解器
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    label_names = [k[0] for k in train_data.provide_label_single]

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in range(batch_size)],
                        max_label_shapes=[max_label_shape for _ in range(batch_size)], fixed_param_prefix=fixed_param_prefix)

    if config.TRAIN.RESUME:
        mod._preload_opt_states = '%s-%04d.states'%(prefix, begin_epoch)

    # decide training params
    # metric
    # 以下主要是RPN和RCNN相关的一些评价指标
    rpn_eval_metric = metric.RPNAccMetric()
    rpn_cls_metric = metric.RPNLogLossMetric()
    rpn_bbox_metric = metric.RPNL1LossMetric()
    eval_metric = metric.RCNNAccMetric(config)
    cls_metric = metric.RCNNLogLossMetric(config)
    bbox_metric = metric.RCNNL1LossMetric(config)
    # mxnet中合成的评估指标，可以增加以上所有的评估指标，包括rpn_eval_metrix、rpn_cls_metric、rpn_bbox_metric和rcnn_eval_metric、rcnn_cls_metric、rcnn_bbox_metric
    eval_metrics = mx.metric.CompositeEvalMetric()
    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)

    # callback
    # batch后的callback回调以及epoch后的callback回调
    # batch_end_callback是在训练一定batch_size后进行的相应回调，回调频率为frequent
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    # means和stds，如果BBOX是类无关的，那么means为复制means两个，否则复制数量为NUM_CLASSES
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    # epoch为一个周期结束后的回调
    epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True), callback.do_checkpoint(prefix, means, stds)]
    # decide learning rate
    # 以下主要根据不同的学习率调整策略来决定学习率，这里如voc中默认的初始lr为0.0005
    base_lr = lr
    # 学习率调整因子
    lr_factor = config.TRAIN.lr_factor
    # 学习率调整周期，lr_step一般格式为3, 5，表示在3和5周期中进行学习率调整
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    # 如果当前周期大于begin_epoch那么lr_epoch_diff为epoch-begin_epoch
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    print('lr_epoch', lr_epoch, 'begin_epoch', begin_epoch)
    # 通过当前的epoch来计算当前应该具有的lr
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    # learning rate调整机制，warmup multi factor scheduler预训练多因子调整器
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
    # optimizer
    # 优化器参数，包含momentum、wd、lr、lr_scheduler、rescale_grad和clip_gradient
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}

    if not isinstance(train_data, PrefetchingIter):
        print('!!!train_data is not PrefetchingIter!!!')
        train_data = PrefetchingIter(train_data)

    # train
    # 模型训练过程，输入train_data，评估指标包括eval_metrics等一系列指标，每一个epoch结束后进入epoch_end_callback，每一个batch结束后进入batch_end_callback，优化器使用sgd，同时优化参数、输入参数和辅助参数以及begin周期和end周期
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def main():
    print('Called with argument:', args)
    # 训练网络，从预训练模型，预训练epoch和model_prefix
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
