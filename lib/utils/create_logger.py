# coding=utf-8
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Bin Xiao
# --------------------------------------------------------

import os
import logging
import time

# 创建全局日志记录器
def create_logger(root_output_path, cfg, image_set):
    # set up logger
    # 首先创建该模型相关的全局路径
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    # 然后获取具体的config文件名
    cfg_name = os.path.basename(cfg).split('.')[0]
    config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    if not os.path.exists(config_output_path):
        os.makedirs(config_output_path)

    # 图像操作相应集合
    image_sets = [iset for iset in image_set.split('+')]
    final_output_path = os.path.join(config_output_path, '{}'.format('_'.join(image_sets)))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, final_output_path

