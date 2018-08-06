# coding=utf-8
# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import mxnet as mx


def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file. 从文件中load模型checkpoint
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights. 模型参数，网络权重的name字典
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states. 模型参数，网络辅助状态的name字典
    """
    # 根据不同的epoch和前缀来load模型参数dict
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    # 按照存储的dict不同的type将arg和aug加载到不同的arg_params和aux_params
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def convert_context(params, ctx):
    """
    转换模型参数到指定的ctx，比如cpu或者gpu
    :param params: dict of str to NDArray
    :param ctx: the context to convert to
    :return: dict of str of NDArray with context ctx
    """
    new_params = dict()
    for k, v in params.items():
        new_params[k] = v.as_in_context(ctx)
    return new_params


def load_param(prefix, epoch, convert=False, ctx=None, process=False):
    """
    wrapper for load checkpoint 加载checkpoint的包装器
    :param prefix: Prefix of model name. 模型名字的前缀
    :param epoch: Epoch number of model we would like to load. 想要加载的模型预训练周期
    :param convert: reference model should be converted to GPU NDArray first 是否首先将参考模型转换为GPU
    :param ctx: if convert then ctx must be designated. 如果需要转换为GPU那么ctx必须被设计
    :param process: model should drop any test 模型应该在任何测试中drop
    :return: (arg_params, aux_params)
    """
    arg_params, aux_params = load_checkpoint(prefix, epoch)
    if convert:
        # 如果需要convert但是ctx为None，那么转换为cpu，否则转换为gpu
        if ctx is None:
            ctx = mx.cpu()
        arg_params = convert_context(arg_params, ctx)
        aux_params = convert_context(aux_params, ctx)
    if process:
        tests = [k for k in arg_params.keys() if '_test' in k]
        for test in tests:
            arg_params[test.replace('_test', '')] = arg_params.pop(test)
    return arg_params, aux_params
