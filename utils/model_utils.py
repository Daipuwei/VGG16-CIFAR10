# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 11:18
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : model_utils.py
# @Software: PyCharm

from tensorflow import keras as K

def binary_cross_entropy_loss(y_true,y_pred):
    """
    这是定义二分类交叉熵损失的函数
    :param y_true:真实标签
    :param y_pred:预测标签
    :return:
    """
    return K.losses.binary_crossentropy(y_true,y_pred)

def categorical_crossentropy_loss(y_true,y_pred):
    """
    这是定义多分类交叉熵损失的函数
    :param y_true:真实标签
    :param y_pred:预测标签
    :return:
    """
    return K.losses.categorical_crossentropy(y_true,y_pred)

def binary_accuracy(y_true,y_pred):
    """
    这是二分类的精度函数
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return:
    """
    return K.metrics.binary_accuracy(y_true,y_pred)

def categorical_accuracy(y_true,y_pred):
    """
    这是多分类的精度函数
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return:
    """
    return K.metrics.categorical_accuracy(y_true,y_pred)

# 这是数据平均类
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)