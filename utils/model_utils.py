# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 11:18
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : model_utils.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow import keras as K

def categorical_crossentropy_loss(y_true,y_pred):
    """
    这是定义多分类交叉熵损失的函数
    :param y_true:真实标签
    :param y_pred:预测标签
    :return:
    """
    return tf.nn.softmax_cross_entropy_with_logits_v2(y_true,y_pred)

def categorical_accuracy(y_true,y_pred):
    """
    这是多分类的精度函数
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return:
    """
    correct_prediction = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
    # Operation calculating the accuracy of our predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

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