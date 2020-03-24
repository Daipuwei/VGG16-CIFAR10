# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 11:32
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : plot_utils.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from tensorflow.core.framework import summary_pb2

def make_summary(name, value):
    """
    这是在tensorboard中创建视图的函数
    :param name: 视图的名称
    :param value: 数值
    :return:
    """
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=value)])

def plot_scalar(x,y,legend_arr,xlabel,ylabel,path):
    """
    这是绘制折现走势图的函数
    :param x: x轴下标数组
    :param y: y轴数值集合
    :param legend_arr: 图例
    :param xlabel: x坐标名称
    :param ylabel: y坐标名称
    :param path: 图像保存地址
    :return:
    """
    train_y, val_y = y
    plt.plot(x, train_y, 'r-')
    plt.plot(x, val_y, 'b--')
    plt.grid(True)
    plt.xlim(0, x[-1] + 2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend_arr, loc="best")
    plt.savefig(path)
    plt.close()

def plot_accuracy(x,y,path):
    """
    这是绘制精度的函数
    :param x: x轴下标数组
    :param y: y轴数值集合
    :param path: 结果保存地址
    :return:
    """
    legend_array = ["train_acc", "val_acc"]
    xlabel = "epoch"
    ylabel = "accuracy"
    plot_scalar(x,y,legend_array,xlabel,ylabel,path)

def plot_loss(x,y,path):
    """
    这是绘制损失的函数
    :param x: x轴下标数组
    :param y: y轴数值集合
    :param path: 结果保存地址
    :return:
    """
    legend_array = ["train_loss", "val_loss"]
    xlabel = "epoch"
    ylabel = "loss"
    plot_scalar(x,y,legend_array,xlabel,ylabel,path)