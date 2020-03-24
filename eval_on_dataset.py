# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 19:15
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : eval_on_dataset.py
# @Software: PyCharm

import os

from model.vgg16 import VGG16
from config.config import config
from tensorflow import keras as K

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    pre_model_path = os.path.abspath("xxxx")
    cfg = config(pre_model_path=pre_model_path)

    # 构造训练集和测试集数据生成器
    #dataset_dir = os.path.join(cfg.dataset_dir, "train")
    dataset_dir = os.path.join(cfg.dataset_dir, "test")
    image_data =  K.preprocessing.image.ImageDataGenerator()
    datagen = image_data.flow_from_directory(dataset_dir,
                                             class_mode='categorical',
                                             batch_size = 1,
                                             target_size=(224, 224),
                                             shuffle=False)

    # 初始化相关参数
    iter_num = datagen.samples       # 训练集1个epoch的迭代次数

    # 初始化VGG16，并进行训练
    vgg16 = VGG16(cfg)
    vgg16.eval_on_dataset(datagen,iter_num,weight_path = cfg.pre_model_path)

if __name__ == '__main__':
    run_main()