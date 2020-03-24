# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 17:40
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : train_vgg16.py
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
    pre_model_path = None
    cfg = config(pre_model_path=pre_model_path)

    # 构造训练集和测试集数据生成器
    train_dataset_dir = os.path.join(cfg.dataset_dir,"train")
    val_dataset_dir = os.path.join(cfg.dataset_dir,"val")
    image_data =  K.preprocessing.image.ImageDataGenerator(rotation_range=0.2,
                                                           width_shift_range=0.05,
                                                           height_shift_range=0.05,
                                                           shear_range=0.05,
                                                           zoom_range=0.05,
                                                           horizontal_flip=True,
                                                           vertical_flip=True,
                                                           fill_mode='nearest')
    train_datagen = image_data.flow_from_directory(train_dataset_dir,
                                                   class_mode='categorical',
                                                   batch_size = cfg.batch_size,
                                                   target_size=(224,224),
                                                   shuffle=True)
    val_datagen = image_data.flow_from_directory(val_dataset_dir,
                                                 class_mode='categorical',
                                                 batch_size=cfg.batch_size,
                                                 target_size=(224, 224),
                                                 shuffle=True)

    # 初始化相关参数
    interval = 2                                                   # 验证间隔
    train_iter_num = train_datagen.samples // cfg.batch_size       # 训练集1个epoch的迭代次数
    val_iter_num = val_datagen.samples // cfg.batch_size           # 测试集1个epoch的迭代次数

    # 初始化VGG16，并进行测试批量图像
    vgg16 = VGG16(cfg)
    vgg16.train(train_generator = train_datagen,
               val_generatort = val_datagen,
               interval = interval,
               train_iter_num = train_iter_num,
               val_iter_num = val_iter_num,
               pre_model_path = cfg.pre_model_path)

if __name__ == '__main__':
    run_main()