# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 17:57
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : train_batch_images.py
# @Software: PyCharm

import os

from model.vgg16 import VGG16
from config.config import config

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    pre_model_path = os.path.abspath(
        "./pre_weights/20200320070419/Epoch004_ val_loss_0.693,val_accuracy_100.000%.ckpt")
    cfg = config(pre_model_path=pre_model_path)

    # 初始化图像与结果路径
    image_dir = os.path.abspath("./test")

    # 初始化UNet，并进行训练
    print(cfg.pre_model_path)
    vgg16 = VGG16(cfg)
    vgg16.test_single_image(image_dir,cfg.pre_model_path)

if __name__ == '__main__':
    run_main()