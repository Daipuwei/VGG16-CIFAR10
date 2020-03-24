# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 19:15
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : test_batch_image.py
# @Software: PyCharm

import os

from model.vgg16 import VGG16
from config.config import config

def run_main():
    """
       这是主函数
    """
    # 初始化参数配置类
    pre_model_path = os.path.abspath("xxx")
    cfg = config(pre_model_path=pre_model_path)

    # 初始化图像与结果路径
    image_dir = os.path.abspath("./test")

    # 初始化VGG16，并进行测试批量图像
    print(cfg.pre_model_path)
    vgg16 = VGG16(cfg)
    vgg16.test_single_image(image_dir,cfg.pre_model_path)

if __name__ == '__main__':
    run_main()