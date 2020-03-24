# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 11:39
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : cifar_preprocess.py
# @Software: PyCharm

import os
import cv2
import numpy as np
import pickle as pk

from multiprocessing import Process

def unpickle(data_path):
    """
    这是解压pickle数据的函数
    :param data_path: 数据路径
    """
    # 解压数据
    with open(data_path,'rb') as f:
        data_dict = pk.load(f,encoding='latin1')
        # 获取标签，
        #labels = np.array(data_dict['fine_labels'])          # CIFAR100数据集
        labels = np.array(data_dict['labels'])                # CIFAR10数据集
        # 获取图像数据
        data = np.array(data_dict['data'])
        # 转换图像数据形状,形状为(10000,32,32,3)
        size = len(data)
        data = np.reshape(data,(size,3,32,32)).transpose(0,2,3,1).astype("float")
    return data,labels

def save_single_iamge(image,image_path):
    """
    这是保存单张图像的函数
    :param image: 图像
    :param image_path: 图像地址
    :return:
    """
    cv2.imwrite(image_path,image)

def save_batch_images(batch_images,batch_image_ptahs):
    """
    这是保存批量图像的函数
    :param batch_images: 批量图像
    :param batch_image_ptahs: 图量图像地址
    :return:
    """
    for image,image_path in zip(batch_images,batch_image_ptahs):
        save_single_iamge(image,image_path)

def cifar_preprocess(cifar_dataset_dir,new_cifar_dataset_dir,val_rate,batch_size):
    """
    这是对CIFAR数据集进行预处理的函数
    :param cifar_dataset_dir: cifar数据集
    :param new_cifar_dataset_dir: 新cifar数据集
    :param val_rate: 验证集划分比例
    :param batch_size: 小批量数据集规模
    :return:
    """
    # 初始化路径原始CIFAR10数据集训练集和测试集路径
    train_batch_paths = [os.path.join(cifar_dataset_dir,"data_batch_%d"%(i+1)) for i in range(5)]
    test_batch_path = os.path.join(cifar_dataset_dir,'test_batch')

    # 初始化新格式下CIFAR10数据集的训练、验证和测试集目录
    new_train_dataset_dir = os.path.join(new_cifar_dataset_dir, "train")
    new_val_dataset_dir = os.path.join(new_cifar_dataset_dir, 'val')
    new_test_dataset_dir = os.path.join(new_cifar_dataset_dir, 'test')
    if not os.path.exists(new_train_dataset_dir):
        os.mkdir(new_train_dataset_dir)
    if not os.path.exists(new_val_dataset_dir):
        os.mkdir(new_val_dataset_dir)
    if not os.path.exists(new_test_dataset_dir):
        os.mkdir(new_test_dataset_dir)

    # 解析原始数据集文件
    train_val_data = []
    train_val_labels = []
    for i,train_batch_path in enumerate(train_batch_paths):
        batch_images,batch_labels = unpickle(train_batch_path)
        if i == 0:
            train_val_data = batch_images
            train_val_labels = batch_labels
        else:
            train_val_data = np.concatenate((train_val_data,batch_images),axis=0)
            train_val_labels = np.concatenate((train_val_labels,batch_labels),axis=0)
    test_data,test_labels = unpickle(test_batch_path)
    print(np.shape(train_val_data))
    print(np.shape(train_val_labels))
    print(np.shape(test_data))
    print(np.shape(test_labels))

    # 初始化训练验证集中，每个类别图像的目录
    label_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    for label_name in label_names:
        train_label_dir = os.path.join(new_train_dataset_dir,label_name)
        val_label_dir = os.path.join(new_val_dataset_dir, label_name)
        test_label_dir = os.path.join(new_test_dataset_dir, label_name)
        if not os.path.exists(train_label_dir):
            os.mkdir(train_label_dir)
        if not os.path.exists(val_label_dir):
            os.mkdir(val_label_dir)
        if not os.path.exists(test_label_dir):
            os.mkdir(test_label_dir)

    # 划分训练集和验证集
    train_val_index = np.arange(len(train_val_labels))
    train_images = []
    new_train_image_paths = []
    val_images = []
    new_val_image_paths = []
    # 遍历每种类别，分别获取每种类别图像，然后按照比例划分成训练和测试集，并生成对应图像的路径
    for i,label_name in enumerate(label_names):
        # 获取指定类别的下标
        label_index = np.random.permutation(train_val_index[train_val_labels == i])
        # 划分训练集和验证集
        val_num = int(len(label_index) * val_rate)
        val_label_index = label_index[0:val_num]
        train_label_index = label_index[val_num:]
        train_images.append(train_val_data[train_label_index])
        val_images.append(train_val_data[val_label_index])
        # 生成指定类别训练集图像的路径
        batch_new_train_image_paths = []
        for j,index in enumerate(train_label_index):
            image_name = "%06d.jpg"%(j)
            new_train_image_path = os.path.join(new_train_dataset_dir,label_name,image_name)
            batch_new_train_image_paths.append(new_train_image_path)
        new_train_image_paths.append(batch_new_train_image_paths)
        # 生成指定类别验证集图像的路径
        batch_new_val_image_paths = []
        for j, index in enumerate(val_label_index):
            image_name = "%06d.jpg" % (j)
            new_val_image_path = os.path.join(new_val_dataset_dir, label_name, image_name)
            batch_new_val_image_paths.append(new_val_image_path)
        new_val_image_paths.append(batch_new_val_image_paths)

    # 遍历每种类别，分别获取每种类别的测试图像，并生成对应测试图像的路径
    test_index = np.arange(len(test_labels))
    test_images = []
    new_test_image_paths = []
    for i, label_name in enumerate(label_names):
        test_label_index = test_index[test_labels == i]
        test_images.append(test_data[test_label_index])
        batch_new_test_image_paths = []
        for j, index in enumerate(test_label_index):
            image_name = "%06d.jpg" % (j)
            new_test_image_path = os.path.join(new_test_dataset_dir, label_name, image_name)
            batch_new_test_image_paths.append(new_test_image_path)
        new_test_image_paths.append(batch_new_test_image_paths)

    # 按照每种类别，分别划分成小批量训练集，利用多进程保存图像
    print("Start preprocessing train dataset")
    for single_label_train_images,single_label_train_image_paths in zip(train_images,new_train_image_paths):
        size = len(single_label_train_image_paths)
        for start in np.arange(0,size,batch_size):
            end = int(np.min([start+batch_size,size]))               # 防止最后一组数量不足batch_size
            batch_train_images = single_label_train_images[start:end]
            batch_train_image_paths = single_label_train_image_paths[start:end]
            p = Process(target=save_batch_images,args=(batch_train_images,batch_train_image_paths))
            p.start()
            p.join()
    print("Finish preprocessing train dataset")

    # 按照每种类别，分别划分成小批量验证集，利用多进程保存图像
    print("Start preprocessing val dataset")
    for single_label_val_images,single_label_val_image_paths in zip(val_images,new_val_image_paths):
        size = len(single_label_val_image_paths)
        for start in np.arange(0,size,batch_size):
            end = int(np.min([start+batch_size,size]))              # 防止最后一组数量不足batch_size
            batch_val_images = single_label_val_images[start:end]
            batch_val_image_paths = single_label_val_image_paths[start:end]
            p = Process(target=save_batch_images,args=(batch_val_images,batch_val_image_paths))
            p.start()
            p.join()
    print("Finish preprocessing val dataset")

    # 按照每种类别，分别划分成小批量测试集，利用多进程保存图像
    print("Start preprocessing test dataset")
    for single_label_test_images,single_label_test_image_paths in zip(test_images,new_test_image_paths):
        size = len(single_label_test_image_paths)
        for start in np.arange(0,size,batch_size):
            end = int(np.min([start+batch_size,size]))                  # 防止最后一组数量不足batch_size
            batch_test_images = single_label_test_images[start:end]
            batch_test_image_paths = single_label_test_image_paths[start:end]
            p = Process(target=save_batch_images,args=(batch_test_images,batch_test_image_paths))
            p.start()
            p.join()
    print("Finish preprocessing test dataset")

def run_main():
    """
       这是主函数
    """
    cifar_dataset_dir = os.path.abspath("./data/cifar-10-batches-py")
    new_cifar_dataset_dir = os.path.abspath("./data/cifar10")
    val_rate = 0.2
    batch_size = 100
    cifar_preprocess(cifar_dataset_dir, new_cifar_dataset_dir,val_rate,batch_size)

if __name__ == '__main__':
    run_main()