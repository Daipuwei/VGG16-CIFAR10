# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 10:47
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : vgg16.py
# @Software: PyCharm

import os
import cv2
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras as K

from utils.model_utils import AverageMeter
from utils.model_utils import categorical_accuracy
from utils.model_utils import categorical_crossentropy_loss

from utils.plot_utils import plot_loss
from utils.plot_utils import plot_accuracy
from utils.plot_utils import make_summary

class VGG16(object):

    def __init__(self,cfg):
        """
        这是VGG16的初始化函数
        :param cfg: 参数配置类
        """
        self.cfg = cfg

        # 定义相关变量
        self.real_label = tf.placeholder(tf.float32, shape=(None, self.cfg.label_num))          # 真丝标签
        self.global_step = tf.Variable(tf.constant(0), trainable=False)                         # 当前迭代次数
        self.init_learning_rate = tf.Variable(tf.constant(self.cfg.init_learning_rate))         # 初始学习率

        # 定义VGG16的前向传播过程
        self.build_model()
        """
        self.model.compile(optimizer=K.optimizers.SGD(learning_rate=self.cfg.init_learning_rate,
                                                      momentum=self.cfg.momentum_rate),
                           loss=categorical_crossentropy_loss)
        """
        self.model.compile(optimizer=K.optimizers.Adam(learning_rate=self.cfg.init_learning_rate),
                           loss="categorical_crossentropy")

        # 定义损失函数与精度
        self.loss = categorical_crossentropy_loss(self.real_label, self.pred_label)
        self.accuracy = categorical_accuracy(self.real_label, self.pred_label)

        # 定义学习率、优化器和训练过程
        """
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.init_learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=self.cfg.decay_step,
                                                        decay_rate=self.cfg.decay_rate,
                                                        staircase=True)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                    momentum=self.cfg.momentum_rate)
        """
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.init_learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # 定义模型保存类与加载类
        self.saver_save = tf.train.Saver(max_to_keep=100)  # 设置最大保存检测点个数为周期数

    def build_model(self):
        """
        这是VGG16网络的搭建函数
        :return:
        """
        self.image_input = K.layers.Input(shape=self.cfg.image_input_shape,name="image_input")
        # Block 1
        x = K.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1')(self.image_input)
        x = K.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv2')(x)
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = K.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv1')(x)
        x = K.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv2')(x)
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = K.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv1')(x)
        x = K.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv2')(x)
        x = K.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv3')(x)
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = K.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv1')(x)
        x = K.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv2')(x)
        x = K.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv3')(x)
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = K.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1')(x)
        x = K.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2')(x)
        x = K.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv3')(x)
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # classification block
        x = K.layers.Flatten(name='flatten')(x)
        x = K.layers.Dense(4096, activation='relu', name='fc1')(x)
        x = K.layers.Dense(4096, activation='relu', name='fc2')(x)
        self.pred_label = K.layers.Dense(self.cfg.label_num, activation='softmax', name='pred_label')(x)

        self.model = K.Model(self.image_input,self.pred_label)
        self.model.summary()

    def train(self,train_datagen,val_datagen,train_iter_num,val_iter_num,interval,pre_model_path=None):
        """
        这是VGG16的训练函数
        :param train_datagen: 训练数据集生成器
        :param val_datagen: 验证数据集生成器
        :param train_iter_num: 一个epoch训练迭代次数
        :param val_iter_num: 一个epoch验证迭代次数
        :param interval: 验证间隔
        :param pre_model_path: 预训练模型地址,与训练模型为ckpt文件，注意文件路径只需到.ckpt即可。
        """
        # 初始化相关文件目录路径
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = os.path.join(self.cfg.checkpoints_dir,time)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        log_dir = os.path.join(self.cfg.logs_dir, time)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        result_dir = os.path.join(self.cfg.result_dir, time)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.cfg.save_config(time)

        # 初始化训练损失和精度数组
        train_loss_results = []                     # 保存训练loss值
        train_accuracy_results = []                 # 保存训练accuracy值

        # 初始化验证损失和精度数组，验证最大精度
        val_ep = []
        val_loss_results = []                     # 保存验证loss值
        val_accuracy_results = []                 # 保存验证accuracy值
        val_acc_max = 0                           # 最大验证精度

        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 加载预训练模型
            if pre_model_path is not None:
                if ".ckpt" in pre_model_path:           # pre_model_path的地址写到.ckpt
                    saver_restore = tf.train.import_meta_graph(pre_model_path+".meta")
                    saver_restore.restore(sess,pre_model_path)
                    print(1)
                elif ".h5" in pre_model_path:
                    self.model.load_weights(pre_model_path,by_name=True)
                    print(2)
                print("restore model from : %s" % (pre_model_path))

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(log_dir, sess.graph)
            self.writer1 = tf.summary.FileWriter(os.path.join("./tf_dir"), sess.graph)

            print('\n----------- start to train -----------\n')

            total_global_step = self.cfg.epoch * train_iter_num
            for ep in np.arange(self.cfg.epoch):
                # 初始化每次迭代的训练损失与精度平均指标类
                epoch_loss_avg = AverageMeter()
                epoch_accuracy = AverageMeter()

                # 初始化精度条
                progbar = K.utils.Progbar(train_iter_num)
                print('Epoch {}/{}'.format(ep+1, self.cfg.epoch))
                batch_domain_labels = np.vstack([np.tile([1., 0.], [self.cfg.batch_size // 2, 1]),
                                           np.tile([0., 1.], [self.cfg.batch_size // 2, 1])])
                for i in np.arange(1,train_iter_num+1):
                    # 获取小批量数据集及其图像标签与域标签
                    batch_images, batch_labels = train_datagen.__next__()#train_source_datagen.next_batch()

                    # 前向传播,计算损失及其梯度
                    op,train_loss,train_acc,learning_rate,global_step = \
                                    sess.run([self.train_op,self.loss,self.accuracy,self.learning_rate,self.global_step],
                                                feed_dict={self.image_input:batch_images,
                                                            self.real_label:batch_labels})
                    self.writer.add_summary(make_summary('learning_rate', learning_rate),global_step=global_step)
                    self.writer1.add_summary(make_summary('learning_rate', learning_rate), global_step=global_step)

                    # 更新训练损失与训练精度
                    epoch_loss_avg.update(train_loss,1)
                    epoch_accuracy.update(train_acc,1)

                    # 更新进度条
                    progbar.update(i, [('train_loss', train_loss),("train_acc",train_acc)])

                # 保存相关损失与精度值，可用于可视化
                train_loss_results.append(epoch_loss_avg.average)
                train_accuracy_results.append(epoch_accuracy.average)

                self.writer.add_summary(make_summary('train/train_loss', epoch_loss_avg.average),global_step=ep+1)
                self.writer.add_summary(make_summary('accuracy/train_accuracy', epoch_accuracy.average),global_step=ep+1)

                self.writer1.add_summary(make_summary('train/train_loss', epoch_loss_avg.average),global_step=ep+1)
                self.writer1.add_summary(make_summary('accuracy/train_accuracy', epoch_accuracy.average),global_step=ep+1)

                if (ep+1) % interval == 0:
                    # 评估模型在验证集上的性能
                    val_ep.append(ep)
                    val_loss,val_accuracy = self.eval_on_val_dataset(sess,val_datagen,val_iter_num,ep+1)
                    val_loss_results.append(val_loss)
                    val_accuracy_results.append(val_accuracy)
                    str =  "Epoch{:03d}_val_loss_{:.3f},val_accuracy_{:.3%}".format(ep+1,val_loss,val_accuracy)
                    print(str)

                    if val_accuracy >= val_acc_max:              # 验证精度达到当前最大，保存模型
                        val_acc_max = val_accuracy
                        self.saver_save.save(sess,os.path.join(checkpoint_dir,str+".ckpt"))
                        print(1)
                        self.model.save(os.path.join(checkpoint_dir,str+".h5"))
                        print(2)

            # 保存训练与验证结果
            path = os.path.join(result_dir, "loss.jpg")
            plot_loss(np.arange(1, len(train_loss_results) + 1),
                        [np.array(train_loss_results), np.array(val_loss_results)],path)

            train_acc_results = np.array(train_accuracy_results)[np.array(val_ep)]
            path = os.path.join(result_dir, "accuracy.jpg")
            plot_accuracy(np.array(val_ep) + 1, [train_acc_results, val_accuracy_results], path)

            # 保存最终的模型
            model_path = os.path.join(checkpoint_dir,"trained_model.ckpt")
            self.saver_save.save(sess,model_path)
            print("Train model finshed. The model is saved in : ", model_path)
            print('\n----------- end to train -----------\n')

    def eval_on_val_dataset(self,sess,val_datagen,val_batch_num,ep):
        """
        这是评估模型在验证集上的性能的函数
        :param sess: tf的会话变量
        :param val_datagen: 验证集数据集生成器
        :param val_batch_num: 验证集数据集批量个数
        :param ep: 当前周期数
        """
        epoch_loss_avg = AverageMeter()
        epoch_accuracy = AverageMeter()
        for i in np.arange(1, val_batch_num + 1):
            # 获取小批量数据集及其图像标签与域标签
            batch_images, batch_labels = val_datagen.__next__()

            # 在验证阶段只利用目标域数据及其标签进行测试,计算模型在验证集上相关指标的值
            val_loss, val_acc = sess.run([self.loss, self.accuracy],
                                            feed_dict={self.image_input: batch_images,
                                                    self.real_label: batch_labels})
            # 更新损失与精度的平均值
            epoch_loss_avg.update(val_loss, 1)
            epoch_accuracy.update(val_acc, 1)

        self.writer.add_summary(make_summary('val/val_loss', epoch_loss_avg.average),global_step=ep)
        self.writer.add_summary(make_summary('accuracy/val_accuracy', epoch_accuracy.average),global_step=ep)

        self.writer1.add_summary(make_summary('val/val_loss', epoch_loss_avg.average),global_step=ep)
        self.writer1.add_summary(make_summary('accuracy/val_accuracy', epoch_accuracy.average),global_step=ep)
        return epoch_loss_avg.average,epoch_accuracy.average

    def test_single_image(self,image_path,weight_path):
        """
        这是对一张图像进行分割的函数
        :param image_path: 图像路径
        :param weight_path: 模型路径
        :return:
        """
        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 加载预训练模型
            if weight_path is not None:
                if ".ckpt" in weight_path:          # pre_model_path的地址写到.ckpt
                    saver_restore = tf.train.import_meta_graph(weight_path + ".meta")
                    saver_restore.restore(sess, weight_path)
                    print(1)
                elif ".h5" in weight_path:
                    self.model.load_weights(weight_path, by_name=True)
                    print(2)
                print("restore model from : %s" % (weight_path))

            # 导入图像,并扩充一维以满足tf中输入张量形状要求
            image = cv2.imread(image_path)
            image = cv2.resize(image,(224,224))
            batch_image = np.expand_dims(image,axis=0)
            print(np.shape(batch_image))

            # 运行模型，获取分类结果
            pred_label = sess.run([self.pred_label],feed_dict={self.image_input:batch_image})
            pred_label = np.argmax(pred_label)
            print("%s is predited as %s" %(image_path,self.cfg.label_dict[pred_label]))

    def test_batch_images(self,image_dir,weight_path):
        """
        这是对一张图像进行分割的函数
        :param image_dir: 图像保存目录
        :param weight_path: 模型路径
        :return:
        """
        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 加载预训练模型
            if weight_path is not None:
                if ".ckpt" in weight_path:          # pre_model_path的地址写到.ckpt
                    saver_restore = tf.train.import_meta_graph(weight_path + ".meta")
                    saver_restore.restore(sess, weight_path)
                    print(1)
                elif ".h5" in weight_path:
                    self.model.load_weights(weight_path, by_name=True)
                    print(2)
                print("restore model from : %s" % (weight_path))

            # 导入图像,并扩充一维以满足tf中输入张量形状要求
            batch_images = []
            batch_image_paths = []
            for image_name in os.listdir(image_dir):
                image_path = os.path.join(image_dir,image_name)
                image = cv2.resize(image, (224, 224))
                batch_image_paths.append(image_path)
                batch_images.append(cv2.imread(image_path))
            batch_images = np.array(batch_images)
            print(np.shape(batch_images))

            # 运行模型，获取分类结果
            batch_pred_labels = sess.run([self.pred_label],feed_dict={self.image_input:batch_images})
            batch_pred_labels = np.argmax(batch_pred_labels,axis=-1).flatten()
            for image_path,pred_label in zip(batch_image_paths,batch_pred_labels):
                print("%s is predited as %s" % (image_path, self.cfg.label_dict[pred_label]))

    def eval_on_dataset(self,datagen,batch_num,weight_path):
        """
        这是评估模型在验证集上的性能的函数
        :param sess: tf的会话变量
        :param datagen: 数据集生成器
        :param batch_num: 数据集批量个数
        :param weight_path: 模型路径
        """
        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())

            # 加载预训练模型
            if weight_path is not None:
                if ".ckpt" in weight_path:          # pre_model_path的地址写到.ckpt
                    saver_restore = tf.train.import_meta_graph(weight_path + ".meta")
                    saver_restore.restore(sess, weight_path)
                    print(1)
                elif ".h5" in weight_path:
                    self.model.load_weights(weight_path, by_name=True)
                    print(2)
                print("restore model from : %s" % (weight_path))

            loss = AverageMeter()
            accuracy = AverageMeter()
            for i in np.arange(batch_num):
                # 获取小批量数据集及其图像标签与域标签
                batch_images, batch_labels = datagen.__next__()

                # 在验证阶段只利用目标域数据及其标签进行测试,计算模型在验证集上相关指标的值
                val_loss, val_acc = sess.run([self.loss, self.accuracy],
                                                feed_dict={self.image_input: batch_images,
                                                        self.real_label: batch_labels})
                # 更新损失与精度的平均值
                loss.update(val_loss, 1)
                accuracy.update(val_acc, 1)

            print("eval on this dataset, loss is {:.3f} , accuracy is {:.3f%}".format(loss.average,accuracy.average))