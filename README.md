# VGG16-CIFAR10
这个是利用VGG16网络实现CIFAR10数据集分类的项目。VGG16网络的参考论文为：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

---
# 环境依赖

 **1. tensorflow 1.14.0
 2. numpy
 3. matplotlib
 4. openccv 3.4.5.20**
 
 ---

# CIFAR10数据集预处理
在对VGG16网络进行训练和测试之前之前首先需要下载CIFAR10数据集，CIFAR10数据集下载网址为：[CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html)。将下载好的CIFAR10数据集解压到./data目录下，然后运行如下代码来改变CIFAR10存储格式，为后续VGG16的训练过程做准备。转化数据存储格式的CIFAR10数据集放置在./data/cifar10目录下，若想改变新CIFAR10数据集的存储目录，请在`cifar_preprocess.py`中相关目录地址进行修改
```bash
python cifar_preprocess.py
```

---
# VGG16模型训练
VGG16模型的训练脚本为`train_vgg16.py`，训练VGG16模型的命令为如下，在`train_vgg16.py`对相关参数进行修改。VGG16训练CIFAR10数据集需要使用Keras官方的VGG16的预训练模型，其地址为：[预训练模型](https://github.com/fchollet/deep-learning-models/releases/tag/v0.1)。**将vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5下载好并放在./pre_weight目录下**。
```bash
python train_vgg16.py
```

---
# VGG16测试单张图像
VGG16模型测试单张图像的脚本为`test_single_image.py`，VGG16测试单张图像的命令为如下，若要测试自定义图像，请在`test_single_image.py`中修改图像路径和VGG16模型地址。
```bash
python test_single_image.py
```

---
# VGG16测试批量图像集
VGG16测试批量图像集的脚本为`test_single_image.py`，VGG16测试批量图像集的命令为如下，若要测试自定义小批量图像集，请在`test_single_image.py`中修改小批量图像集的目录地址和VGG16模型地址。
```bash
python test_batch_image.py
```

---
# VGG16评估数据集
 VGG16评估数据集的脚本为`eval_on_dataset.py`，VGG16评估数据集的命令为如下，若要评估自定义数据集，请在`eval_on_dataset.py`中修改数据集目录地址和VGG16模型地址。
```bash
python eval_on_dataset.py
```
