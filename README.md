# Emotion Recognition on Jetson Nano


### 预训练与模型蒸馏

train.py包括了模型的选取，数据集的增广以及训练过程

运行``python train.py``即可训练，但需要把其中``model``类型修改刀指定模型

process.py包括了数据集的提取与预处理，我们使用的fer2013数据集可以在以下链接下载：https://www.kaggle.com/c/3364/download-all

distill.py是模型蒸馏部分代码，需要把``model_t``和``model_s``改为对应模型


### 剪枝部分

./net目录包含了剪枝和稀疏化处理时的自定义卷积层和线性层。

./saves目录用于保存中间模型

Xception.py, resnet.py包含了本次剪枝的目标模型

util.py包含了与剪枝相关的工具函数

#### 运行方法

修改MiniPruning.py中加载的model为指定剪枝目标模型，运行MiniPruning.py会进行迭代剪枝和稀疏化处理操作，并生成对应的模型。

### 系统部署

video.py实现了人脸情绪识别的全流程，修改对应的``emotion_model_path``为指定的情绪识别模型，并运行``python video.py``，其中的人脸位置识别文件可在https://github.com/opencv/opencv/tree/master/data/haarcascades 下载


