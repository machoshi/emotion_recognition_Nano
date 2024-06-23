# Emotion Recognition on Jetson Nano

### 剪枝部分

./net目录包含了剪枝和稀疏化处理时的自定义卷积层和线性层。

./saves目录用于保存中间模型

Xception.py, resnet.py包含了本次剪枝的目标模型

util.py包含了与剪枝相关的工具函数

#### 运行方法

修改MiniPruning.py中加载的model为指定剪枝目标模型，运行MiniPruning.py会进行迭代剪枝和稀疏化处理操作，并生成对应的模型。

### 系统部署

video.py实现了人脸情绪识别的全流程，修改对应的emotion_model_path为指定的情绪识别模型，并运行video.py

### 预训练与模型蒸馏




