import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import util
from net.SparseLayer import SparseLinear, SparseConv2d

def load_LeNet_5(model_path):
    model = torch.load(model_path)
    conv_modules = []
    fc_modules = []
    # util.print_model_module(model)
    # util.print_model_parameters(model)
    for name, module in model.named_modules():
        if 'LeNet_5' in name:
            continue #忽略本身那一层
        if 'fc' in name:
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias
            new_linear = SparseLinear(in_features,out_features,bias)
            new_linear.load_weight(module.weight,module.bias)
            fc_modules.append(new_linear)
        if 'conv' in name:
            in_channels = module.in_features
            out_channels = module.out_features
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            bias = module.bias 
            new_conv = SparseConv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=bias)
            new_conv.load_weight(module.weight,module.bias)
            conv_modules.append(new_conv)
    return Sparse_LeNet_5(conv_modules,fc_modules)
    
class Sparse_LeNet_5(nn.Module):
    def __init__(self,conv_modules,fc_modules):
        super(Sparse_LeNet_5,self).__init__()
        self.conv1 = conv_modules[0]
        self.conv2 = conv_modules[1]
        self.fc1 = fc_modules[0]
        self.fc2 = fc_modules[1]
        self.fc3 = fc_modules[2]

    def forward(self,x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3

        # Fully-connected
        x = x.view(-1, 16*5*5)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x
 



model_path = 'saves'
model_name = 'model_after_retraining.ptmodel'
model_path = os.path.join(model_path,model_name)

# util.test(model, torch.cuda.is_available())
print("Start loading weight")
# sparse_model = load_LeNet_5(model_path)
sparse_model = torch.load(model_path)
util.test(sparse_model, torch.cuda.is_available())
util.replace_layers_sparse(sparse_model) #将导入的剪枝模型替换成稀疏模型

# util.print_model_parameters(sparse_model)
util.print_model_module(sparse_model)
torch.save(sparse_model,'saves/deploy_model.ptmodel')
util.test(sparse_model, torch.cuda.is_available())
# util.print_model_parameters(model)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# input = torch.randn(5, 5)
# kernel = torch.randn(3, 3)
# kernel1 = torch.randn(2,2,3,3)
# input1 = torch.randn(3,2,5,5)

# bias = torch.randn(2)

# # 用矩阵运算实现二维卷积 input flatten版
# def matrix_multiplication_for_conv2d_flatten(input, kernel,bias = 0, stride=1, padding=0):
    
#     batch,_,input_h, input_w = input.shape
#     channel_out,channel_in,kernel_h, kernel_w = kernel.shape

#     print(input.shape)

#     input = F.unfold(input,kernel_size= (kernel_h,kernel_w),padding=padding,stride=stride)

#     input = input.permute(1,0,2) #输入张量变换以进行张平，使得batchsize在列数据上
#     print(input.shape)

#     input = input.reshape(input.shape[0],input.shape[1]*input.shape[2]) #行为channel_in*size, 列为batch*output

#     print(input.shape)

#     # input = input.reshape(batch,input.shape[1],input.shape[2]) #输入张量张平
#     kernel = kernel.reshape(channel_out,channel_in*kernel_h*kernel_w) #kernel张平

#     kernel_coo = kernel.to_sparse()
#     # output = torch.mm(kernel,input)
#     # output = torch.mm(kernel,input)
#     output = torch.sparse.mm(kernel_coo,input)

#     output_h = (math.floor((input_h - kernel_h+2*padding) / stride) + 1)
#     output_w = (math.floor((input_w - kernel_w+2*padding) / stride) + 1)
    
#     print(output.shape,bias.shape)

#     output = output.t()+bias
#     output = output.t()

#     output = output.reshape((channel_out,batch,output_h,output_w))
#     output = output.permute(1,0,2,3)
    
#     return output

# # 调用自己实现的函数
# res = matrix_multiplication_for_conv2d_flatten(input1, kernel1,bias = bias, padding=1, stride=2)
# print(res)

# # 验证一下
# # 调用API结果
# res2 = F.conv2d(
#     input1,
#     kernel1,
#     padding=1,
#     bias=bias,
#     stride=2
# )
# print(res2)
# # res2 = res2.squeeze(0).squeeze(0)
# print(torch.allclose(res, res2))

