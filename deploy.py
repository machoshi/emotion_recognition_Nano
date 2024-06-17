import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from Xception import MiniXception
from process import load_fer2013
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import util
import logging

# parameters
batch_size = 32
num_epochs = 10
input_shape = (1, 48, 48)
validation_split = .2
num_classes = 7
patience = 50

logging.basicConfig(filename='logs/training.log', level=logging.INFO)
# Data Augmentation
data_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(input_shape[1:3], scale=(0.8, 1.2), ratio=(0.75, 1.333)),
])


# model parameters/compilation
device = 'cuda'
# model = MiniXception(input_shape, num_classes).cuda()
# print(model)
# util.replace_layers(model)
# print(model)
model = torch.load("saves/deploy_miniModel.ptmodel")

# util.print_model_module(model)
# util.replace_layers_sparse(model)
# util.print_model_parameters(model)
# util.replace_layers(model)
# util.print_model_parameters(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience / 4), verbose=True)
initial_optimizer_state_dict = optimizer.state_dict()


# loading dataset
faces, emotions = load_fer2013()
# faces = preprocess_input(faces)
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)
train_data = TensorDataset(xtrain, ytrain)
test_data = TensorDataset(xtest, ytest)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

def test():
    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), torch.argmax(target.int(), dim=1).to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
        
    # scheduler
    scheduler.step(total_loss)
    acc = 100.*correct/len(test_loader.dataset)
    print(f'Test set: Average loss:{total_loss},Accuracy:{acc}')
    return acc

test()



# model_path = 'saves'
# model_name = 'model_after_retraining.ptmodel'
# model_path = os.path.join(model_path,model_name)

# # util.test(model, torch.cuda.is_available())
# print("Start loading weight")
# # sparse_model = load_LeNet_5(model_path)
# sparse_model = torch.load(model_path)
# util.test(sparse_model, torch.cuda.is_available())
# util.replace_layers_sparse(sparse_model) #将导入的剪枝模型替换成稀疏模型

# # util.print_model_parameters(sparse_model)
# util.print_model_module(sparse_model)
# torch.save(sparse_model,'saves/deploy_model.ptmodel')
# util.test(sparse_model, torch.cuda.is_available())
# util.print_model_parameters(model)


# import torch
# import torch.nn as nn
# # import torch.nn.functional as F
# import math
# # model = torch.load("saves/miniModel_after_retraining.ptmodel")
# # util.replace_layers_sparse(model)
# # torch.save(model,f"saves/deploy_miniModel.ptmodel")
# model = torch.load("saves/deploy_miniModel.ptmodel")

# input = torch.randn(5, 5)
# kernel = torch.randn(3, 3)
# kernel1 = torch.randn(2,2,3,3)
# input1 = torch.randn(3,2,5,5)

# bias = torch.randn(2)

# # 用矩阵运算实现二维卷积 input flatten版
# def matrix_multiplication_for_conv2d_flatten(input, kernel,bias = 0, stride=1, padding=0,groups = 2):
    
#     batch,_,input_h, input_w = input.shape
#     channel_out,channel_in,kernel_h, kernel_w = kernel.shape

#     print(input.shape)

#     input = F.unfold(input,kernel_size= (kernel_h,kernel_w),padding=padding,stride=stride,groups = groups)

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

