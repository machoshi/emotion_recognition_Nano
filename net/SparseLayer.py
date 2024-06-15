import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from scipy.sparse import csr_matrix

# class SparseLinear(Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(SparseLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         sparse_weight =  torch.sparse_coo_tensor(size=(out_features, in_features))
#         self.sparse_weight = Parameter(sparse_weight)
        
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.sparse_weight.size(1))
#         # self.sparse_weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input):
#         output = torch.sparse.mm(self.sparse_weight,input.T) + self.bias.unsqueeze(1)
#         return output.T
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'in_features=' + str(self.in_features) \
#             + ', out_features=' + str(self.out_features) \
#             + ', bias=' + str(self.bias is not None) + ')'

# class SparseLinear(Module):
#     def __init__(self,module):
#         super(SparseLinear, self).__init__()
#         self.in_features = module.in_features
#         self.out_features = module.out_features
#         print(module.weight.data.shape)

#         # indices = module.weight.data.nonzero().T
#         # values = module.weight.data[indices[0],indices[1]]
#         # # sparse_weight =  torch.sparse_coo_tensor(size=(self.out_features, self.in_features))
#         # # sparse_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
#         # sparse_weight = torch.sparse_coo_tensor(indices,values,(module.out_features,module.in_features))
#         sparse_weight = module.weight.data.to_sparse()
#         print(sparse_weight)
#         self.sparse_weight = Parameter(sparse_weight)
        
#         if module.bias != None:
#             self.bias = Parameter(module.bias.data)
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, input):
#         # print(input.shape)
#         output = torch.sparse.mm(self.sparse_weight,input.T)
#         output = output + self.bias.unsqueeze(1)
#         return output.T
#         # return F.linear(input, self.sparse_weight, self.bias)
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'in_features=' + str(self.in_features) \
#             + ', out_features=' + str(self.out_features) \
#             + ', bias=' + str(self.bias is not None) + ')'
    

# class SparseConv2d(Module):
#     def __init__(self,module):
#         super(SparseConv2d, self).__init__()
#         self.padding = module.padding
#         self.stride = module.stride
#         print(module.weight.data.shape)

#         channel_out, channel_in, kernel_h, kernel_w = module.weight.data.shape
#         self.channel_out = channel_out
#         self.channel_in = channel_in
#         self.kernel_h = kernel_h
#         self.kernel_w = kernel_w



#         kernel = module.weight.data
#         kernel = kernel.reshape(channel_out,channel_in*kernel_h* kernel_w)#张平kernel,方便用稀疏矩阵存储

#         sparse_weight = kernel.to_sparse()

#         self.sparse_weight = Parameter(sparse_weight)
        
#         if module.bias != None:
#             self.bias = Parameter(module.bias.data)
#         else:
#             self.register_parameter('bias', None)

#     def forward(self, input):
#         # print(input.shape)
#         weight = self.sparse_weight
#         bias = self.bias

#         batch,_,input_h,input_w = input.shape
#         input = F.unfold(input,kernel_size=(self.kernel_h,self.kernel_w),padding=self.padding,stride=self.stride)
#         input = input.permute(1,0,2)#对输入张量进行卷积展开并重排序，让batchsize在列数据上
#         input = input.reshape(input.shape[0],input.shape[1]*input.shape[2])#行为channel_in*size, 列为batch*output

        
#         output = torch.sparse.mm(self.sparse_weight.data,input) #进行二维矩阵乘法代替卷积

#         output_h = (math.floor((input_h - self.kernel_h+2*self.padding) / self.stride) + 1)
#         output_w = (math.floor((input_w - self.kernel_w+2*self.padding) / self.stride) + 1)

#         if self.bias != None:
#             output = output.t()+bias
#             output = output.t()
        
#         output = output.reshape((self.channel_out,batch,output_h,output_w))
#         output = output.permute(1,0,2,3) #将batchsize重新换回到第一个维度


#         return output
#         # return F.linear(input, self.sparse_weight, self.bias)
    


    
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'channel_in=' + str(self.channel_in) \
#             + ', channel_out=' + str(self.channel_out) \
#             + ', kernel_h=' + str(self.kernel_h) \
#             + ', kernel_w=' + str(self.kernel_w) \
#             + ', bias=' + str(self.bias is not None) + ')'

class SparseLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.Tensor(out_features, in_features)
        sparse_weight = weight.to_sparse()
        self.sparse_weight = Parameter(sparse_weight)
        
        if bias is not None:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
    def load_weight(self,weight,bias):
        self.sparse_weight.data = weight.data.to_sparse()
        if bias is not None:
            self.bias.data = bias.data

    
    def forward(self, input):
        # print(input.shape)
        output = torch.sparse.mm(self.sparse_weight,input.T)

        if self.bias is not None:
            output = output + self.bias.unsqueeze(1)
        return output.T
        # return F.linear(input, self.sparse_weight, self.bias)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
    

class SparseConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride = 1,padding = 0,bias=True):
        super(SparseConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size= kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        print(self.padding)
        kernel_h,kernel_w = self.kernel_size

        kernel = torch.Tensor(out_channels, in_channels, *self.kernel_size)
        kernel = kernel.reshape(out_channels,in_channels*kernel_h* kernel_w)#张平kernel,方便用稀疏矩阵存储

        sparse_weight = kernel.to_sparse()

        self.sparse_weight = Parameter(sparse_weight)
        
        if bias is not None:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def load_weight(self,weight,bias):
        kernel_h,kernel_w = self.kernel_size

        kernel = weight.data
        kernel = kernel.reshape(self.out_channels,self.in_channels*kernel_h* kernel_w)
        self.sparse_weight.data = kernel.to_sparse()
        if bias is not None:
            self.bias.data = bias.data

    def forward(self, input):
        # print(input.shape)
        kernel_h,kernel_w = self.kernel_size
        if isinstance(self.padding,tuple):
            padding_h,padding_w = self.padding
        else:
            padding_h = self.padding
            padding_w = self.padding

        if isinstance(self.stride,tuple):
            stride_h,stride_w = self.stride
        else:
            stride_h = self.stride
            stride_w = self.stride
    
        batch,_,input_h,input_w = input.shape
        input = F.unfold(input,kernel_size=(kernel_h,kernel_w),padding=self.padding,stride=self.stride)
        input = input.permute(1,0,2)#对输入张量进行卷积展开并重排序，让batchsize在列数据上
        input = input.reshape(input.shape[0],input.shape[1]*input.shape[2])#行为channel_in*size, 列为batch*output

        
        output = torch.sparse.mm(self.sparse_weight.data,input) #进行二维矩阵乘法代替卷积

        output_h = (math.floor((input_h - kernel_h+2*padding_h) / stride_h) + 1)
        output_w = (math.floor((input_w - kernel_w+2*padding_w) / stride_w) + 1)

        if self.bias is not None:
            output = output.t()+self.bias
            output = output.t()
        
        output = output.reshape((self.out_channels,batch,output_h,output_w))
        output = output.permute(1,0,2,3) #将batchsize重新换回到第一个维度


        return output
    


    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', bias=' + str(self.bias is not None) + ')'