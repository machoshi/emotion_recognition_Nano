import os
import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.module import Module
from net.SparseLayer import SparseLinear, SparseConv2d
from net.prune import PruningModule, MaskedLinear, MaskedConv2d
import torch.nn.functional as F
from torchvision import datasets, transforms


def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)

def prune_by_std(model,s = 0.25):
    for name, module in model.named_modules():
            if isinstance(module,MaskedConv2d) or isinstance(module,MaskedLinear):
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)

def froze_layers(model):
    for name,param in list(model.named_parameters()): #冻结除了卷积层和全连接层之外的层
        if 'fc' not in name and 'conv' not in name:
            param.requires_grad=False


def replace_layers(model): #将model替换成剪枝层
    for name,module in model.named_children():
        if isinstance(module,nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            bias = module.bias 
            new_conv_layer = MaskedConv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=bias)
            new_conv_layer.load_weight(module.weight,bias)
            setattr(model,name,new_conv_layer)
        elif isinstance(module,nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias
            new_linear_layer = MaskedLinear(in_features,out_features,bias)
            new_linear_layer.load_weight(module.weight,bias)
            setattr(model,name,new_linear_layer)
        elif len(list(module.children()))>0:
            replace_layers(module)

def replace_layers_sparse(model): #将model的剪枝层替换成sparse
    for name,module in model.named_children():
        if isinstance(module,MaskedConv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            # print(type(padding))
            bias = module.bias 
            new_conv_layer = SparseConv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=bias)
            new_conv_layer.load_weight(module.weight,bias)
            setattr(model,name,new_conv_layer)
        elif isinstance(module,MaskedLinear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias
            new_linear_layer = SparseLinear(in_features,out_features,bias)
            new_linear_layer.load_weight(module.weight,bias)
            setattr(model,name,new_linear_layer)
        elif len(list(module.children()))>0:
            replace_layers(module)

def print_model(model):
    for name, module in model.named_modules():
        if isinstance(module,nn.Module):
            print(name)
            print(module)
            print('---')

def print_model_module(model):
    print("输出模型组件")
    for name, module in model.named_modules():
        print(name,module)
        print(type(module))


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    model.to(device)
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy
