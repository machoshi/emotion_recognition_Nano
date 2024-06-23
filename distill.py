import os
import argparse
import socket
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from Xception import MiniXception
from resnet import ResNet101, ResNet18
from process import load_fer2013
from sklearn.model_selection import train_test_split

import logging
import random

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

# parameters
batch_size = 64
num_epochs = 30
input_shape = (1, 48, 48)
validation_split = .2
num_classes = 7
patience = 50

logging.basicConfig(filename='logs/distill.log', level=logging.INFO)

# Data Augmentation
data_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(input_shape[1:3], scale=(0.8, 1.2), ratio=(0.75, 1.333)),
])

class FED(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        data = self.X[i]
        
        if self.transforms:
            data = self.transforms(data)
            
        return (data, self.y[i])


def main():
    # model parameters/compilation
    model_t = ResNet101(num_classes=num_classes).cuda()
    model_t.load_state_dict(torch.load('./resnet101.pth'))
    # model_s = MiniXception(input_shape, num_classes).cuda()
    model_s = ResNet18(num_classes=num_classes).cuda()
    # loading dataset
    faces, emotions = load_fer2013()
    # faces = preprocess_input(faces)
    xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, random_state=42)
    train_data = FED(xtrain, ytrain, transforms=data_transforms)
    test_data = FED(xtest, ytest)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    model_t.eval()
    model_s.eval()

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(4)

    # optimizer
    optimizer = torch.optim.Adam(model_s.parameters(), lr=0.01)

    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience / 4), verbose=True)

    # routine
    for epoch in range(1, num_epochs + 1):
        model_t.eval()
        model_s.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), torch.argmax(target.int(), dim=1).cuda()
            # print(target)
            optimizer.zero_grad()
            logit_s = model_s(data)
            with torch.no_grad():
                logit_t = model_t(data)

            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)
            loss = 0.1 * loss_cls + 0.9 * loss_div
            loss.backward()
            optimizer.step()

        # validation
        model_s.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.cuda(), torch.argmax(target.int(), dim=1).cuda()
                output = model_s(data)
                test_loss = criterion_cls(output, target).item()
                total_loss += test_loss
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        scheduler.step(total_loss)
        logging.info('Epoch: {}, Train Loss: {:.6f}, Test Loss: {:.6f}, Accuracy: {:.6f}'.format(
            epoch, loss.item(), test_loss, 100. * correct / len(test_loader.dataset)))

    # save model
    torch.save(model_s.state_dict(), './resnet18_distilled.pth')


if __name__ == '__main__':
    main()
