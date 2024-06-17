import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from Xception import MiniXception, BigXception,DeepXception
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


model_name = "DeepXception"
logging.basicConfig(filename='logs/training.log', level=logging.INFO)
# Data Augmentation
data_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(input_shape[1:3], scale=(0.8, 1.2), ratio=(0.75, 1.333)),
])


# model parameters/compilation
device = 'cuda'
model = DeepXception(input_shape, num_classes).cuda()
print(model)
util.replace_layers(model)
print(model)
# model = torch.load("saves/initial_miniModel.ptmodel")

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


def train():
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # print(data.shape)
            data, target = data.to(device), torch.argmax(target.int(), dim=1).to(device)
            # print(target)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # validation
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
        
        # logging
        logging.info('Epoch: {}, Train Loss: {:.6f}, Test Loss: {:.6f}, Accuracy: {:.6f}'.format(
            epoch, loss.item(), total_loss, 100. * correct / len(test_loader.dataset)))  
        
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


#Initial training
print("---Initial training---")
train()
accuracy = test()
torch.save(model,f"saves/initial_{model_name}.ptmodel")
print("---Before pruning---")
util.print_nonzeros(model)

#Pruning
util.prune_by_std(model,1)
accuracy = test()
print("---After pruning---")
util.print_nonzeros(model)

#Retrain
print("---Retraining---")
util.froze_layers(model)
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train()
torch.save(model,f"saves/{model_name}_after_retraining.ptmodel")
accuracy = test()

print("---After retraining---")
util.print_nonzeros(model)

util.replace_layers_sparse(model)
torch.save(model,f"saves/deploy_{model_name}.ptmodel")
accuracy = test()


