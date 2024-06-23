import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from Xception import MiniXception
from resnet import ResNet101
from process import load_fer2013
from sklearn.model_selection import train_test_split
import numpy as np

import logging

# parameters
batch_size = 64
num_epochs = 35
input_shape = (1, 48, 48)
validation_split = .2
num_classes = 7
patience = 50

logging.basicConfig(filename='logs/resnet101.log', level=logging.INFO)

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

# model parameters/compilation
device = 'cuda'
model = ResNet101(num_classes=num_classes).cuda()
# model = MiniXception(input_shape, num_classes).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience / 4), verbose=True)

# loading dataset
faces, emotions = load_fer2013()
# faces = preprocess_input(faces)
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, random_state=42)
train_data = FED(xtrain, ytrain, transforms=data_transforms)
test_data = FED(xtest, ytest)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
# train_data = TensorDataset(xtrain, ytrain)
# test_data = TensorDataset(xtest, ytest)

# training
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
            test_loss = criterion(output, target).item()
            total_loss += test_loss
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
        
    # scheduler
    scheduler.step(total_loss)
    
    # logging
    logging.info('Epoch: {}, Train Loss: {:.6f}, Test Loss: {:.6f}, Accuracy: {:.6f}'.format(
        epoch, loss.item(), test_loss, 100. * correct / len(test_loader.dataset)))

torch.save(model.state_dict(), './resnet101.pth')