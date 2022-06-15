# -*- coding: utf-8 -*-
"""
Created on Tue Mar 1 17:35:03 2022

@author: zunayed mahmud
"""

import os
from loader_aeri import trainset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.aeri_unet import AERI_UNet
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

train = trainset()
train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=True, num_workers=4)

writer = SummaryWriter('/path/to/summary')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Model building")
net=AERI_UNet().to(device)

print("optimizer building")
optimizer = optim.Adam(net.parameters(), lr=0.00001)
criterion = nn.MSELoss()

train_loss_values = []

print("Training")
for epoch in range(30):
    train_running_loss = 0.0
    net.train()
    
    for i, data in enumerate(tqdm(train_loader, 0)):
        # Acquire data
        eye, label = data 
        eye = eye.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        
        # Forward
        out = net(eye.float())
        
        # loss calculation
        loss=criterion(out, label.float())
        train_running_loss += (loss.item()*eye.size(0))
        
        # backward
        loss.backward()
        optimizer.step()
        
        
    
    train_epoch_loss = train_running_loss/len(train)
    train_loss_values.append(train_epoch_loss)
    
    
    print('[%d] loss: %.3f' %      #.3f means 3 decimal points
          (epoch + 1, train_epoch_loss))
    writer.add_scalar('Loss/MSE', train_epoch_loss, epoch)
    
    print('Saving Weights for MSE loss:', train_epoch_loss)
    state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
    savepath = 'weights/aeri_weights/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
             
    model_name=savepath+'AERI_E:'+str(epoch+1)+'_L:'+str(train_epoch_loss)+'.t7'
    torch.save(state,model_name)
