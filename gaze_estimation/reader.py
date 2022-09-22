# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 17:35:03 2022

@author: 19zm5
"""

import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
from utils import *
from models.aeri_unet import AERI_UNet 

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])
  
class train_loader(Dataset): 
  def __init__(self, path, root, device, header=True):
    self.lines = []
    self.device = device
    self.isolator = AERI_UNet().to(self.device)
    self.checkpoint = torch.load("/weights/aeri_weights/*.t7") # load the weights of the AERI network
    self.isolator.load_state_dict(self.checkpoint['state_dict'])
    
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[1]
    gaze2d = line[5]
    eye = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)
    
    img = cv2.imread(os.path.join(self.root, eye).replace('\\','/'),0)
    img = blur_img(img)
    img = change_contrast(img))
    img = blur_region(img)
    img = remove_region(img)
    img = add_line(img)
    img = noise(img)
    img = np.clip(img,0,255)
    img = img.astype(np.uint8)
    img = img/255.0
    img = img[:,:,np.newaxis]
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    
    out = self.isolator(torch.unsqueeze(img,0).to(self.device))
    
    iris_mask=out.detach().cpu().permute(0,2,3,1)[:,:,:,1] 
    iris_mask[iris_mask>0.5]=1
    iris_mask[iris_mask<=0.5]=0
    
    eyemask=out.detach().cpu().permute(0,2,3,1)[:,:,:,0] 
    eyemask[eyemask>0.5]=1
    eyemask[eyemask<=0.5]=0
    

    info = {"eye":img,
            "iris": iris_mask,
            "eye_mask":eyemask}

    return info, label

class test_loader(Dataset): 
  def __init__(self, path, root, device, header=True):
    
    self.lines = []
    self.device = device
    self.isolator = AERI_UNet().to(self.device)
    self.checkpoint = torch.load("/weights/aeri_weights/*.t7") # load the weights of the AERI network
    self.isolator.load_state_dict(self.checkpoint['state_dict'])
    
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[1]
    gaze2d = line[5]
    eye = line[0]
    
    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    img = cv2.imread(os.path.join(self.root, eye).replace('\\','/'),0)/255.0
    img = img[:,:,np.newaxis]
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    
    out = self.isolator(torch.unsqueeze(img,0).to(self.device))
    
    iris_mask=out.detach().cpu().permute(0,2,3,1)[:,:,:,1] 
    iris_mask[iris_mask>0.5]=1
    iris_mask[iris_mask<=0.5]=0
    
    eyemask=out.detach().cpu().permute(0,2,3,1)[:,:,:,0] 
    eyemask[eyemask>0.5]=1
    eyemask[eyemask<=0.5]=0
    
    info = {"eye":img,
            "iris": iris_mask,
            "eye_mask":eyemask}

    return info, label

def txtload(labelpath, imagepath, batch_size, device, mode, shuffle=True, num_workers=0, header=True):
  if mode == 'train':
    dataset = train_loader(labelpath, imagepath, device, header)
    print(f"[Read Data]: Total num: {len(dataset)}")
    print(f"[Read Data]: Label path: {labelpath}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  else:
    dataset = test_loader(labelpath, imagepath, device, header)
    print(f"[Read Data]: Total num: {len(dataset)}")
    print(f"[Read Data]: Label path: {labelpath}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  
  return load
