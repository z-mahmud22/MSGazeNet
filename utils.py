# -*- coding: utf-8 -*-
"""
Created on Tue Mar 1 17:35:03 2022

@author: zunayed mahmud
"""

import torch
import numpy as np
import random
import cv2

def create_trainset(folder, input_list):
      for j in input_list:
          folder.remove(j)
      return folder
  
def create_testset(input_list):
    new_folder = []
    for j in input_list:
        new_folder.append(j)
    return new_folder

# =============================================================================
# GAZE FUNCTIONS
# =============================================================================

def gazeto3d(gaze):
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi
  
def mean_angular_error(a, b, batch_size):
  error=0
  for k, g_pred in enumerate(a):
      error+=angular(gazeto3d(g_pred), gazeto3d(b[k]))
  return error/batch_size


# =============================================================================
# AUGMENTATIONS
# =============================================================================

def add_line(img):
    num_of_line=random.randint(0,4)
    
    for line_no in range(num_of_line):
        x1=random.randint(0,img.shape[0])
        x2=random.randint(0,img.shape[0])
        y1=random.randint(0,img.shape[1])
        y2=random.randint(0,img.shape[1])
        
        color=random.randint(0,255)
        thickness=random.randint(1,3)
        # thickness=random.random()
        img=cv2.line(img, (x1,y1), (x2,y2), color, thickness)
        
    return img


def change_contrast(img):
    op_no=random.randint(0,3)
    
    if op_no==0:
        return img
    if op_no==1:
        black_pixel=random.randint(0,100)
        img[img<black_pixel]=0
        return img
    if op_no==2:
        white_pixel=random.randint(155,255)
        img[img>white_pixel]=255
        return img
    if op_no==3:
        black_pixel=random.randint(0,100)
        black_pixel_new=random.randint(0,black_pixel)
        white_pixel=random.randint(155,255)
        white_pixel_new=random.randint(white_pixel,255)
        
        xp = [0, black_pixel, 128, white_pixel, 255]
        fp = [0, black_pixel_new, 128, white_pixel_new, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        img = cv2.LUT(img, table)
        return img
    
    
def resize_img(img):
    randimage=random.randint(0,1)
    if randimage==0:
        
        return img
    else:
        compress_range=random.uniform(1,2)
        img= cv2.resize(img, (int(img.shape[1]/compress_range), int(img.shape[0]/compress_range)), interpolation = cv2.INTER_AREA)
        img= cv2.resize(img, (60,36), interpolation = cv2.INTER_AREA)
        
        return img


def remove_region(img):
    num_of_line=random.randint(0,4)
    
    for line_no in range(num_of_line):
        w=random.randint(0,10)
        h=random.randint(0,10)
        
        x1=random.randint(0,img.shape[0]-w)
        y1=random.randint(0,img.shape[1]-h)
       
        
        color=random.randint(img.min(),img.max())
        
        img[x1:x1+w,y1:y1+h]= color
        
    return img


def blur_region(img):
    num_of_line=random.randint(0,4)
    
    for line_no in range(num_of_line):
        w=random.randint(0,10)
        h=random.randint(0,10)
        
        x1=random.randint(0,img.shape[0]-w)
        y1=random.randint(0,img.shape[1]-h)
       
        
        ksize=random.randrange(1,10,2)
        BlurImage=cv2.GaussianBlur(img,(ksize,ksize),0)
        img[x1:x1+w,y1:y1+h]= BlurImage[x1:x1+w,y1:y1+h]
        
    return img


def blur_img(img):
    ksize=random.randrange(1,5,2)
    sigma=random.uniform(0,2)
    BlurImage=cv2.GaussianBlur(img,(ksize,ksize),sigma)
    return BlurImage


def adjust_contrast(img1,contrast_factor):
     mean=np.mean(img1)
     bound=255
     ratio=1-random.uniform(0,contrast_factor )
     img=np.clip((ratio * img1 + (1.0 - ratio) * mean),0,bound)
     return img.astype(img1.dtype)
 

def noise(img):
    mean = 0   # some constant
    std = random.uniform(0,20.0)    # some constant (standard deviation)
    noisy_img = img + np.random.normal(mean, std, (36, 60))
    return noisy_img
