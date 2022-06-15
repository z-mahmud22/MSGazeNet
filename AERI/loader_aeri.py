# -*- coding: utf-8 -*-
"""
Created on Tue Mar 1 17:35:03 2022

@author: zunayed mahmud
"""

"""
A portion of this code is borrowed from: https://github.com/swook/GazeML 
Please also refer to this source if you directly use this code! 
"""

from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch
import json
import cv2
import random
from utils import *

data_transforms=transforms.Compose([
    transforms.ToTensor(),
])



class trainset(Dataset):
    def __init__(self):
        self.image_path="./images/" # path to the EyeMask dataset
        
    
    def __getitem__(self, idx):
        self.full_image = cv2.imread(self.image_path+str(idx+1)+'.jpg', 0)
        self.mask1 = cv2.imread(self.image_path+str(idx+1)+'_eyeshape.jpg', 0)
        self.mask2 = cv2.imread(self.image_path+str(idx+1)+'_iris.jpg', 0)
        
        ih, iw= self.full_image.shape
        iw_2, ih_2 = 0.5 * iw, 0.5 * ih
        oh = 36
        ow = 60
        
        with open(self.image_path + str(idx+1) + '.json') as f:
            data = json.load(f)
        
        def process_coords(coords_list):
            coords = [eval(l) for l in coords_list]
            return np.array([(x, ih-y, z) for (x, y, z) in coords])
        interior_landmarks = process_coords(data['interior_margin_2d'])
        caruncle_landmarks = process_coords(data['caruncle_2d'])
        iris_landmarks = process_coords(data['iris_2d'])
        
        left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
        right_corner = interior_landmarks[8, :2]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                              np.amax(interior_landmarks[:, :2], axis=0)], axis=0)
        
        random_multipliers = []
        difficulty = 1.0
        
        augmentation_ranges = {  # (easy, hard)
            'translation': (2.0, 10.0),
            'rotation': (0.1, 2.0),
            'intensity': (0.5, 20.0),
            'blur': (0.1, 1.0),
            'scale': (0.01, 0.1),
            'rescale': (1.0, 0.2),
            'num_line': (0.0, 2.0),
            'heatmap_sigma': (5.0, 2.5),
        }

        def value_from_type(augmentation_type):
                # Scale to be in range
                easy_value, hard_value = augmentation_ranges[augmentation_type]
                value = (hard_value - easy_value) * difficulty + easy_value
                value = (np.clip(value, easy_value, hard_value)
                         if easy_value < hard_value
                         else np.clip(value, hard_value, easy_value))
                return value
        
        def noisy_value_from_type(augmentation_type):
            # Get normal distributed random value
            if len(random_multipliers) == 0:
                random_multipliers.extend(
                        list(np.random.normal(size=(len(augmentation_ranges),))))
            return random_multipliers.pop() * value_from_type(augmentation_type)
        
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-iw_2], [-ih_2]]
        
        rotate_mat = np.asmatrix(np.eye(3))
        rotation_noise = noisy_value_from_type('rotation')
        
        if rotation_noise > 0:
            rotate_angle = np.radians(rotation_noise)
            cos_rotate = np.cos(rotate_angle)
            sin_rotate = np.sin(rotate_angle)
            rotate_mat[0, 0] = cos_rotate
            rotate_mat[0, 1] = -sin_rotate
            rotate_mat[1, 0] = sin_rotate
            rotate_mat[1, 1] = cos_rotate
            
        scale_mat = np.asmatrix(np.eye(3))
        scale = 1. + noisy_value_from_type('scale')
        scale_inv = 1. / scale
        np.fill_diagonal(scale_mat, ow / eye_width * scale)
        original_eyeball_radius = 71.7593
        eyeball_radius = original_eyeball_radius * scale_mat[0, 0]  # See: https://goo.gl/ZnXgDE
        eyeball_radius = np.float32(eyeball_radius)
        
        recentre_mat = np.asmatrix(np.eye(3))
        recentre_mat[0, 2] = iw/2 - eye_middle[0] + 0.5 * eye_width * scale_inv
        recentre_mat[1, 2] = ih/2 - eye_middle[1] + 0.5 * oh / ow * eye_width * scale_inv
        recentre_mat[0, 2] += noisy_value_from_type('translation')  # x
        recentre_mat[1, 2] += noisy_value_from_type('translation')  # y
        
        transform_mat = recentre_mat * scale_mat * rotate_mat * translate_mat
        
        self.eye = cv2.warpAffine(self.full_image, transform_mat[:2, :3], (ow, oh))
        self.eyeshape = cv2.warpAffine(self.mask1, transform_mat[:2, :3], (ow, oh))
        self.iris = cv2.warpAffine(self.mask2, transform_mat[:2, :3], (ow, oh))
        eye = self.eye
        eye = cv2.equalizeHist(eye)
        eye = resize_img(eye)
        eye = blur_img(eye)
        eye = change_contrast(eye)
        eye = blur_region(eye)
        eye = remove_region(eye)
        eye = add_line(eye)
        eye = noise(eye)
        eye = np.clip(eye,0,255)
        eye = eye.astype(np.uint8)
        
        mask1 = self.eyeshape
        mask1 = mask1.astype(np.uint8)
        mask1 = data_transforms(mask1[:,:,np.newaxis])
        
        mask2 = self.iris
        mask2 = mask2.astype(np.uint8)
        mask2 = data_transforms(mask2[:,:,np.newaxis])
        
        label=torch.cat((mask1,mask2),0)
        return data_transforms(eye[:,:,np.newaxis]), label
    
    def __len__(self):
        return 60000
