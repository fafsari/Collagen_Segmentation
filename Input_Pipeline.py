# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:35:06 2021

@author: spborder


Data input pipeline for Deep Glomerular Compartment Segmentation

from: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
"""

import os
import pandas as pd
import numpy as np
from math import floor

from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread
from glob import glob

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import albumentations

from skimage.transform import resize

from Augmentation_Functions import *
from CollagenSegUtils import resize_special

# Input class with required len and getitem functions
# in this case, the inputs and targets are lists of paths matching paths for image and segmentation ground-truth
# *** Have to make sure the input and target paths are aligned ***
# Need to change shape of input data to be (batch size, channels, height, width)
# with linear scaling depending on network expectations,
# target to be (batch size, height, width) with dense integer encoding for each class

class SegmentationDataSet(Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform = None,
                 use_cache = False,
                 pre_transform = None,
                 target_type = None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        if target_type == 'binary':
            self.targets_dtype = torch.long
        elif target_type == 'nonbinary':
            self.targets_dtype = torch.float32
        
        # increasing dataset loading efficiency
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        
        if self.use_cache:
            self.cached_data = []
            self.cached_names = []
            
            progressbar = tqdm(range(len(self.inputs)), desc = 'Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                if 'tif' in tar_name:
                    img, tar = imread(str(img_name)), imread(str(tar_name),plugin='pil')

                else:
                    img, tar = imread(str(img_name)), imread(str(tar_name))
                
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)
                    #print(f'Size of input image:{np.shape(img)}')
                    #print(f'Size of target:{np.shape(tar)}')
                    
                self.cached_data.append((img,tar))
                self.cached_names.append(img_name)
        
    def __len__(self):
        return len(self.inputs)
    
    # Getting matching input and target(label)
    def __getitem__(self,
                    index: int):
        
        if self.use_cache:
            x, y = self.cached_data[index]
            input_ID = self.cached_names[index]
        else:
            input_ID = self.inputs[index]
            target_ID = self.targets[index]
        
            # Reading x(input) and y(target) images using skimage.io.imread
            if 'tif' in target_ID:
                x, y = imread(input_ID),imread(target_ID,plugin='pil')
            else:
                x, y = imread(input_ID),imread(target_ID)
        
        # Preprocessing steps (if there are any)
        if self.transform is not None:
            x, y = self.transform(x, y)
            
        # Getting in the right input/target data types
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        
        return x, y, input_ID
    
def stupid_mask_thing(target):
    #print('Target shape{}'.format(np.shape(target)))
    #print('Unique Values{}'.format(np.unique(target)))
    if np.shape(target)[-1] != 3:
        unique_vals = np.unique(target)
        final = np.zeros((np.shape(target)[0],np.shape(target)[1],len(unique_vals)))
        for idx,value in enumerate(unique_vals):
            
            dummy = np.zeros_like(target)
            mask = np.where(target==value)
            dummy[mask] = 1
            
            final[:,:,idx] += dummy
    else:
        
        """
        new_target = np.stack((np.zeros((256,256)), target[:,:,0],target[:,:,1],target[:,:,2]), -1) 
        mask = np.where(target.sum(axis=-1)==0)
        new_target[:,:,0][mask] = 1
        
        final = new_target
        """
        
        # Fix for binary labels
        new_target = np.zeros((256,256,2))
        mask = np.where(target.sum(axis=-1)<100)
        new_target[:,:,0][mask] = 1
        mask = np.where(target.sum(axis=-1)>100)
        new_target[:,:,1][mask] = 1
        
        final = new_target
        
        
    return final

    

def make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar,target_type,parameters):
 
    color_transform = parameters['in_channels']
    if phase == 'train' or phase == 'optimize':

        if target_type=='binary':
            pre_transforms = ComposeDouble([
                    FunctionWrapperDouble(resize_special,
                                        input = True,
                                        target = False,
                                        output_size = (512,512,3),
                                        transform = color_transform),
                    FunctionWrapperDouble(resize_special,
                                        input = False,
                                        target = True,
                                        output_size = (512,512,3),
                                        transform = color_transform),
                    FunctionWrapperDouble(stupid_mask_thing,
                                        input = False,
                                        target = True)
            ])
        elif target_type=='nonbinary':
            pre_transforms = ComposeDouble([
                    FunctionWrapperDouble(resize_special,
                                        input = True,
                                        target = False,
                                        output_size = (512,512,3),
                                        transform = color_transform),
                    FunctionWrapperDouble(resize_special,
                                        input = False,
                                        target = True,
                                        output_size = (512,512,1),
                                        transform = color_transform)
            ])        

        if target_type=='binary':
            # Training transformations + augmentations
            transforms_training = ComposeDouble([
                    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
                    AlbuSeg2d(albumentations.IAAPerspective(p=0.5)),
                    AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
                    FunctionWrapperDouble(create_dense_target, input = False, target = True),
                    FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0),
                    FunctionWrapperDouble(normalize_01, input = True, target = True)
                    ])
                
            # Validation transformations 
            transforms_validation = ComposeDouble([
                    FunctionWrapperDouble(create_dense_target, input = False, target = True),
                    FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0),
                    FunctionWrapperDouble(normalize_01, input = True, target = True)
                    ])
        elif target_type=='nonbinary':
            # Continuous target type augmentations
            transforms_training = ComposeDouble([
                AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
                AlbuSeg2d(albumentations.IAAPerspective(p=0.5)),
                AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
                FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0),
                FunctionWrapperDouble(normalize_01, input = True, target = True)
            ])

            transforms_validation = ComposeDouble([
                FunctionWrapperDouble(np.moveaxis,input=True,target=True,source=-1,destination=0),
                FunctionWrapperDouble(normalize_01, input = True, target = True)
            ])

        
        dataset_train = SegmentationDataSet(inputs = train_img_paths,
                                             targets = train_tar,
                                             transform = transforms_training,
                                             use_cache = True,
                                             pre_transform = pre_transforms,
                                             target_type = target_type)
        
        dataset_valid = SegmentationDataSet(inputs = valid_img_paths,
                                             targets = valid_tar,
                                             transform = transforms_validation,
                                             use_cache = True,
                                             pre_transform = pre_transforms,
                                             target_type = target_type)
        
    elif phase == 'test':
        
        color_transform = parameters['in_channels']
        if target_type=='binary':
            pre_transforms = ComposeDouble([
            FunctionWrapperDouble(resize_special,
                                input = True,
                                target = False,
                                output_size = (512,512,3),
                                transform = color_transform),
            FunctionWrapperDouble(resize_special,
                                input = False,
                                target = True,
                                output_size = (512,512,3),
                                transform = color_transform),
            FunctionWrapperDouble(stupid_mask_thing,
                                input = False,
                                target = True)
            ])
            
            transforms_testing = ComposeDouble([
                    FunctionWrapperDouble(create_dense_target, input = False, target = True),
                    FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0),
                    FunctionWrapperDouble(normalize_01, input = True, target = True)
                    ])
        
        elif target_type=='nonbinary':

            pre_transforms = ComposeDouble([
            FunctionWrapperDouble(resize_special,
                                input = True,
                                target = False,
                                output_size = (512,512,3),
                                transform = color_transform),
            FunctionWrapperDouble(resize_special,
                                input = False,
                                target = True,
                                output_size = (512,512,1),
                                transform = color_transform)
            ])
            
            transforms_testing = ComposeDouble([
                    FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0),
                    FunctionWrapperDouble(normalize_01, input = True, target = True)
                    ])

        # this is 'None' because we are just testing the network
        dataset_train = None
        
        dataset_valid = SegmentationDataSet(inputs = valid_img_paths,
                                            targets = valid_tar,
                                            transform = transforms_testing,
                                            use_cache = True,
                                            pre_transform = pre_transforms,
                                            target_type = target_type)
        
    
    return dataset_train, dataset_valid
    















