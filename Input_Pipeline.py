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
from math import floor, ceil

from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from glob import glob

from random import sample

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
                 target_type = None,
                 batch_size = None,
                 parameters = {}):
        

        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        if type(target_type)==list:
            self.targets_dtype = torch.float32
            self.multi_task = True
        else:
            self.multi_task = False
            if target_type == 'binary':
                self.targets_dtype = torch.long
            elif target_type == 'nonbinary':
                self.targets_dtype = torch.float32
        
        self.batch_size = batch_size

        # Adding for predicting on large (but not that large) images
        if 'patch_batch' in parameters:
            self.patch_batch = parameters['patch_batch']
            self.cached_item_names = []
        else:
            self.patch_batch = False

        print(f'patch_batch: {self.patch_batch}')

        if len(self.targets)==0:
            self.testing_metrics = False
        else:
            self.testing_metrics = True

        # increasing dataset loading efficiency
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        
        if self.use_cache:
            self.cached_data = []
            self.cached_names = []

            self.image_means = []
            self.image_stds = []
            
            progressbar = tqdm(range(len(self.inputs)), desc = 'Caching')

            if len(self.targets)>0:
                for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                    try:
                        if 'tif' in tar_name:
                            img, tar = imread(str(img_name)), imread(str(tar_name),plugin='pil')

                        else:
                            # For multi-channel image inputs
                            if type(img_name)==list:
                                img1,img2,tar = imread(str(img_name[0])), imread(str(img_name[1])),imread(str(tar_name))
                                img = np.concatenate((img1,img2),axis=-1)
                                img_name = img_name[0]
                            else:
                                img, tar = imread(str(img_name)), imread(str(tar_name))
                        
                        if self.pre_transform is not None:
                            img, tar = self.pre_transform(img, tar)

                        # Calculating dataset mean and standard deviation
                        img_channel_mean = np.mean(img,axis=(0,1))
                        img_channel_std = np.std(img,axis=(0,1))

                        self.image_means.append(img_channel_mean)
                        self.image_stds.append(img_channel_std)
                            
                        self.cached_data.append((img,tar))
                        self.cached_names.append(img_name)
                    except FileNotFoundError:
                        print(f'File not found: {img_name},{tar_name}')
            else:
                try:
                    for i, img_name in zip(progressbar, self.inputs):

                        if type(img_name)==list:
                            img1,img2 = imread(str(img_name[0])), imread(str(img_name[1]))
                            img = np.concatenate((img1,img2),axis=-1)
                            img_name = img_name[0]

                        else:
                            img = imread(str(img_name))
                        
                        tar = np.zeros((np.shape(img)[0],np.shape(img)[1],1))

                        if not self.patch_batch:
                            if self.pre_transform is not None:
                                img, tar = self.pre_transform(img, tar)

                            # Calculating dataset mean and standard deviation
                            img_channel_mean = np.mean(img,axis=(0,1))
                            img_channel_std = np.std(img,axis=(0,1))

                            self.image_means.append(img_channel_mean)
                            self.image_stds.append(img_channel_std)

                            self.cached_data.append((img,tar))
                            self.cached_names.append(img_name)
                        
                        else:

                            # Overlap percentage defined in self.patch_batch, hardcoded patch size
                            patch_size = [512,512]
                            start_coords = [0,0]
                            stride = [int(patch_size[0]*(1-self.patch_batch)), int(patch_size[1]*(1-self.patch_batch))]
                            n_patches = [1+floor((np.shape(img)[0]-patch_size[0])/stride[0]), 1+floor((np.shape(img)[1]-patch_size[1])/stride[1])]

                            row_starts = [int(start_coords[0]+(i*stride[0])) for i in range(0,n_patches[0])]
                            col_starts = [int(start_coords[1]+(i*stride[1])) for i in range(0,n_patches[1])]
                            row_starts.append(int(np.shape(img)[0]-patch_size[0]))
                            col_starts.append(int(np.shape(img)[1]-patch_size[1]))
                            
                            self.original_image_size = list(np.shape(img))
                            self.patch_size = patch_size

                            #self.cached_data.append((img,tar))
                            self.cached_item_names.append(img_name)

                            for r_s in row_starts:
                                for c_s in col_starts:
                                    new_img = img[r_s:r_s+patch_size[0], c_s:c_s+patch_size[1],:]
                                    new_tar = np.zeros((np.shape(new_img)[0],np.shape(new_img)[1]))

                                    if self.pre_transform is not None:
                                        new_img, new_tar = self.pre_transform(new_img, new_tar)

                                    img_channel_mean = np.mean(new_img,axis=(0,1))
                                    img_channel_std = np.std(img,axis=(0,1))
                                    self.image_means.append(img_channel_mean)
                                    self.image_stds.append(img_channel_std)

                                    self.cached_data.append((new_img,new_tar))
                                    self.cached_names.append(img_name.replace(f'.{img_name.split(".")[-1]}',f'_{r_s}_{c_s}.{img_name.split(".")[-1]}'))
                            

                except FileNotFoundError:
                    print(f'File not found: {img_name}')

            print(f'Cached Data: {len(self.cached_data)}')
            print(f'image_means mean: {np.mean(self.image_means,axis=0)}')
            print(f'image_stds mean: {np.mean(self.image_stds,axis=0)}')


    def __len__(self):
        if not self.patch_batch:
            return len(self.cached_data)
        else:
            return len(self.cached_names)
    
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
    
    def add_sub_categories(self,sub_categories, sub_cat_column):
        # Implementing sub-category weighting if provided with dataframe
        sub_cat_counts = sub_categories[sub_cat_column].value_counts(normalize=True).to_dict()
        print(f'Dataset composition: {sub_cat_counts}')

        # Applying inverse weight to each sample (evens out sampling)
        sub_cat_counts = {i:1.0-j for i,j in zip(list(sub_cat_counts.keys()),list(sub_cat_counts.values()))}

        # Calculating sample weight
        self.sample_weight = []
        for s in self.cached_names:
            sample_name = s.split('/')[-1]
            print(sample_name)
            if sample_name in sub_categories['Image_Names'].tolist():
                sample_label = sub_categories[sub_categories['Image_Names'].str.match(sample_name)]['Labels'].tolist()[0]
                print(sample_label)
                self.sample_weight.append(sub_cat_counts[sample_label])
            else:
                print(f'sample not in dataframe: {sample_name}')
                self.sample_weight.append(0)

        self.sample_weight = [i/sum(self.sample_weight) for i in self.sample_weight]
        print(f'sample_weight: {self.sample_weight}')
    
    def __iter__(self):

        return self
    
    def __next__(self):

        img_list = []
        tar_list = []
        name_list = []
        for b in range(self.batch_size):
            
            s_idx = np.random.choice(list(range(len(self.cached_data))),p=self.sample_weight)

            if self.use_cache:
                img, tar = self.cached_data[s_idx]
                input_id = self.cached_names[s_idx]

                # Add cache-less later
            
            if self.transform is not None:
                x, y = self.transform(img,tar)
            
            x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

            img_list.append(x)
            tar_list.append(y)
            name_list.append(input_id)

        return torch.stack(img_list), torch.stack(tar_list), name_list

    def normalize_cache(self,means,stds):
        # Applying normalization to a dataset according to a given set of means and standard deviations per channel
        for img,tar in self.cached_data:
            img = np.float64(img)
            for j in range(img.shape[-1]):
                img[:,:,j] -= means[j]
                img[:,:,j] /= stds[j]


def stupid_mask_thing(target):
    if np.shape(target)[-1] != 3:
        unique_vals = np.unique(target)
        final = np.zeros((np.shape(target)[0],np.shape(target)[1],len(unique_vals)))
        for idx,value in enumerate(unique_vals):
            
            dummy = np.zeros_like(target)
            mask = np.where(target==value)
            dummy[mask] = 1
            
            final[:,:,idx] += dummy
    else:
        
        # Fix for binary labels
        new_target = np.zeros((512,512,2))
        mask = np.where(target.sum(axis=-1)==0)
        new_target[:,:,0][mask] = 1
        mask = np.where(target.sum(axis=-1)>0)
        new_target[:,:,1][mask] = 1
        
        final = new_target
        
        
    return final

def make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar,parameters):
 
    image_dim = 512

    if 'color_transform' in parameters:
        color_transform = parameters['color_transform']
    else:
        color_transform = ''

    if type(parameters['in_channels'])==int:
        """
        if parameters['in_channels'] == 6:
            img_size = (image_dim,image_dim,6)
        elif parameters['in_channels'] == 4:
            img_size = (image_dim,image_dim,4)
        elif parameters['in_channels'] == 2:
            img_size = (image_dim,image_dim,2)
        else:
            img_size = (image_dim,image_dim,3)
        """
        img_size = (image_dim,image_dim,parameters['in_channels'])
    elif type(parameters['in_channels'])==list:
        img_size = (image_dim,image_dim,sum(parameters['in_channels']))


    mask_size = (image_dim,image_dim,1)
    
    target_type = parameters['target_type']
    batch_size = parameters['batch_size']

    if phase == 'train' or phase == 'optimize':

        if target_type=='binary':
            pre_transforms = ComposeDouble([
                    FunctionWrapperDouble(resize_special,
                                        input = True,
                                        target = False,
                                        output_size = img_size,
                                        transform = color_transform),
                    FunctionWrapperDouble(resize,
                                        input = False,
                                        target = True,
                                        output_shape = img_size),
                    FunctionWrapperDouble(stupid_mask_thing,
                                        input = False,
                                        target = True)
            ])
        elif target_type=='nonbinary':
            pre_transforms = ComposeDouble([
                    FunctionWrapperDouble(resize_special,
                                        input = True,
                                        target = False,
                                        output_size = img_size,
                                        transform = color_transform),
                    FunctionWrapperDouble(resize,
                                        input = False,
                                        target = True,
                                        output_shape = mask_size)
            ])        

        if target_type=='binary':
            # Training transformations + augmentations
            transforms_training = ComposeDouble([
                    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
                    AlbuSeg2d(albumentations.IAAPerspective(p=0.5)),
                    AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
                    AlbuSeg2d(albumentations.IAAPiecewiseAffine(p=0.2)),
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
                #AlbuSeg2d(albumentations.IAAPerspective(p=0.5)),
                AlbuSeg2d(albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5,rotate_limit=45,interpolation=1,p=0.1)),
                AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
                FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0)
                #FunctionWrapperDouble(normalize_01, input = True, target = True)
            ])

            transforms_validation = ComposeDouble([
                FunctionWrapperDouble(np.moveaxis,input=True,target=True,source=-1,destination=0)
                #FunctionWrapperDouble(normalize_01, input = True, target = True)
            ])

        
        dataset_train = SegmentationDataSet(inputs = train_img_paths,
                                             targets = train_tar,
                                             transform = transforms_training,
                                             use_cache = True,
                                             pre_transform = pre_transforms,
                                             target_type = target_type,
                                             batch_size = batch_size,
                                             parameters = parameters)
        
        dataset_valid = SegmentationDataSet(inputs = valid_img_paths,
                                             targets = valid_tar,
                                             transform = transforms_validation,
                                             use_cache = True,
                                             pre_transform = pre_transforms,
                                             target_type = target_type,
                                             batch_size = batch_size,
                                             parameters = parameters)
        
    elif phase == 'test':
        
        if 'color_transform' in parameters:
            color_transform = parameters['color_transform']

        if type(parameters['in_channels'])==int:
            img_size = (image_dim,image_dim,parameters['in_channels'])
        elif type(parameters['in_channels'])==list:
            img_size = (image_dim,image_dim,sum(parameters['in_channels']))
            
        mask_size = (image_dim,image_dim,1)

        if target_type=='binary':
            pre_transforms = ComposeDouble([
            FunctionWrapperDouble(resize_special,
                                input = True,
                                target = False,
                                output_size = img_size,
                                transform = color_transform),
            FunctionWrapperDouble(resize,
                                input = False,
                                target = True,
                                output_shape = img_size),
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
                                output_size = img_size,
                                transform = color_transform),
            FunctionWrapperDouble(resize,
                                input = False,
                                target = True,
                                output_shape = mask_size)
            ])
            
            transforms_testing = ComposeDouble([
                    FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0),
                    #FunctionWrapperDouble(normalize_01, input = True, target = True)
                    ])

        # this is 'None' because we are just testing the network
        dataset_train = None
        
        dataset_valid = SegmentationDataSet(inputs = valid_img_paths,
                                            targets = valid_tar,
                                            transform = transforms_testing,
                                            use_cache = True,
                                            pre_transform = pre_transforms,
                                            target_type = target_type,
                                            batch_size = batch_size,
                                            parameters = parameters)


    return dataset_train, dataset_valid
    















