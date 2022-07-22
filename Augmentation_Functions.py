# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:06:56 2021

@author: spborder


Functions necessary to transform data before input into segmentation network

from: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55
"""

from typing import List, Callable, Tuple

import numpy as np
import albumentations as A
#from sklearn.externals._pilutil import bytescale
from skimage.util import crop
from sklearn.model_selection import train_test_split


# Gets the input image into 0-1 range as opposed to 0-255
def normalize_01(inp: np.ndarray):
    
    inp_out = (inp - np.min(inp))/np.ptp(inp)
    
    return inp_out

# Normalize input array based on mean and standard deviation
def normalize(inp: np.ndarray, mean: float, std: float):
    
    inp_out = (inp-mean)/std
    
    return inp_out

# Getting target array into right format
def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    
    # For each class, replace region in target (labeled image) with a unique integer
    # from 0 to the number of classes
    for idx, value in enumerate(classes):
        mask = np.where(tar==value)
        dummy[mask] = idx

    return dummy

# Center cropping array (alternative to resizing?)
# '->' is a function annotation in Python which is accessible through function_name.__annotations__
# pretty cool feature
def center_crop_to_size(x: np.ndarray,
                        size: Tuple,
                        copy: bool = False) -> np.ndarray:
    
    x_shape = np.array(x.shape)
    size = np.array(size)
    params_list = ((x_shape-size)/2).astype(np.int).tolist()
    params_tuple = tuple([(i,i) for i in params_list])
    cropped_image = crop(x, crop_width=params_tuple, copy=copy)
    return cropped_image

# Re-normalizing image to between 0 and 255
def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255):
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out

# Randomly flipping input and target according to a set ratio
def random_flip(inp:np.ndarray, tar: np.ndarray, ndim_spatial: int):
    flip_dims = [np.random.randint(low=0, high = 2) for dim in range(ndim_spatial)]
    
    # input has extra dimension compared to target
    flip_dims_inp = tuple([i+1 for i, element in enumerate(flip_dims) if element == 1])
    flip_dims_tar = tuple([i for i, element in enumerate(flip_dims) if element == 1])

    inp_flipped = np.flip(inp, axis = flip_dims_inp)
    tar_flipped = np.flip(tar, axis = flip_dims_tar)
    
    return inp_flipped, tar_flipped

# Extra class thing to work with string representations of objects
class Repr:
    
    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'
    
    
# Wrapper for only inputs
class FunctionWrapperSingle(Repr):
    
    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        
    def __call__(self, inp:np.ndarray): return self.function(inp)
    
# Wrapper for input-target pairs
class FunctionWrapperDouble(Repr):
    
    def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target
        
    def __call__(self, inp: np.ndarray, tar: dict):
        if self.input: inp = self.function(inp)
        if self.target: tar = self.function(tar)
        return inp, tar

# Special case with multiple targets
class FunctionWrapperTriple(Repr):

    def __init__(self, function: Callable, input: bool = True, target1: bool = False, target2: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function,*args,**kwargs)
        self.input = input
        self.target1 = target1
        self.target2 = target2

    def __call__(self, inp: np.ndarray, tar1: dict, tar2: dict):
        if self.input: inp = self.function(inp)
        if self.target1 and self.target2: tar = self.function(tar1,tar2)
        #if self.target2: tar2 = self.function(tar2)
        return inp, tar

# Composing multiple transforms together
class Compose:
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
        
    def __repr__(self): return str([transform for transform in self.transforms])
    
# When there are input-target pairs
class ComposeDouble(Compose):
    
    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target
    
# When there is only an input
class ComposeSingle(Compose):
    
    def __call__(self, inp: np.ndarray):
        for t in self.transforms:
            inp = t(inp)
        return inp
    
# Wrapper for 2D albumentations
# expected input : (channels, spatial dimensions)
# expected target : (spatial dimensions) no channels
class AlbuSeg2d(Repr):
    
    def __init__(self, albumentation: Callable):
        self.albumentation = albumentation
        
    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        
        out_dict = self.albumentation(image = inp, mask = tar)
        input_out = out_dict['image']
        target_out = out_dict['mask']
        
        return input_out, target_out
    
# Skipping AlbuSeg3d because this is only with 2D
    
# Randomly flipping inputs and targets    
# expected input : (channels, spatial dimensions)
# expected targets : (spatial dimensions)

# Arguments: ndim_spatial: Number of spatial dimensions in input
# ndim_spatial = 2 for input shape (channels, height, width)
# ndim_spatial = 3 for input shape (channels, dimensions, height, width)
class RandomFlip(Repr):
    
    def __init__(self, ndim_spatial):
        self.ndim_spatial = ndim_spatial
        
    def __call__(self, inp, target):
        
        flip_dims = [np.random.randint(low=0, high = 2) for dim in range(self.ndim_spatial)]
        
        flip_dims_inp = tuple([i+1 for i, element in enumerate(flip_dims) if element == 1])
        flip_dims_target = tuple([i for i, element in enumerate(flip_dims) if element == 1])
        
        inp_flip - np.flip(inp, axis = flip_dims_inp)
        target_flip = np.flip(target, axis = flip_dims_target)
        
        return np.copy(inp_flip), np.copy(target_flip)
    
    



