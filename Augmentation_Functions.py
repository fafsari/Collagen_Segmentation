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


# Gets the input image into 0-1 range as opposed to 0-255
def normalize_01(inp: np.ndarray):

    inp_out = (inp - np.min(inp))/np.ptp(inp)
    
    return inp_out

# Normalize input array based on mean and standard deviation
def normalize(inp: np.ndarray, mean: list, std: list):
    
    inp_out = np.zeros_like(inp)
    for i in range(np.shape(inp)[-1]):
        inp_out[:,:,i] += (inp[:,:,i]-mean[i])/std[i]
    
    return inp_out

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

class ComposeTriple(Compose):

    def __call__(self, inp: np.ndarray, target1: dict, target2: dict):
        for t in self.transforms:
            inp, target = t(inp, target1, target2)
        return inp, target
    
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



