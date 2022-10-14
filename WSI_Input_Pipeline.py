"""

WSI-level prediction of collagen content using model trained on patches

"""

import os
from tkinter import Grid
import numpy as np
import pandas as pd
from math import floor

from tqdm import tqdm
from glob import glob

import openslide

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import albumentations

from skimage.transform import resize
from Augmentation_Functions import *
from CollagenSegUtils import resize_special

from histolab.tiler import GridTiler


class WSISegmentationDataSet(Dataset):
    def __init__(self,
                input_type: str,
                target_type: str,
                wsi_dir = None,
                transforms = None,
                mask_dir = None
                ):

        self.wsi_dir = wsi_dir
        self.input_type = input_type
        self.target_type = target_type
        self.transforms = transforms
        self.mask_dir = mask_dir

        self.slide_idx = -1
        self.patch_size = 512
        self.overlap = 0.25

        if input_type == 'multi_image':
            self.multi_image = True
        else:
            self.multi_image = False

        if target_type == 'binary':
            self.target_dtype = torch.Long
        else:
            self.target_dtype = torch.float32

        self.slides = []
        if not self.multi_image:
            wsi_filenames = glob(self.wsi_dir+'*.svs')

            for slide in wsi_filenames:
                self.slides.append(openslide.OpenSlide(slide))

        else:
            wsi_filenames = []
            for f in os.listdir(self.wsi_dir):
                wsi_list = glob(self.wsi_dir+os.sep+f+'/*.svs')
                wsi_list = sorted(wsi_list)
                wsi_filenames.append(wsi_list)
            
            for i in range(len(wsi_filenames[0])):
                self.slides.append([openslide.OpenSlide(wsi_filenames[j][i]) for j in range(len(wsi_filenames))])
            
    def __len__(self):
        return len(self.slides)

    def __iter__(self):

        self.slide_idx+=1
        self.current_slide = self.slides[self.slide_idx]

        self.grid_extractor = GridTiler(
            tile_size 
        )

    def __next__(self):
        






































