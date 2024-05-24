"""

WSI-level prediction of collagen content using model trained on patches

"""

import os
import numpy as np
import pandas as pd
from math import floor

from tqdm import tqdm
from glob import glob

from PIL import Image
import tifffile as ti
#import ome_types as ot

import torch
from torch.utils.data import Dataset

from Augmentation_Functions import normalize_01

from histolab.tiler import GridTiler
from histolab.slide import Slide

import datetime


class WSISegmentationDataSet(Dataset):
    def __init__(self,
                target_type: str,
                batch_size: int,
                wsi_dir:str,
                ):

        self.wsi_dir = wsi_dir
        self.target_type = target_type

        self.slide_idx = -1
        self.patch_size = 512
        self.batch_size = batch_size

        if target_type == 'binary':
            self.target_dtype = torch.long
        else:
            self.target_dtype = torch.float32

        self.slides = []
        wsi_filenames = glob(self.wsi_dir+'/*.svs')
        for i in tqdm(range(len(wsi_filenames))):
            self.slides.append(Slide(wsi_filenames[i],wsi_filenames[i].replace('.svs','_processed_path/')))
        
    def __len__(self):
        return len(self.slides)

    def __iter__(self):

        self.slide_idx+=1
        if self.slide_idx<len(self.slides):
            self.current_slide = self.slides[self.slide_idx]

            # Initialize prediction mask (reversed so it is height X width)
            self.combined_mask = np.zeros(self.current_slide.dimensions[::-1])

            if not self.slide_idx==0:
                previous_slide = self.slides[self.slide_idx-1]
                os.rmdir(previous_slide._processed_path)

            self.grid_extractor = GridTiler(
                tile_size = (self.patch_size,self.patch_size),
                level = 0,
                tissue_percent = 1.0,
                pixel_overlap = 500
            )

            # Saves extracted tiles to processed directory for each slide
            if not os.path.exists(self.current_slide._processed_path):
                print(f'Extracting tiles start: {datetime.datetime.now()}')
                self.grid_extractor.extract(self.current_slide)
                print(f'Extracting tiles stop: {datetime.datetime.now()}')
            self.patch_dir = glob(self.current_slide._processed_path+'*.png')
            self.patch_idx = 0

            self.batches = round(len(self.patch_dir)/self.batch_size)
            
            return self
        else:
            raise StopIteration
  
    def add_to_mask(self,pred_batch,coordinates):
        for idx,coords in enumerate(coordinates):
            self.combined_mask[coords[1]:coords[3],coords[0]:coords[2]]+=np.mean(np.concatenate((self.combined_mask[coords[1]:coords[3],coords[0]:coords[2],None],
                                                                                    pred_batch[idx,1,:,:,None]),axis=-1))

    def make_ome_tiff(self,cyz = False):

        if cyz:
            img = np.transpose(self.combined_mask,axes=(2,0,1))
        else:
            img = self.combined_mask

        img = img.astype(np.uint8)
        
        tiff_writer = ti.TiffWriter(self.current_slide._processed_path+'slide_tiff.ome.tiff',ome=True,bigtiff=True)
        if cyz:
            tiff_writer.write(img,metadata={'axes':'CYX'})
        else:
            tiff_writer.write(img,metadata={'axes':'XYC'})
        tiff_writer.close()

        tiff_file = ot.from_tiff(self.current_slide._processed_path+'slide_tiff.ome.tiff')
        xml_name = self.current_slide._processed_path+'slide_tiff.ome.xml'
        xml_data = ot.to_xml(tiff_file)
        xml_data = xml_data.replace('<Pixels','<Pixels PhysicalSizeXUnit="\u03BCm" PhysicalSizeYUnit="u03BCm"')
        with open(xml_name,'wt+') as fh:
            fh.write(xml_data)

    def make_tiff(self,save_path = None):
        if save_path is None:
            tiff_writer = ti.TiffWriter(self.current_slide._processed_path+'slide_tiff.tiff',ome=False,bigtiff=True)
        else:
            tiff_writer = ti.TiffWriter(save_path,ome=False,bigtiff=True)

        tiff_writer.write(self.combined_mask,metadata={'axes':'XYC'})
        tiff_writer.close()

    def __next__(self):
        
        # Getting new batch of images from slide processed path
        remaining_patches = len(self.patch_dir)-self.patch_idx

        if remaining_patches == 0:
            raise StopIteration
        else:
            if self.batch_size<=remaining_patches:
                batch_img_paths = self.patch_dir[self.patch_idx:self.patch_idx+self.batch_size]
                self.patch_idx+=self.batch_size
            else:
                batch_img_paths = self.patch_dir[self.patch_idx:self.patch_idx+remaining_patches]

            batch_img_list = []
            coords_list = []
            for img in batch_img_paths:

                current_img = normalize_01(np.array(Image.open(img))[:,:,0:3])
                current_img = np.moveaxis(current_img,source=-1,destination=0)
                batch_img_list.append(current_img)

                coords = img.split('_')[-1].replace('.png','').split('-')
                coords_list.append([int(i) for i in coords])

            batch_img_list = torch.from_numpy(np.stack(batch_img_list,axis=0)).type(torch.float32)

            return batch_img_list, coords_list



