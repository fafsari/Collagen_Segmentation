"""

Collagen segmentation prediction pipeline for WSI input


"""

from typing import Type
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from tqdm import tqdm
import os
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import neptune.new as neptune

from Segmentation_Metrics_Pytorch.metric import BinaryMetrics
from CollagenSegUtils import visualize_continuous, get_metrics, visualize_multi_task


def Test_Network(model_path,test_dataset,output_dir,test_parameters):

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    if test_parameters['target_type']=='binary':
        n_classes = 2
    else:
        n_classes = 1

    model = smp.UnetPlusPlus(
        encoder_name = test_parameters['encoder'],
        encoder_weights = test_parameters['encoder_weights'],
        in_channels = 3,
        classes = n_classes,
        active = test_parameters['active']
    )

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():

        test_dataloader = iter(test_dataset)

        for i in tqdm(len(test_dataloader.slides)):

            for j in range(test_dataloader.batches):

                img_batch, coords = next(test_dataloader)
                pred_masks = model.predict(img_batch.to(device))

                # Assembling predicted masks into combined tif file
                test_dataloader.add_to_mask(pred_masks.detach().cpu().numpy(),coords)
            
            final_mask = Image.fromarray(test_dataloader.combined_mask)
            final_mask.save(output_dir+test_dataloader.current_slide.name+'.tif')

            test_dataloader = iter(test_dataloader)



