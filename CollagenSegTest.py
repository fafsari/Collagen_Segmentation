# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:28:08 2021

@author: spborder


CS Testing best model

from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb


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
from PIL import Image, ImageFilter
import pandas as pd

import neptune

from Segmentation_Metrics_Pytorch.metric import BinaryMetrics
from CollagenSegUtils import visualize_continuous, get_metrics
    
        
def Test_Network(model_path, dataset_valid, nept_run, test_parameters):

    model_details = test_parameters['model_details']
    encoder = model_details['encoder']
    encoder_weights = model_details['encoder_weights']

    ann_classes = model_details['ann_classes']
    active = model_details['active']
    target_type = model_details['target_type']
    output_dir = test_parameters['output_dir']

    if active == 'None':
        active = None

    if target_type=='binary':
        n_classes = len(ann_classes)
    elif target_type == 'nonbinary':
        n_classes = 1

    in_channels = int(test_parameters['preprocessing']['image_size'].split(',')[-1])
    output_type = model_details['target_type']

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    if model_details['architecture']=='Unet++':
        model = smp.UnetPlusPlus(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = in_channels,
                classes = n_classes,
                activation = active
                )

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():

        # Evaluating model on test set
        test_dataloader = DataLoader(dataset_valid)

        test_output_dir = output_dir+'Testing_Output/'
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
            
        if dataset_valid.testing_metrics:
            if target_type=='binary':
                metrics_calculator = BinaryMetrics()
                testing_metrics_df = pd.DataFrame(data = {'Dice':[],'Accuracy':[],'Recall':[],'Precision':[],'Specificity':[]})
            elif target_type =='nonbinary':
                metrics_calculator = []
                testing_metrics_df = pd.DataFrame(data = {'MSE':[],'Norm_MSE':[]})

        # Setting up iterator to generate images from the validation dataset
        data_iterator = iter(test_dataloader)
        with tqdm(range(len(dataset_valid)),desc='Testing') as pbar:
            for i in range(0,len(dataset_valid.images)):
                
                # Initializing combined mask from patched predictions
                if 'patch_batch' in dir(dataset_valid):

                    # Getting original image dimensions from test_dataloader
                    original_image, _ = dataset_valid.images[i]
                    original_image_size = np.shape(original_image)
                    final_pred_mask = np.zeros((original_image_size[0],original_image_size[1]))
                    overlap_mask = np.zeros_like(final_pred_mask)

                    patch_size = [int(i) for i in test_parameters['preprocessing']['image_size'].split(',')[0:-1]]
                    
                    # Now getting the number of patches needed for the current image
                    n_patches = len(dataset_valid.cached_data[i])
                    image_name = dataset_valid.cached_item_names[i]

                    for n in range(0,n_patches):

                        image, _, input_name = next(data_iterator)
                        input_name = ''.join(input_name).split(os.sep)[-1]

                        pred_mask = model(image.to(device))

                        if target_type=='binary':        
                            pred_mask_img = pred_mask.detach().cpu().numpy()

                        elif target_type=='nonbinary':
                            pred_mask_img = pred_mask.detach().cpu().numpy()

                        # Getting patch locations from input_name
                        row_start = int(input_name.split('_')[-2])
                        col_start = int(input_name.split('_')[-1].split('.')[0])

                        fig = visualize_continuous(
                            images = {'Pred_Mask':pred_mask_img},
                            output_type = 'prediction'
                        )
                        
                        # Adding to final_pred_mask and overlap_mask
                        final_pred_mask[row_start:row_start+patch_size[0],col_start:col_start+patch_size[1]] += (fig*255).astype(np.uint8)
                        overlap_mask[row_start:row_start+patch_size[0],col_start:col_start+patch_size[1]] += np.ones((patch_size[0],patch_size[1]))

                        pbar.update(1)

                    # Scaling predictions by overlap (mean pixel prediction where there is overlap)
                    final_pred_mask = np.multiply(final_pred_mask,1/overlap_mask)

                    im = Image.fromarray((final_pred_mask).astype(np.uint8))
                    # Smoothing image to get rid of grid lines
                    im = im.filter(ImageFilter.SMOOTH_MORE)
                    im.save(test_output_dir+f'{dataset_valid.cached_item_names[dataset_valid.cached_item_index].split(os.sep)[-1].replace(".tif","_prediction.tif")}')

                    # Saving overlap mask
                    #overlap_mask = (overlap_mask-np.min(overlap_mask))/(np.max(overlap_mask))
                    #overlap_im = Image.fromarray((overlap_mask*255).astype(np.uint8))
                    #overlap_im.save(test_output_dir+'Overlap_Mask.tif')

                else:

                    try:
                        image, target, input_name = next(data_iterator)
                        input_name = ''.join(input_name)
                    except StopIteration:
                        data_iterator = iter(test_dataloader)
                        image, target, input_name = next(data_iterator)
                        input_name = ''.join(input_name)

                    input_name = input_name.split('/')[-1]

                    # Add something here so that it calculates perforance metrics and outputs
                    # raw values for 2-class segmentation(not binarized output masks)
                    #pred_mask = model.predict(image.to(device))
                    pred_mask = model(image.to(device))

                    if target_type=='binary':        
                        target_img = target.cpu().numpy().round()
                        pred_mask_img = pred_mask.detach().cpu().numpy()

                        if dataset_valid.testing_metrics:
                            testing_metrics_df = testing_metrics_df.append(pd.DataFrame(get_metrics(pred_mask.detach().cpu(),target.cpu(), input_name, metrics_calculator,target_type)),ignore_index=True)
                        # Outputting the prediction as a continuous mask even though running binary metrics

                    elif target_type=='nonbinary':
                        pred_mask_img = pred_mask.detach().cpu().numpy()
                        target_img = target.cpu().numpy()

                        if dataset_valid.testing_metrics:
                            testing_metrics_df = testing_metrics_df.append(pd.DataFrame(get_metrics(pred_mask.detach().cpu(),target.cpu(), input_name, metrics_calculator,target_type)),ignore_index=True)

                    image = image.cpu().numpy()
                    if type(in_channels)==int:
                        if in_channels==6:
                            image = np.concatenate((image[:,0:3,:,:],image[:,2:5,:,:]),axis=2)
                        elif in_channels == 4:
                            image = np.concatenate((np.stack((image[:,0,:,:],)*3,axis=1),image[:,0:3,:,:]),axis=2)
                        elif in_channels == 2:
                            image = np.concatenate((image[:,0,:,:],image[:,1,:,:]),axis=-1)
                    elif type(in_channels)==list:
                        if sum(in_channels)==6:
                            image = np.concatenate((image[:,0:3,:,:],image[:,2:5,:,:]),axis=2)
                        elif sum(in_channels)==2:
                            image = np.concatenate((image[:,0,:,:][None,:,:],image[:,1,:,:][None,:,:]),axis=2)

                    img_dict = {'Image':image,'Pred_Mask':pred_mask_img,'Ground_Truth':target_img}

                    fig = visualize_continuous(img_dict,output_type)

                    pbar.update(1)

                    # Different process for saving comparison figures vs. only predictions
                    if output_type=='comparison':
                        fig.savefig(test_output_dir+'Test_Example_'+input_name)
                        #nept_run['testing/Testing_Output_'+input_name].upload(test_output_dir+'Test_Example_'+input_name)
                    elif output_type=='prediction':

                        im = Image.fromarray((fig*255).astype(np.uint8))
                        im.save(test_output_dir+'Test_Example_'+input_name.replace('.jpg','.tif'))

                # Used during hyperparameter optimization to compute objective value
                if dataset_valid.testing_metrics:
                    testing_metrics_df.to_csv(test_output_dir+'Test_Metrics.csv')
                
                    if not 'current_k_fold' in test_parameters:
                        nept_run['Test_Image_metrics'].upload(neptune.types.File.as_html(testing_metrics_df))

                        for met in testing_metrics_df.columns.values.tolist():
                            try:
                                print(f'{met} value: {testing_metrics_df[met].mean()}')
                                nept_run[met] = testing_metrics_df[met].mean()
                            except TypeError:
                                print(f'Number of samples: {testing_metrics_df.shape[0]}')
                    else:
                        current_k_fold = test_parameters['current_k_fold']
                        nept_run[f'Test_Image_metrics_{current_k_fold}'].upload(neptune.types.File.as_html(testing_metrics_df))

                        for met in testing_metrics_df.columns.values.tolist():
                            try:
                                print(f'{met}: value: {testing_metrics_df[met].mean()}')
                                nept_run[met+f'_{current_k_fold}'] = testing_metrics_df[met].mean()
                            except TypeError:
                                print(f'Number of samples: {testing_metrics_df.shape[0]}')
