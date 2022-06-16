# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:28:08 2021

@author: spborder


DGCS Testing best model

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
import pandas as pd

import neptune.new as neptune

from Segmentation_Metrics_Pytorch.metric import BinaryMetrics



def back_to_reality(tar):
    
    # Getting target array into right format
    classes = np.shape(tar)[-1]
    dummy = np.zeros((np.shape(tar)[0],np.shape(tar)[1]))
    for value in range(classes):
        mask = np.where(tar[:,:,value]!=0)
        dummy[mask] = value

    return dummy

def apply_colormap(img):

    n_classes = np.shape(img)[-1]

    image = img[:,:,0]
    for cl in range(1,n_classes):
        image = np.concatenate((image, img[:,:,cl]),axis = 1)

    return image
    
def visualize(images,output_type):
    
    n = len(images)
    
    for i,key in enumerate(images):

        plt.subplot(1,n,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(key)
        
        if len(np.shape(images[key]))==4:
            img = images[key][0,:,:,:]
        else:
            img = images[key]
            
        img = np.float32(np.moveaxis(img, source = 0, destination = -1))
        if key == 'Pred_Mask' or key == 'Ground_Truth':
            if output_type=='binary' or key == 'Ground_Truth':
                #print('using back_to_reality')
                img = back_to_reality(img)

                plt.imshow(img)
            if output_type == 'continuous' and not key == 'Ground_Truth':
                #print('applying colormap')
                img = apply_colormap(img)

                plt.imshow(img,cmap='jet')
        else:
            plt.imshow(img)

    return plt.gcf()

def visualize_continuous(images,output_type):

    n = len(images)
    for i,key in enumerate(images):

        plt.subplot(1,n,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(key)

        if len(np.shape(images[key])) == 4:
            img = images[key][0,:,:,:]
        else:
            img = images[key]

        img = np.float32(img)

        if np.shape(img)[0]<np.shape(img)[-1]:
            img = np.moveaxis(img,source=0,destination=-1)

        if key == 'Pred_Mask' or key == 'Ground_Truth':
            img = apply_colormap(img)
            plt.imshow(img,cmap='jet')

        else:
            plt.imshow(img)

    return plt.gcf()

def get_metrics(pred_mask,ground_truth,img_name,calculator,target_type):

    metrics_row = {}

    if target_type=='binary':
        edited_gt = ground_truth[:,1,:,:]
        edited_gt = torch.unsqueeze(edited_gt,dim = 1)
        edited_pred = pred_mask[:,1,:,:]
        edited_pred = torch.unsqueeze(edited_pred,dim = 1)

            #print(f'edited pred_mask shape: {edited_pred.shape}')
            #print(f'edited ground_truth shape: {edited_gt.shape}')
            #print(f'Unique values prediction mask : {torch.unique(edited_pred)}')
            #print(f'Unique values ground truth mask: {torch.unique(edited_gt)}')

        acc, dice, precision, recall,specificity = calculator(edited_gt,torch.round(edited_pred))
        metrics_row['Accuracy'] = [round(acc.numpy().tolist(),4)]
        metrics_row['Dice'] = [round(dice.numpy().tolist(),4)]
        metrics_row['Precision'] = [round(precision.numpy().tolist(),4)]
        metrics_row['Recall'] = [round(recall.numpy().tolist(),4)]
        metrics_row['Specificity'] = [round(specificity.numpy().tolist(),4)]
        
        #print(metrics_row)
    elif target_type == 'nonbinary':
        square_diff = (ground_truth.numpy()-pred_mask.numpy())**2
        mse = np.mean(square_diff)

        metrics_row['MSE'] = [round(mse,4)]

    metrics_row['ImgLabel'] = img_name

    return metrics_row
    
        
def Test_Network(classes, model_path, dataset_valid, output_dir, nept_run, test_parameters, target_type):

    encoder = test_parameters['encoder']
    encoder_weights = test_parameters['encoder_weights']
    ann_classes = classes
    active = test_parameters['active']

    if target_type=='binary':
        n_classes = len(ann_classes)
    elif target_type == 'nonbinary':
        n_classes = 1

    output_type = test_parameters['output_type']

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    if test_parameters['architecture']=='Unet++':
        model = smp.UnetPlusPlus(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = 3,
                classes = n_classes,
                activation = active
                )
    elif test_parameters['architecture'] == 'Unet':
        model = smp.Unet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = n_classes,
            activation = active
        )
    elif test_parameters['architecture'] == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = n_classes,
            activation = active
        )
    elif test_parameters['architecture'] == 'DeepLabV3+':
        model = smp.DeepLabPlus(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = n_classes,
            activation = active
        )
    elif test_parameters['architecture'] == 'MANet':
        model = smp.MANet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
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
            
        if target_type=='binary':
            metrics_calculator = BinaryMetrics()
            testing_metrics_df = pd.DataFrame(data = {'Dice':[],'Accuracy':[],'Recall':[],'Precision':[],'Specificity':[]})
        elif target_type =='nonbinary':
            metrics_calculator = []
            testing_metrics_df = pd.DataFrame(data = {'MSE':[]})
        
        # Setting up iterator to generate images from the validation dataset
        data_iterator = iter(test_dataloader)
        for i in tqdm(range(0,len(dataset_valid)),desc = 'Testing'):
            
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

            if output_type=='binary':        
                target_img = target.cpu().numpy().round()
            else:
                target_img = target.cpu().numpy()

            pred_mask = model.predict(image.to(device))

            #print(f'pred_mask size: {np.shape(pred_mask.detach().cpu().numpy())}')
            #print(f'target size: {np.shape(target.cpu().numpy())}')

            testing_metrics_df = testing_metrics_df.append(pd.DataFrame(get_metrics(pred_mask.detach().cpu(),target.cpu(), input_name, metrics_calculator,target_type)),ignore_index=True)
            
            if output_type == 'binary':
                pred_mask_img = pred_mask.detach().cpu().numpy().round()
            else:
                pred_mask_img = pred_mask.detach().cpu().numpy()

            img_dict = {'Image':image.cpu().numpy(),'Pred_Mask':pred_mask_img,'Ground_Truth':target_img}
            
            if target_type=='binary':
                fig = visualize(img_dict,output_type)
            elif target_type=='nonbinary':
                fig = visualize_continuous(img_dict,output_type)       

            fig.savefig(test_output_dir+'Test_Example_'+input_name)
            nept_run['testing/Testing_Output_'+input_name].upload(test_output_dir+'Test_Example_'+input_name)

        # Used during hyperparameter optimization to compute objective value
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


