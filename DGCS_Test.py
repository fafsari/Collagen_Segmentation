# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:28:08 2021

@author: spborder


DGCS Testing best model

from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb


"""

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

def visualize_continuous(images):

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

        img = np.float32(np.moveaxis(img,source=0,destination=-1))
        if key == 'Pred_Mask' or key == 'Ground_Truth':
            img = apply_colormap(img)

            plt.imshow(img,cmap='jet')
        else:
            plt.imshow(img)

    return plt.gcf()

def get_metrics(pred_mask,ground_truth,calculator):

    metrics_row = {}

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

    return metrics_row
    
        
def Test_Network(classes, model_path, dataset_valid, output_dir, nept_run,target_type):
     

    # Finding the best performing model
    #models = glob(model_dir+'*.pth')
    #model_eps = [i.split('_')[-1].replace('.pth','') for i in models]
    #model_eps = [int(i) for i in model_eps]
    
    #best_ep = np.argmax(model_eps)
    
    #best_model = torch.load(models[best_ep])
    
    #device = 'cuda:1'
    #best_model.to(device)
    encoder = 'resnet34'
    encoder_weights = 'imagenet'
    ann_classes = classes
    active = None

    if target_type=='binary':
        n_classes = len(ann_classes)
    elif target_type == 'nonbinary':
        n_classes = 1

    output_type = 'continuous'

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    model = smp.UnetPlusPlus(
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
            
        metrics_calculator = BinaryMetrics()
        testing_metrics_df = pd.DataFrame(data = {'Dice':[],'Accuracy':[],'Recall':[],'Precision':[],'Specificity':[]})
        # Setting up iterator to generate images from the validation dataset
        data_iterator = iter(test_dataloader)
        for i in tqdm(range(0,len(dataset_valid)),desc = 'Testing'):
            
            try:
                image, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(test_dataloader)
                image, target = next(data_iterator)
                
            # Add something here so that it calculates perforance metrics and outputs
            # raw values for 2-class segmentation(not binarized output masks)        
            target_img = target.cpu().numpy().round()
            pred_mask = model.predict(image.to(device))

            #print(f'pred_mask size: {np.shape(pred_mask.detach().cpu().numpy())}')
            #print(f'target size: {np.shape(target.cpu().numpy())}')

            testing_metrics_df = testing_metrics_df.append(pd.DataFrame(get_metrics(pred_mask.detach().cpu(),target.cpu(), metrics_calculator)),ignore_index=True)
            
            pred_mask_img = pred_mask.detach().cpu().numpy().round()
            
            img_dict = {'Image':image.cpu().numpy(),'Pred_Mask':pred_mask_img,'Ground_Truth':target_img}
            
            if target_type=='binary':
                fig = visualize(img_dict,output_type)
            elif target_type=='nonbinary':
                fig = visualize_continuous(img_dict)       

            fig.savefig(test_output_dir+'Test_Example_'+str(i)+'.png')
            nept_run['testing/Testing_Output_'+str(i)].upload(test_output_dir+'Test_Example_'+str(i)+'.png')


        nept_run['Test_Image_metrics'].upload(neptune.types.File.as_html(testing_metrics_df))

        for met in testing_metrics_df.columns.values.tolist():
            print(f'{met} value: {testing_metrics_df[met].mean()}')
            nept_run[met] = testing_metrics_df[met].mean()



