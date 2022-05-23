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


def back_to_reality(tar):
    
    # Getting target array into right format
    classes = np.shape(tar)[-1]
    dummy = np.zeros((np.shape(tar)[0],np.shape(tar)[1]))
    for value in range(classes):
        mask = np.where(tar[:,:,value]!=0)
        dummy[mask] = value

    return dummy

def visualize(images):
    
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
        
        if np.shape(img)[-1]!=3:
            img = back_to_reality(img)
            
        plt.imshow(img)
    #plt.show()

    return plt.gcf()

def get_metrics(pred_mask,ground_truth,n_labels):

    metrics_names = ['IoU','F1 Score','Accuracy','Precision','Recall']
    metrics_row = {}

    if n_labels>2: 
        tp,fp,fn,tn = smp.metrics.get_stats(pred_mask,ground_truth, mode = 'multilabel',threshold = 1/n_labels)
    else:
        tp,fp,fn,tn = smp.metrics.get_stats(pred_mask,ground_truth,mode = 'binary')

    metrics_row['IoU'] = smp.metrics.iou_score(tp,fp,fn,tn)
    metrics_row['F1 Score'] = smp.metrics.f1_score(tp,fp,fn,tn)
    metrics_row['Accuracy'] = smp.metrics.accuracy(tp,fp,fn,tn)
    metrics_row['Precision'] = smp.metrics.precision(tp,fp,fn,tn)
    metrics_row['Recall'] = smp.metrics.recall(tp,fp,fn,tn)
    

    return metrics_row
    
        
def Test_Network(classes, model_dir, dataset_valid, output_dir, nept_run):
     

    # Finding the best performing model
    models = glob(model_dir+'*.pth')
    model_eps = [i.split('_')[-1].replace('.pth','') for i in models]
    model_eps = [int(i) for i in model_eps]
    
    best_ep = np.argmax(model_eps)
    
    best_model = torch.load(models[best_ep])
    
    # Evaluating model on test set
    test_dataloader = DataLoader(dataset_valid)
    """
    test_epoch = smp.utils.train.ValidEpoch(
            model = best_model,
            loss = smp.utils.losses.DiceLoss(),
            metrics = [smp.utils.metrics.IoU(threshold = 1/len(classes))],
            device = 'cuda')
    
    
    test_logs = test_epoch.run(test_dataloader)
    """
    test_output_dir = output_dir+'Testing_Output/'
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
        
    testing_metrics_df = pd.DataFrame(data = {'IoU':[],'Accuracy':[],'FScore':[],'Recall':[],'Precision':[]})
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
        pred_mask = best_model.predict(image.to('cuda'))

        testing_metrics_df.append(get_metrics(pred_mask,target,classes))
        pred_mask_img = pred_mask.detach().cpu().numpy().round()
        
        img_dict = {'Image':image.cpu().numpy(),'Pred_Mask':pred_mask_img,'Ground_Truth':target_img}
        
        fig = visualize(img_dict)
        
        fig.savefig(test_output_dir+'Test_Example_'+str(i)+'.png')

    nept_run['Test_Image_metrics'].upload(neptune.types.File.as_html(testing_metrics_df))

    for met in testing_metrics_df.columns.values.tolist():
        nept_run[met] = testing_metrics_df[met].mean()



