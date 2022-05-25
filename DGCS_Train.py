# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:37:56 2021

@author: spborder


DGCS Training Loop 

from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

A lot of different segmentation networks are available from the 'segmentation_models_pytorch' module
"""

import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd


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

    return plt.gcf()
    
    
        
        
def Training_Loop(ann_classes, dataset_train, dataset_valid, model_dir, output_dir,target_type,nept_run):
    
    encoder = 'resnet34'
    encoder_weights = 'imagenet'
    

    if target_type=='binary':
        active = 'softmax2d'
        loss = smp.utils.losses.DiceLoss()

        metrics = [smp.utils.metrics.IoU(threshold = 1/len(ann_classes))]
    else:
        active = 'relu2d'
        loss = 

    device = 'cuda:1'
    
    nept_run['encoder'] = encoder
    nept_run['encoder_pre_train'] = encoder_weights
    nept_run['Architecture'] = 'UNet'
    nept_run['Loss'] = 'Dice'
    nept_run['Metrics'] = 'IoU'
    
    model = smp.Unet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = len(ann_classes),
            activation = active
            )
    

    
    metrics = [smp.utils.metrics.IoU(threshold = 1/len(ann_classes))]
    
    optimizer = torch.optim.Adam([
            dict(params = model.parameters(), lr = 0.0001)
            ])
    
    train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True)
    
    valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=True)
    
    train_loader = DataLoader(dataset_train, batch_size = 1, shuffle = True, num_workers = 12)
    valid_loader = DataLoader(dataset_valid, batch_size = 1, shuffle = True, num_workers = 4)
        
    # To minimize loss
    max_score = 0
    
    # Max # of epochs = 40
    epoch_num = 80
    save_step = 20
    
    for i in range(0,epoch_num):
        
        print('\nEpoch: {}'.format(i))
        
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # Saving loss and metrics to dataframe
        if i==0:
            train_df = pd.DataFrame(train_logs, index = [0])
            valid_df = pd.DataFrame(valid_logs, index = [0])
            
            #valid_df.to_csv(output_dir+'Validation_Loss.csv')
            #train_df.to_csv(output_dir+'Training_Loss.csv')
        else:
            train_df = train_df.append(train_logs, ignore_index = True)
            valid_df = valid_df.append(valid_logs, ignore_index = True)
            
            #valid_df.to_csv(output_dir+'Validation_Loss.csv')
            #train_df.to_csv(output_dir+'Training_Loss.csv')
        
        # Saving model if the loss went down
        if max_score< valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model,model_dir+'Collagen_Seg_Model_'+str(i)+'.pth')
            
        if i%save_step == 0:
            
            image,target = next(iter(valid_loader))
            
            target = target.cpu().numpy().round()
            """
            unique_vals = np.unique(target)
            dummy = np.zeros_like(target)
            # For each class, replace region in target (labeled image) with a unique integer
            # from 0 to the number of classes
            for idx, value in enumerate(unique_vals):
                mask = np.where(target==value)
                dummy[mask] = idx
            """
            
            pred_mask = model(image.to(device))
            pred_mask = pred_mask.detach().cpu().numpy().round()
            img_dict = {'Image':image.cpu().numpy(),'Pred_Mask':pred_mask,'Ground_Truth':target}
            
            fig = visualize(img_dict)
            
            fig.savefig(output_dir+'Training_Epoch_'+str(i)+'_Example.png')
            
            nept_run['Example_Output_'+str(i)].upload(output_dir+'Training_Epoch_'+str(i)+'_Example.png')

            # Combining training and validation dataframes
            col_names = train_df.columns.values.tolist()
            val_names = ['val_'+i for i in col_names]
            
            new_valid = valid_df.copy()
            new_valid.columns = val_names
            
            combined_df = pd.concat([train_df, new_valid], axis = 1, ignore_index = True)
            combined_df.columns = col_names+val_names
            combined_df['Epoch_Number'] = list(range(0,combined_df.shape[0]))
            combined_df = combined_df.set_index('Epoch_Number')
            
            combined_df.to_csv(output_dir+'Training_Validation_Loss.csv')
            
            # Saving loss figure
            ax = combined_df.plot()
            fig = ax.get_figure()
            fig.savefig(output_dir+'Training_Loss_Metrics_Plot.png')

            nept_run['Loss_Plot'].upload(output_dir+'Training_Loss_Metrics_Plot.png')





