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
from tqdm import tqdm
import sys

from CollagenSegUtils import visualize_continuous


class Custom_MSE_Loss(torch.nn.Module):
    def __init__(self):
        super(Custom_MSE_Loss,self).__init__()


    def forward(self,output,target):
        diff = (output-target)**2
        normed = (diff - torch.min(diff))/torch.max(diff)
        meaned = torch.mean(normed)
        return meaned
    

def Training_Loop(ann_classes, dataset_train, dataset_valid, model_dir, output_dir,target_type, train_parameters, nept_run):
    
    encoder = train_parameters['encoder']
    encoder_weights = train_parameters['encoder_weights']

    output_type = train_parameters['output_type']
    active = train_parameters['active']

    if target_type=='binary':
        loss = smp.losses.DiceLoss(mode='binary')
        n_classes = len(ann_classes)

    elif target_type=='nonbinary':
        if train_parameters['loss']=='MSE':
            loss = torch.nn.MSELoss(reduction='mean')
        elif train_parameters['loss'] == 'L1':
            loss = torch.nn.L1Loss()
        elif train_parameters['loss'] == 'custom':
            loss = Custom_MSE_Loss()
        n_classes = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Is training on GPU available? : {torch.cuda.is_available()}')
    print(f'Device is : {device}')
    print(f'Torch Cuda version is : {torch.version.cuda}')    

    nept_run['encoder'] = encoder
    nept_run['encoder_pre_train'] = encoder_weights
    nept_run['Architecture'] = 'Unet++'
    nept_run['Loss'] = 'MSE'
    nept_run['output_type'] = output_type
    

    if train_parameters['architecture']=='Unet++':
        model = smp.UnetPlusPlus(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = 3,
                classes = n_classes,
                activation = active
                )
    elif train_parameters['architecture'] == 'Unet':
        model = smp.Unet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = n_classes,
            activation = active
        )
    elif train_parameters['architecture'] == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = n_classes,
            activation = active
        )
    elif train_parameters['architecture'] == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = n_classes,
            activation = active
        )
    elif train_parameters['architecture'] == 'MAnet':
        model = smp.MAnet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = 3,
            classes = n_classes,
            activation = active
        )
    
    # Sending model to current device ('cuda','cuda:0','cuda:1',or 'cpu')
    model = model.to(device)
    loss = loss.to(device)
    
    optimizer = torch.optim.Adam([
            dict(params = model.parameters(), lr = train_parameters['lr'],weight_decay = 0.01)
            ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 35)

    # Not sure if this is necessary or if torch.optim.Adam has a .to() method
    #optimizer = optimizer.to(device)

    batch_size = train_parameters['batch_size']
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 12)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = 4)
        
    # Initializing loss score
    train_loss = 0
    val_loss = 0
    
    # Maximum number of epochs defined here as well as how many steps between model saves and example outputs
    epoch_num = train_parameters['epoch_num']
    save_step = train_parameters['save_step']
    
    with tqdm(total = epoch_num, position = 0, leave = True, file = sys.stdout) as pbar:

        for i in range(0,epoch_num):
        
            # Turning on dropout
            model.train()

            # Controlling the progress bar, printing training and validation losses
            if i==1: 
                pbar.set_description(f'Epoch: {i}/{epoch_num}')
                pbar.update(i)
            elif i%5==0:
                pbar.set_description(f'Epoch: {i}/{epoch_num}, Train/Val Loss: {round(train_loss,4)},{round(val_loss,4)}')
                pbar.update(5)
        
            # Clear existing gradients in optimizer
            optimizer.zero_grad()

            # Loading training and validation samples from dataloaders
            train_imgs, train_masks, _ = next(iter(train_loader))
            # Sending to device
            train_imgs = train_imgs.to(device)
            train_masks = train_masks.to(device)

            # Testing whether the images were actually sent to cuda
            #print(f'Images are on device: {train_imgs.get_device()}')

            # Running predictions on training batch
            train_preds = model(train_imgs)

            """
            if target_type=='nonbinary':
                train_preds = torch.squeeze(train_preds)
                train_masks = torch.squeeze(train_masks)
            """

            # Calculating loss
            train_loss = loss(train_preds,train_masks)

            # Printing max and min for training GTs and predictions
            #print(f'Min prediction: {np.min(train_preds.detach().cpu().numpy())}, Max prediction: {np.max(train_preds.detach().cpu().numpy())}')
            #print(f'Min GT: {np.min(train_masks.cpu().numpy())}, Max GT: {np.max(train_masks.cpu().numpy())}')
            
            # Backpropagation
            train_loss.backward()
            train_loss = train_loss.item()

            if not 'current_k_fold' in train_parameters:
                nept_run['training_loss'].log(train_loss)
            else:
                nept_run[f'training_loss_{train_parameters["current_k_fold"]}'].log(train_loss)

            # Updating optimizer
            optimizer.step()
            #print(f'Model device: {next(model.parameters()).device}')

            # Validation (don't want it to influence gradients in network)
            with torch.no_grad():
                # This turns off any dropout in the network 
                model.eval()

                val_imgs, val_masks, _ = next(iter(valid_loader))
                val_imgs = val_imgs.to(device)
                val_masks = val_masks.to(device)

                val_preds = model(val_imgs)
                val_loss = loss(val_preds,val_masks)
                val_loss = val_loss.item()

                if not 'current_k_fold' in train_parameters:
                    nept_run['validation_loss'].log(val_loss)
                else:
                    nept_run[f'validation_loss_{train_parameters["current_k_fold"]}'].log(val_loss)

            # Updating learning rate scheduler
            scheduler.step(val_loss)

            # Saving model if current i is a multiple of "save_step"
            # Also generating example output segmentation and uploading that to Neptune
            if i%save_step == 0:
                torch.save(model.state_dict(),model_dir+f'Collagen_Seg_Model_{i}.pth')

                if batch_size==1:
                    current_img = val_imgs.cpu().numpy()
                    current_gt = val_masks.cpu().numpy()
                    current_pred = val_preds.cpu().numpy()
                else:
                    current_img = val_imgs[0].cpu().numpy()
                    current_gt = val_masks[0].cpu().numpy()
                    current_pred = val_preds[0].cpu().numpy()

                if target_type=='binary':
                    current_pred = current_pred.round()

                img_dict = {'Image':current_img, 'Pred_Mask':current_pred,'Ground_Truth':current_gt}

                """
                if target_type=='binary':
                    fig = visualize(img_dict,output_type)
                elif target_type=='nonbinary':
                    fig = visualize_continuous(img_dict)
                """

                fig = visualize_continuous(img_dict,output_type)

                # Different process for saving comparison figures vs. only predictions
                if output_type

                fig.savefig(output_dir+f'Training_Epoch_{i}_Example.png')
                nept_run[f'Example_Output_{i}'].upload(output_dir+f'Training_Epoch_{i}_Example.png')

    if not i%save_step==0:
        torch.save(model.state_dict(),model_dir+f'Collagen_Seg_Model_{i}.pth')

    return model_dir+f'Collagen_Seg_Model_{i}.pth'



