# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:37:56 2021

@author: spborder


DGCS Training Loop 

from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb

A lot of different segmentation networks are available from the 'segmentation_models_pytorch' module
"""

import torch
# from apex import amp
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
from tqdm import tqdm
import sys
import os

from CollagenSegUtils import visualize_continuous
from CollagenModels import EnsembleModel, EnsembleModelMIT, AttentionModel
    
def Training_Loop(dataset_train, dataset_valid, train_parameters, nept_run):
    
    model_details = train_parameters['model_details']

    if not model_details['architecture'] == 'DUnet':
        encoder = model_details['encoder']
        encoder_weights = model_details['encoder_weights']
        dropout_rate = model_details['dropout_rate']
        nept_run['encoder'] = encoder
        nept_run['encoder_pre_train'] = encoder_weights

    output_type = 'comparison'
    active = model_details['active']
    target_type = model_details['target_type']
    ann_classes = model_details['ann_classes'].split(',')
    output_dir = train_parameters['output_dir']
    model_dir = output_dir+'/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if active=='None':
        active = None

    in_channels = model_details['in_channels']

    if target_type=='binary':
        loss = smp.losses.DiceLoss(mode='binary')
        #n_classes = len(ann_classes)
        n_classes = 1

    elif target_type=='nonbinary':
        if train_parameters['loss']=='MSE':
            loss = torch.nn.MSELoss(reduction='mean')
        elif train_parameters['loss'] == 'L1':
            loss = torch.nn.L1Loss(reduction='mean')
        elif train_parameters['loss'] == 'BCE':
            loss = torch.nn.BCELoss()
        n_classes = 1


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Is training on GPU available? : {torch.cuda.is_available()}')
    print(f'Device is : {device}')
    print(f'Torch Cuda version is : {torch.version.cuda}')    

    nept_run['Architecture'] = model_details['architecture']
    nept_run['Loss'] = train_parameters['loss']
    nept_run['output_type'] = output_type
    nept_run['lr'] = train_parameters['lr']
    
    if model_details['architecture']=='Unet++':
        model = smp.UnetPlusPlus(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = in_channels,
                classes = n_classes,
                activation = active
                )
    elif model_details['architecture']=='ensemble':
        model = EnsembleModel(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            active = active,
            n_classes = n_classes
            )
    elif model_details['architecture']=='ensembleMIT':
        model = EnsembleModelMIT(            
            in_channels = in_channels,
            active = active,
            n_classes = n_classes
            )
    elif model_details['architecture'] == 'attention':
        model = AttentionModel(
            in_channels = in_channels,
            active = active,
            n_classes = n_classes
            )
        
    optimizer = torch.optim.Adam([
            dict(params = model.parameters(), lr = train_parameters['lr'],weight_decay = 0.0001)
            ])
    
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=250)

    # Sending model to current device ('cuda','cuda:0','cuda:1',or 'cpu')
    model = model.to(device)
    loss = loss.to(device)
    
    # Wrap model and optimizer with Amp for mixed precision training
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O3")

    batch_size = train_parameters['batch_size']
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    train_iter = iter(train_loader)
    val_iter = iter(valid_loader)      
    
    # # Save one batch of data from the train and validation loader to test
    # train_batch = next(train_iter)
    # valid_batch = next(val_iter)

    # # Save the batches as a tuple
    # torch.save({
    #     'train_batch': train_batch,
    #     'valid_batch': valid_batch
    # }, "train_val_loader_test.pt")
    # print("L132==============", len(train_loader), len(valid_loader))      

    # Maximum number of epochs defined here as well as how many steps between model saves and example outputs
    epoch_num = train_parameters['step_num']
    save_step = train_parameters['save_step']

    train_loss = 0
    val_loss = 0

    # Recording training and validation loss
    train_loss_list = []
    val_loss_list = []
    
    with tqdm(total = epoch_num, position = 0, leave = True, file = sys.stdout) as pbar:
        # steps = 0
        for i in range(0,epoch_num):            
            # Turning on dropout
            model.train()

            # Controlling the progress bar, printing training and validation losses
            if i==1: 
                pbar.set_description(f'Step: {i}/{epoch_num}')
                pbar.update(i)
            elif i%5==0:
                pbar.set_description(f'Step: {i}/{epoch_num}, Train/Val Loss: {round(train_loss,4)},{round(val_loss,4)}')
                pbar.update(5)
        
            # Clear existing gradients in optimizer
            optimizer.zero_grad()

            # Loading training and validation samples from dataloaders
            # train_imgs, train_masks, _ = next(train_loader)
            try:
                train_imgs, train_masks, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_imgs, train_masks, _ = next(train_iter)
            
            # print(f"L271-- train_imgs.dtype: {train_imgs.dtype}\ttrain_masks.dtype: {train_masks.dtype}")
            # Sending to device
            train_imgs = train_imgs.to(device)
            train_masks = train_masks.to(device)


            # Forward pass
            # with amp.autocast():
            # Running predictions on training batch
            train_preds = model(train_imgs)

            # Calculating loss
            train_loss = loss(train_preds,train_masks)

            # Backpropagation
            train_loss.backward()
            train_loss = train_loss.item()

            train_loss_list.append(train_loss)

            if not 'current_k_fold' in train_parameters:
                nept_run['training_loss'].log(train_loss)
            else:
                nept_run[f'training_loss_{train_parameters["current_k_fold"]}'].log(train_loss)

            # Backward pass
            # with amp.scale_loss(train_loss, optimizer) as scaled_loss:
            #     # Updating optimizer
            #     scaled_loss.backward()
            # Updating optimizer
            optimizer.step()

            # Validation (don't want it to influence gradients in network)
            with torch.no_grad():
                # This turns off any dropout in the network 
                model.eval()

                # val_imgs, val_masks, _ = next(iter(valid_loader))
                try:
                    val_imgs, val_masks, _ = next(val_iter)
                except StopIteration:                        
                    val_iter = iter(valid_loader)
                    val_imgs, val_masks, _ = next(val_iter)
                    
                val_imgs = val_imgs.to(device)
                val_masks = val_masks.to(device)

                # Predicting on the validation images
                val_preds = model(val_imgs)
                # Finding validation loss
                val_loss = loss(val_preds,val_masks)

                val_loss = val_loss.item()
                val_loss_list.append(val_loss)

                if not 'current_k_fold' in train_parameters:
                    nept_run['validation_loss'].log(val_loss)
                else:
                    nept_run[f'validation_loss_{train_parameters["current_k_fold"]}'].log(val_loss)

            # Stepping the learning rate plateau with the current validation loss
            #scheduler.step()
            lr_plateau.step(val_loss)

            # Saving model if current i is a multiple of "save_step"
            # Also generating example output segmentation and uploading that to Neptune
            if i%save_step == 0:
                torch.save(model.state_dict(),model_dir+f'Collagen_Seg_Model_Latest.pth')

                if batch_size==1:
                    current_img = val_imgs.cpu().numpy()
                    current_gt = val_masks.cpu().numpy()
                    current_pred = val_preds.cpu().numpy()
                else:
                    current_img = val_imgs[0].cpu().numpy()
                    current_gt = val_masks[0].cpu().numpy()
                    current_pred = val_preds[0].cpu().numpy()

                """
                if target_type=='binary':
                    current_pred = current_pred.round()
                """
                
                if type(in_channels)==int:
                    if in_channels == 6:
                        current_img = np.concatenate((current_img[0:3,:,:],current_img[3:6,:,:]),axis=-2)
                    elif in_channels==4:
                        current_img = np.concatenate((np.stack((current_img[0,:,:],)*3,axis=1),current_img[0:3,:,:]),axis=-2)
                    elif in_channels==2:
                        current_img = np.concatenate((current_img[0,:,:],current_img[1,:,:]),axis=-2)
                elif type(in_channels)==list:
                    if sum(in_channels)==6:
                        current_img = np.concatenate((current_img[0:3,:,:],current_img[3:6,:,:]),axis=-2)
                    elif sum(in_channels)==2:
                        current_img = np.concatenate((current_img[0,:,:][None,:,:],current_img[1,:,:][None,:,:]),axis=-2)

                img_dict = {'Image':current_img, 'Pred_Mask':current_pred,'Ground_Truth':current_gt}

                fig = visualize_continuous(img_dict,output_type)

                # Different process for saving comparison figures vs. only predictions
                if output_type == 'comparison':
                    fig.savefig(output_dir+f'/Training_Epoch_{i}_Example.png')
                    nept_run[f'Example_Output_{i}'].upload(output_dir+f'/Training_Epoch_{i}_Example.png')
                elif output_type == 'prediction':

                    im = Image.fromarray(fig.astype(np.uint8))
                    im.save(output_dir+f'/Training_Epoch_{i}_Example.tif')
                    nept_run[f'Example_Output_{i}'].upload(output_dir+f'/Training_Epoch_{i}_Example.tif')

    if not i%save_step==0:
        torch.save(model.state_dict(),model_dir+f'Collagen_Seg_Model_Latest.pth')

    loss_df = pd.DataFrame(data = {'TrainingLoss':train_loss_list,'ValidationLoss':val_loss_list})
    loss_df.to_csv(output_dir+'Training_Validation_Loss.csv')
    return model_dir+f'Collagen_Seg_Model_Latest.pth'



