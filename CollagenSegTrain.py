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
from PIL import Image

import pandas as pd
from tqdm import tqdm
import sys

from CollagenSegUtils import visualize_continuous, visualize_multi_task

from MultiTaskModel import MultiTaskLoss, MultiTaskModel


class Custom_MSE_Loss(torch.nn.Module):
    def __init__(self):
        super(Custom_MSE_Loss,self).__init__()


    def forward(self,output,target):
        diff = (output-target)**2
        normed = (diff - torch.min(diff))/torch.max(diff)
        meaned = torch.mean(normed)
        return meaned

class Custom_MSE_LossPlus(torch.nn.Module):
    def __init__(self):
        super(Custom_MSE_LossPlus,self).__init__()
    
    def forward(self,output,target):
        diff = (output-target)**2
        mean_square = torch.mean(diff)
        normed = (diff-torch.min(diff))/torch.max(diff)
        meaned = torch.mean(normed)
        return mean_square+meaned
    

def Training_Loop(ann_classes, dataset_train, dataset_valid, model_dir, output_dir,target_type, train_parameters, nept_run):
    
    encoder = train_parameters['encoder']
    encoder_weights = train_parameters['encoder_weights']

    output_type = train_parameters['output_type']
    active = train_parameters['active']

    if train_parameters['in_channels']==3:
        in_channels = 3
    else:
        in_channels = 1

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
        elif train_parameters['loss'] == 'custom+':
            loss = Custom_MSE_LossPlus()

        n_classes = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Is training on GPU available? : {torch.cuda.is_available()}')
    print(f'Device is : {device}')
    print(f'Torch Cuda version is : {torch.version.cuda}')    

    nept_run['encoder'] = encoder
    nept_run['encoder_pre_train'] = encoder_weights
    nept_run['Architecture'] = train_parameters['architecture']
    nept_run['Loss'] = train_parameters['loss']
    nept_run['output_type'] = output_type
    nept_run['lr'] = train_parameters['lr']
    nept_run['multi_task'] = train_parameters['multi_task']
    

    if train_parameters['architecture']=='Unet++':
        model = smp.UnetPlusPlus(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = in_channels,
                classes = n_classes,
                activation = active
                )
    elif train_parameters['architecture'] == 'Unet':
        model = smp.Unet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = n_classes,
            activation = active
        )
    elif train_parameters['architecture'] == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = n_classes,
            activation = active
        )
    elif train_parameters['architecture'] == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = n_classes,
            activation = active
        )
    elif train_parameters['architecture'] == 'MAnet':
        model = smp.MAnet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = in_channels,
            classes = n_classes,
            activation = active
        )
    

    if train_parameters['multi_task']:

        model = MultiTaskModel({'unet_model':model})
        loss = MultiTaskLoss()


    # Sending model to current device ('cuda','cuda:0','cuda:1',or 'cpu')
    model = model.to(device)
    loss = loss.to(device)
    
    optimizer = torch.optim.Adam([
            dict(params = model.parameters(), lr = train_parameters['lr'],weight_decay = 0.001)
            ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 35)

    batch_size = train_parameters['batch_size']
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 12)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = 4)
        
    # Initializing loss score
    train_loss = 0
    val_loss = 0
    
    # Maximum number of epochs defined here as well as how many steps between model saves and example outputs
    epoch_num = train_parameters['epoch_num']
    save_step = train_parameters['save_step']

    # Recording training and validation loss
    train_loss_list = []
    val_loss_list = []
    
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

            # Calculating loss
            if train_parameters['multi_task']:
                bin_loss, reg_loss = loss(train_preds,train_masks)
                nept_run['training_bin_loss'].log(bin_loss.item())
                nept_run['training_reg_loss'].log(reg_loss.item())

                train_loss = bin_loss+reg_loss

            else:
                train_loss = loss(train_preds,train_masks)


            # Printing max and min for training GTs and predictions
            #print(f'Min prediction: {np.min(train_preds.detach().cpu().numpy())}, Max prediction: {np.max(train_preds.detach().cpu().numpy())}')
            #print(f'Min GT: {np.min(train_masks.cpu().numpy())}, Max GT: {np.max(train_masks.cpu().numpy())}')
            
            # Backpropagation
            train_loss.backward()
            train_loss = train_loss.item()

            train_loss_list.append(train_loss)

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

                if train_parameters['multi_task']:
                    val_bin_loss, val_reg_loss = loss(val_preds,val_masks)
                    nept_run['validation_bin_loss'].log(val_bin_loss.item())
                    nept_run['validation_reg_loss'].log(val_reg_loss.item())

                    val_loss = val_bin_loss+val_reg_loss
                else:
                    val_loss = loss(val_preds,val_masks)

                val_loss = val_loss.item()

                val_loss_list.append(val_loss)

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

                if train_parameters['multi_task']:
                    fig = visualize_multi_task(img_dict,output_type)
                else:
                    fig = visualize_continuous(img_dict,output_type)

                # Different process for saving comparison figures vs. only predictions
                if output_type == 'comparison':
                    fig.savefig(output_dir+f'Training_Epoch_{i}_Example.png')
                    nept_run[f'Example_Output_{i}'].upload(output_dir+f'Training_Epoch_{i}_Example.png')
                elif output_type == 'prediction':
                    if train_parameters['multi_task']:
                        coll_im = Image.fromarray(fig[0].astype(np.uint8))
                        coll_im.save(output_dir+f'Training_Epoch_{i}_collagen.tif')
                        nept_run[f'Example_Output_{i}_collagen'].upload(output_dir+f'Training_Epoch_{i}_collagen.tif')

                        bin_im = Image.fromarray(fig[1].astype(np.uint8))
                        bin_im.save(output_dir+f'Training_Epoch_{i}_binary.tif')
                        nept_run[f'Example_Output_{i}_binary'].upload(output_dir+f'Training_Epoch_{i}_binary.tif')
                    else:
                        im = Image.fromarray(fig.astype(np.uint8))
                        im.save(output_dir+f'Training_Epoch_{i}_Example.tif')
                        nept_run[f'Example_Output_{i}'].upload(output_dir+f'Training_Epoch_{i}_Example.tif')

                #fig.savefig(output_dir+f'Training_Epoch_{i}_Example.png')
                #nept_run[f'Example_Output_{i}'].upload(output_dir+f'Training_Epoch_{i}_Example.png')

    if not i%save_step==0:
        torch.save(model.state_dict(),model_dir+f'Collagen_Seg_Model_{i}.pth')

    loss_df = pd.DataFrame(data = {'TrainingLoss':train_loss_list,'ValidationLoss':val_loss_list})
    loss_df.to_csv(output_dir+'Training_Validation_Loss.csv')
    return model_dir+f'Collagen_Seg_Model_{i}.pth'



