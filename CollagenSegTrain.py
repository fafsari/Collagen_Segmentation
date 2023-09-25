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

from CollagenSegUtils import visualize_continuous


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

class Custom_Plus_Plus_Loss(torch.nn.Module):
    def __init__(self):
        super(Custom_Plus_Plus_Loss,self).__init__()

        self.MSE_Loss = torch.nn.MSELoss(reduction='mean')

    def dice_loss(self, output, target):
        numerator = (torch.round(output)*torch.round(target)).sum()
        denominator = torch.round(output).sum() + torch.round(target).sum()
        final = 1-(2*numerator)/(denominator+1e-7)

        return final

    def forward(self,output, target):

        # MSE portion
        mse_loss = self.MSE_Loss(output,target)

        # Binary portion
        bin_loss = self.dice_loss(output,target)

        return mse_loss, bin_loss
    

def Training_Loop(dataset_train, dataset_valid, train_parameters, nept_run):
    
    if not train_parameters['architecture'] == 'DUnet':
        encoder = train_parameters['encoder']
        encoder_weights = train_parameters['encoder_weights']
        nept_run['encoder'] = encoder
        nept_run['encoder_pre_train'] = encoder_weights

    output_type = train_parameters['output_type']
    active = train_parameters['active']
    target_type = train_parameters['target_type']
    ann_classes = train_parameters['target_type']
    output_dir = train_parameters['output_dir']
    model_dir = output_dir+'/models/'

    if active=='None':
        active = None

    in_channels = train_parameters['in_channels']

    if target_type=='binary':
        loss = smp.losses.DiceLoss(mode='binary')
        n_classes = len(ann_classes)

    elif target_type=='nonbinary':
        if train_parameters['loss']=='MSE':
            loss = torch.nn.MSELoss(reduction='mean')
        elif train_parameters['loss'] == 'L1':
            loss = torch.nn.L1Loss(reduction='mean')
        elif train_parameters['loss'] == 'BCE':
            loss = torch.nn.BCELoss()
        elif train_parameters['loss'] == 'custom':
            loss = Custom_MSE_Loss()
        elif train_parameters['loss'] == 'custom+':
            loss = Custom_MSE_LossPlus()
        elif train_parameters['loss'] == 'custom++':
            loss = Custom_Plus_Plus_Loss()

        n_classes = 1


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Is training on GPU available? : {torch.cuda.is_available()}')
    print(f'Device is : {device}')
    print(f'Torch Cuda version is : {torch.version.cuda}')    


    nept_run['Architecture'] = train_parameters['architecture']
    nept_run['Loss'] = train_parameters['loss']
    nept_run['output_type'] = output_type
    nept_run['lr'] = train_parameters['lr']
    
    if train_parameters['architecture']=='Unet++':
        model = smp.UnetPlusPlus(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = in_channels,
                classes = n_classes,
                activation = active
                )
    elif train_parameters['architecture']=='DUnet':
        model = DUNet(
            n_channels = in_channels,
            n_classes = n_classes,
            activation = active
        )

    optimizer = torch.optim.Adam([
            dict(params = model.parameters(), lr = train_parameters['lr'],weight_decay = 0.0001)
            ])
    
    """
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                cycle_momentum = False,
                                                base_lr = train_parameters['lr']/3,
                                                max_lr = train_parameters['lr']*3,
                                                step_size_up = 250)
    """
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=250,verbose=True)

    # Sending model to current device ('cuda','cuda:0','cuda:1',or 'cpu')
    model = model.to(device)
    loss = loss.to(device)

    batch_size = train_parameters['batch_size']
    #train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 12)
    #valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = 4)

    if 'sub_categories_file' in train_parameters:
        sub_categories = pd.read_csv(train_parameters['sub_categories_file'])
        dataset_train.add_sub_categories(sub_categories,'Labels')
        train_loader = iter(dataset_train)
    else:
        train_loader = DataLoader(dataset_valid,batch_size=batch_size,shuffle=True)

    valid_loader = DataLoader(dataset_valid,batch_size=batch_size,shuffle=True)
        
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
            if 'sub_categories_file' not in train_parameters:
                train_imgs, train_masks, _ = next(iter(train_loader))
            else:
                train_imgs, train_masks, _ = next(train_loader)
            # Sending to device
            train_imgs = train_imgs.to(device)
            train_masks = train_masks.to(device)

            # Running predictions on training batch
            train_preds = model(train_imgs)

            # Calculating loss
            if train_parameters['loss']=='custom++':
                mse_loss,bin_loss = loss(train_preds,train_masks)
                nept_run['training_reg_loss'].log(mse_loss.item())
                nept_run['training_bin_loss'].log(bin_loss.item())

                train_loss = mse_loss+bin_loss
            else:
                train_loss = loss(train_preds,train_masks)

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

            # Validation (don't want it to influence gradients in network)
            with torch.no_grad():
                # This turns off any dropout in the network 
                model.eval()

                val_imgs, val_masks, _ = next(iter(valid_loader))
                val_imgs = val_imgs.to(device)
                val_masks = val_masks.to(device)

                val_preds = model(val_imgs)

                if train_parameters['loss'] == 'custom++':
                    val_mse_loss, val_bin_loss = loss(val_preds,val_masks)
                    nept_run['validation_bin_loss'].log(val_bin_loss.item())
                    nept_run['validation_reg_loss'].log(val_mse_loss.item())
                    val_loss = val_mse_loss+val_bin_loss
                else:
                    val_loss = loss(val_preds,val_masks)

                val_loss = val_loss.item()
                val_loss_list.append(val_loss)

                if not 'current_k_fold' in train_parameters:
                    nept_run['validation_loss'].log(val_loss)
                else:
                    nept_run[f'validation_loss_{train_parameters["current_k_fold"]}'].log(val_loss)

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
                        current_img = np.concatenate((current_img[0:3,:,:],current_img[2:5,:,:]),axis=2)
                    elif in_channels==4:
                        current_img = np.concatenate((np.stack((current_img[0,:,:],)*3,axis=1),current_img[0:3,:,:]),axis=2)
                    elif in_channels==2:
                        current_img = np.concatenate((current_img[0,:,:],current_img[1,:,:]),axis=-1)
                elif type(in_channels)==list:
                    if sum(in_channels)==6:
                        current_img = np.concatenate((current_img[0:3,:,:],current_img[2:5,:,:]),axis=2)
                    elif sum(in_channels)==2:
                        current_img = np.concatenate((current_img[0,:,:][None,:,:],current_img[1,:,:][None,:,:]),axis=2)

                img_dict = {'Image':current_img, 'Pred_Mask':current_pred,'Ground_Truth':current_gt}

                fig = visualize_continuous(img_dict,output_type)

                # Different process for saving comparison figures vs. only predictions
                if output_type == 'comparison':
                    fig.savefig(output_dir+f'Training_Epoch_{i}_Example.png')
                    nept_run[f'Example_Output_{i}'].upload(output_dir+f'Training_Epoch_{i}_Example.png')
                elif output_type == 'prediction':

                    im = Image.fromarray(fig.astype(np.uint8))
                    im.save(output_dir+f'Training_Epoch_{i}_Example.tif')
                    nept_run[f'Example_Output_{i}'].upload(output_dir+f'Training_Epoch_{i}_Example.tif')

    if not i%save_step==0:
        torch.save(model.state_dict(),model_dir+f'Collagen_Seg_Model_Latest.pth')

    loss_df = pd.DataFrame(data = {'TrainingLoss':train_loss_list,'ValidationLoss':val_loss_list})
    loss_df.to_csv(output_dir+'Training_Validation_Loss.csv')
    return model_dir+f'Collagen_Seg_Model_Latest.pth'



