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

class EnsembleModel(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 active,
                 n_classes):
        super().__init__()

        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = 'resnet34'
        encoder_weights = 'imagenet'

        if self.active=='sigmoid':
            self.final_active = torch.nn.Sigmoid()
        elif self.active =='softmax':
            self.final_active = torch.nn.Softmax(dim=1)       
        elif self.active == 'linear':
            self.active = None
            self.final_active = torch.nn.Identity()
        else:
            self.active = None
            self.final_active = torch.nn.ReLU()

        self.model_b = smp.UnetPlusPlus(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = int(self.in_channels/2),
                classes = self.n_classes,
                activation = self.active
                )
        
        self.model_d = smp.UnetPlusPlus(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )

        self.combine_layers = torch.nn.Sequential(
            torch.nn.LazyConv2d(64,kernel_size=1),
            torch.nn.Dropout(p=0.1),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64,self.n_classes,kernel_size=1)
        )


    def forward(self,input):

        b_input = input[:,0:int(self.in_channels/2),:,:]
        d_input = input[:,int(self.in_channels/2):self.in_channels,:,:]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        combined_output = torch.cat((b_output,d_output),dim=1)
        final_prediction = self.final_active(self.combine_layers(combined_output))
        
        return final_prediction

class EnsembleModelMIT(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 active,
                 n_classes):
        super().__init__()

        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = 'mit_b2'
        encoder_weights = 'imagenet'

        if self.active=='sigmoid':
            self.final_active = torch.nn.Sigmoid()
        elif self.active =='softmax':
            self.final_active = torch.nn.Softmax(dim=1)       
        elif self.active == 'linear':
            self.active = None
            self.final_active = torch.nn.Identity()
        else:
            self.active = None
            self.final_active = torch.nn.ReLU()

        self.model_b = smp.Unet(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = int(self.in_channels/2),
                classes = self.n_classes,
                activation = self.active
                )
        
        self.model_d = smp.Unet(
            encoder_name = encoder,
            encoder_weights = encoder_weights,
            in_channels = int(self.in_channels/2),
            classes = self.n_classes,
            activation = self.active
        )

        self.combine_layers = torch.nn.Sequential(
            torch.nn.LazyConv2d(64,kernel_size=1),
            torch.nn.Dropout(p=0.1),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64,self.n_classes,kernel_size=1)
        )


    def forward(self,input):

        b_input = input[:,0:int(self.in_channels/2),:,:]
        d_input = input[:,int(self.in_channels/2):self.in_channels,:,:]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        combined_output = torch.cat((b_output,d_output),dim=1)
        final_prediction = self.final_active(self.combine_layers(combined_output))
        
        return final_prediction


class CrossAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.scale = in_channels ** -0.5

    def forward(self, f_input, b_input):
        # # Get dimensions of query and key
        query_dim = f_input.size(-1)
        key_dim = b_input.size(-1)
        
        # Calculate proportional scaling factor
        self.scale = (query_dim ** -0.5) * (key_dim ** -0.5)
        
        # Calculate attention scores
        attn = torch.matmul(f_input, b_input.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn = torch.nn.functional.softmax(attn, dim=-1)
        
        # Attend to values (b_input) using attention weights
        attended_features = torch.matmul(attn, b_input)
        
        return attended_features

class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = torch.nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # channels //= 8
        # Flatten query and key
        proj_query = self.query_conv(x).view(batch_size, channels, -1)  # Shape: (batch_size, channels, height*width)
        proj_key = self.key_conv(x).view(batch_size, channels, -1)      # Shape: (batch_size, channels, height*width)
        print("proj_key.shape, proj_query.shape--->",proj_key.shape, proj_query.shape)
        # Compute dot product
        energy = torch.einsum('bcn,bcm->bnm', proj_query, proj_key)  # Shape: (batch_size, height*width, height*width)
        print("energy.shape:", energy.shape)
        attention = torch.nn.functional.softmax(energy, dim=-1)      # Shape: (batch_size, height*width, height*width)
        print("attention.shape",attention.shape)

        # Flatten value
        proj_value = self.value_conv(x).view(batch_size, channels, -1)  # Shape: (batch_size, channels, height*width)
        print("proj_value.shape, attention.shape---->", proj_value.shape, attention.shape)
        # Compute output
        out = torch.einsum('bcm,bmn->bcn', proj_value, attention)       # Shape: (batch_size, channels, height*width)
        print("out.shape", out.shape)
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out
        
        # proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        # print(proj_key.shape, proj_query.shape)
        # # energy = torch.bmm(proj_query, proj_key)
        # energy = torch.einsum('bid,bjd->bij', proj_query, proj_key)
        # attention = torch.nn.functional.softmax(energy, dim=-1)
        # proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        # # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = torch.einsum('bjd,bid->bij', proj_value, attention)
        # out = out.view(batch_size, channels, height, width)
        # out = self.gamma * out + x
        # return out

class AttentionModel(torch.nn.Module):
    def __init__(self, in_channels, active, n_classes):
        super().__init__()

        self.in_channels = in_channels
        self.active = active
        self.n_classes = n_classes

        encoder = 'mit_b5'
        encoder_weights = 'imagenet'

        if self.active == 'sigmoid':
            self.final_active = torch.nn.Sigmoid()
        elif self.active == 'softmax':
            self.final_active = torch.nn.Softmax(dim=1)
        elif self.active == 'linear':
            self.active = None
            self.final_active = torch.nn.Identity()
        else:
            self.active = None
            self.final_active = torch.nn.ReLU()

        self.model_b = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        self.model_d = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=int(self.in_channels / 2),
            classes=self.n_classes,
            activation=self.active
        )

        self.cross_attention = CrossAttention(in_channels=64)
        # self.self_attention = SelfAttention(32)

        self.combine_layers = torch.nn.Sequential(
            torch.nn.LazyConv2d(64, kernel_size=1),
            torch.nn.Dropout(p=0.1),
            # torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, self.n_classes, kernel_size=1)
        )

    def forward(self, input):
        b_input = input[:, 0:int(self.in_channels / 2), :, :]
        d_input = input[:, int(self.in_channels / 2):self.in_channels, :, :]
        b_output = self.model_b.decoder(*self.model_b.encoder(b_input))
        d_output = self.model_d.decoder(*self.model_d.encoder(d_input))

        # # Apply cross-attention mechanism
        attended_b_output = self.cross_attention(d_output, b_output)

        # Concatenate the attended features
        combined_output = torch.cat((attended_b_output, d_output), dim=1)
        
        # Concatenate the features then find attended features
        # combined_output = torch.cat((b_output, d_output), dim=1)
        # print("combined_output.shape:", combined_output.shape)
        
        # # Apply self-attention mechanism
        # combined_output = self.self_attention(combined_output)
        
        final_prediction = self.final_active(self.combine_layers(combined_output))
        # final_prediction = self.final_active(combined_output)

        return final_prediction
    
    
def Training_Loop(dataset_train, dataset_valid, train_parameters, nept_run):
    
    model_details = train_parameters['model_details']

    if not model_details['architecture'] == 'DUnet':
        encoder = model_details['encoder']
        encoder_weights = model_details['encoder_weights']
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
    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 12)
    valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    train_iter = iter(train_loader)
    val_iter = iter(valid_loader)        

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



