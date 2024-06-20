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
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
from tqdm import tqdm
import sys
import os
import gc

from CollagenSegUtils import visualize_continuous, set_requires_grad, loop_iterable
# from CollagenModels import Discriminator

# Calculate MMD between source and target features
def compute_mmd(source_features, target_features):
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)
    mmd_loss = torch.norm(source_mean - target_mean, p=2)
    return mmd_loss
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def TrainingADDA_Loop(model_path, dataset_train_s, dataset_train_t, dataset_valid, train_parameters, nept_run):
    
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

    # image_size = train_parameters["preprocessing"]["image_size"]
    image_size = [int(i) for i in train_parameters['preprocessing']['image_size'].split(',')]
    n_channels = image_size[-1]//2
    image_size = (image_size[0], image_size[1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Is training on GPU available? : {torch.cuda.is_available()}')
    print(f'Device is : {device}')
    print(f'Torch Cuda version is : {torch.version.cuda}')    

    nept_run['Architecture'] = model_details['architecture']
    nept_run['Loss'] = train_parameters['loss']
    nept_run['output_type'] = output_type
    nept_run['lr'] = train_parameters['lr']
    
    # Define source and target models    
    source_model = create_model(model_details['architecture']).to(device)
    target_model = create_model(model_details['architecture'], ema=True).to(device)
    
    # Load source model from our trained model saved in model_path
    if torch.cuda.is_available():
        source_model.load_state_dict(torch.load(model_path))
        target_model.load_state_dict(torch.load(model_path))
    else:
        source_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        target_model.load_state_dict(torch.load(model_path), map_location=torch.device('cpu'))
    
    source_model.eval()
    # set_requires_grad(source_model, requires_grad=False)

    # Sending target_model to current device ('cuda','cuda:0','cuda:1',or 'cpu')    
    loss = loss.to(device) 
    scaler = GradScaler()   
    
    # Initialize KLDivLoss
    kld_loss = torch.nn.KLDivLoss(reduction='batchmean').to(device)


    batch_size = train_parameters['batch_size']
    # Define Discriminator model
    # discriminator = Discriminator()
    discriminator = Discriminator(img_size=image_size, channels=1).to(device)
    # discriminator = torch.nn.Sequential(
    #     torch.nn.Linear(image_size[0]*image_size[1], 1000),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(1000, 500),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(500, 250),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(250, 1),
    #     torch.nn.Sigmoid()
    # ).to(device)
        
    # Loss function
    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
    
    optimizer_G = torch.optim.Adam([
            dict(params = target_model.parameters(), lr = train_parameters['lr'], weight_decay = 0.0001)
            ])
    
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, patience=250)
    
    optimizer_D = torch.optim.Adam([
            dict(params = discriminator.parameters(), lr = train_parameters['lr'], weight_decay = 0.0001)
            ])      

    # batch_size = train_parameters['batch_size']
    #train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 12)
    #valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = 4)

    train_loader_s = DataLoader(dataset_train_s, batch_size=batch_size//2, shuffle=True, pin_memory=True)
    train_loader_t = DataLoader(dataset_train_t, batch_size=batch_size//2, shuffle=True, pin_memory=True)
    # batch_iterator = zip(loop_iterable(train_loader_s), loop_iterable(train_loader_t))
    batch_iterator = zip(iter(train_loader_s), iter(train_loader_t))
    
    # train_iter_s = iter(train_loader_s)                                  
    # train_iter_t = iter(train_loader_t)
        
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_iter = iter(valid_loader)
    
    # Maximum number of epochs defined here as well as how many steps between model saves and example outputs
    epoch_num = train_parameters['step_num']
    save_step = train_parameters['save_step']

    train_loss = 0
    d_loss = 0
    g_loss = 0
    val_loss = 0

    # Recording training and validation loss
    train_loss_list_d = []
    train_loss_list_g = []
    val_loss_list = []
        
    # target_model.train()
    with tqdm(total = epoch_num, position = 0, leave = True, file = sys.stdout) as pbar:

        # for i in range(0,epoch_num):
        i = 0
        while i < epoch_num: 
            
            # Controlling the progress bar, printing training and validation losses
            if i == 1: 
                pbar.set_description(f'Step: {i}/{epoch_num}')
                pbar.update(i)
            elif i%5 == 0:
                if i == 0:
                    pbar.set_description(f'Step: {i}/{epoch_num}, Train Loss: ({round(d_loss,4)}, {round(g_loss,4)}) / Val Loss: ,{round(val_loss,4)}')
                else:
                    # pbar.set_description(f'Step: {i}/{epoch_num}, Train Loss: ({d_loss.item():.4f}, {g_loss.item():.4f}) / Val Loss: ,{val_loss:.4f}')
                    pbar.set_description(f'Step: {i}/{epoch_num}, Train Loss: ({d_loss:.4f}, {g_loss.item():.4f}) / Val Loss: ,{val_loss:.4f}')
                pbar.update(5)                    
            
            try:
                (source_x, _, _), (target_x, _, _) = next(batch_iterator)
            except StopIteration:
                # batch_iterator = zip(loop_iterable(train_loader_s), loop_iterable(train_loader_t))
                batch_iterator = zip(iter(train_loader_s), iter(train_loader_t))
                (source_x, _, _), (target_x, _, _) = next(batch_iterator)
                
            source_x, target_x = source_x.to(device), target_x.to(device)
            
            # Train discriminator k_disc times
            # set_requires_grad(target_model, requires_grad=False)
            # set_requires_grad(discriminator, requires_grad=True)
            for _ in range(train_parameters["k_disc"]):
                
                source_model.eval()
                target_model.eval()
                discriminator.train()
                
                i += 1
                # print(f"\Iter---> {i}")
                if i > epoch_num:
                    break
                # try:
                #     (source_x, _, _), (target_x, _, _) = next(batch_iterator)
                # except StopIteration:
                #     batch_iterator = zip(loop_iterable(train_loader_s), loop_iterable(train_loader_t))
                #     (source_x, _, _), (target_x, _, _) = next(batch_iterator)
                    
                # source_x, target_x = source_x.to(device), target_x.to(device)

                # Clear existing gradients in optimizer
                optimizer_D.zero_grad() 
                
                # # For Linear Discriminator model
                # source_features = source_model(source_x).view(source_x.shape[0], -1)
                # target_features = target_model(target_x).view(target_x.shape[0], -1)
                                
                # For CNN Discriminator model
                with autocast():
                    source_output_trusted   = source_model(source_x)    # label 1                                               
                    source_output_trusted   = discriminator(source_output_trusted)
                    source_loss_t = adversarial_loss(source_output_trusted, torch.ones_like(source_output_trusted))
                    
                    del source_output_trusted
                    gc.collect()
                    
                    source_output_untrusted = target_model(source_x)    # label 0                
                    source_output_untrusted = discriminator(source_output_untrusted)                
                    source_loss_u = adversarial_loss(source_output_untrusted, torch.zeros_like(source_output_untrusted))
                                    
                    
                    del source_output_untrusted
                    gc.collect()
                    
                    
                    target_output_trusted   = target_model(target_x)    # label 1
                    target_output_trusted   = discriminator(target_output_trusted)
                    target_loss_t = adversarial_loss(target_output_trusted, torch.ones_like(target_output_trusted))                
                    
                    del target_output_trusted                
                    gc.collect()
                    
                    target_output_untrusted = source_model(target_x)    # label 0                
                    target_output_untrusted = discriminator(target_output_untrusted)
                    target_loss_u = adversarial_loss(target_output_untrusted, torch.zeros_like(target_output_untrusted))  
                    
                    del target_output_untrusted
                    gc.collect()
                    
                    # Overall adversarial loss
                    d_loss = 0.25 * source_loss_t + 0.25 * target_loss_t + 0.25 * source_loss_u + 0.25 * target_loss_u                              
                    
                # batch_x = torch.concat([source_x, target_x])
                # source_outputs = source_model(batch_x)
                # target_outputs = target_model(batch_x)    
                
                # source_output = source_model(source_x)
                # target_output = target_model(target_x)
                
                # source_output = discriminator(source_features.detach())
                # target_output = discriminator(target_features.detach())
                
                # source_loss = adversarial_loss(source_output, torch.ones_like(source_output))
                # target_loss = adversarial_loss(target_output, torch.zeros_like(target_output))
                
                # d_loss = 0.5 * source_loss + 0.5 * target_loss

                # optimizer_D.zero_grad()
                # d_loss.backward()
                # clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                # optimizer_D.step()
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                scaler.update()
                
                train_loss_list_d.append(d_loss.item())
                
                if 'current_k_fold' in train_parameters:
                    nept_run[f'training_loss_{train_parameters["current_k_fold"]}'].log(train_loss)
                else:
                    nept_run['training_d_loss'].log(d_loss)            
            
            
            # Train target model k_gen times
            # set_requires_grad(target_model, requires_grad=True)
            # set_requires_grad(discriminator, requires_grad=False)
            
            for _ in range(train_parameters["k_gen"]):
                                
                target_model.train()                
                discriminator.eval()                
                
                # i += 1
                if i > epoch_num:
                    break                
                # try:
                #     _, (target_x, _, _) = next(batch_iterator)
                #     # (source_x, _, _), (target_x, _, _) = next(batch_iterator)
                # except StopIteration:
                #     batch_iterator = zip(loop_iterable(train_loader_s), loop_iterable(train_loader_t))
                #     _, (target_x, _, _) = next(batch_iterator)
                #     # (source_x, _, _), (target_x, _, _) = next(batch_iterator)
                    
                # # source_x = source_x.to(device)
                # target_x = target_x.to(device)
                
                # Clear existing gradients in optimizer
                optimizer_G.zero_grad()
                
                # target_features = target_model(target_x).view(target_x.shape[0], -1)
                # source_features = source_model(source_x)
                target_output = target_model(target_x)
                # print("source_features.shape, target_features.shape", source_features.shape, target_features.shape)
                
                # target_output = discriminator(target_features)

                # flipped labels
                g_loss = adversarial_loss(target_output, torch.ones_like(target_output))

                # # MMD loss
                # g_loss = compute_mmd(torch.squeeze(source_features).flatten(), torch.squeeze(target_features).flatten()) 
                # print("Loss:", g_loss.item())
                
                # KL Divergence Loss                
                # if torch.unique(target_features).item() != 1.0:
                #     print("\nTarget Model Output:",torch.min(target_features), torch.max(target_features), torch.unique(target_features), target_features.shape)
                # if torch.unique(source_features).item() != 1.0:
                #     print("Source Model Output:",torch.min(source_features), torch.max(source_features), torch.unique(source_features), source_features.shape)
                # g_loss = kld_loss(torch.log(target_features), source_features)
                
                
                g_loss.backward()
                clip_grad_norm_(target_model.parameters(), max_norm=5.0)
                optimizer_G.step()
                
                train_loss_list_g.append(g_loss.item())
                
                if 'current_k_fold' in train_parameters:
                    nept_run[f'training_loss_{train_parameters["current_k_fold"]}'].log(train_loss)
                else:            
                    nept_run['training_g_loss'].log(g_loss)        
            
            if i > epoch_num:
                break
            # # Turning on dropout
            # target_model.train()

            
            # # Clear existing gradients in optimizer_G
            # optimizer_G.zero_grad()

            # # Loading training and validation samples from dataloaders
            # try:
            #     train_imgs_src, _, _ = next(train_iter_s)
            # except StopIteration:                        
            #     train_iter_s = iter(train_loader_s)
            #     train_imgs_src, _, _ = next(train_iter_s)
            # try:
            #     train_imgs_tar, _, _ = next(train_iter_t)
            # except StopIteration:                        
            #     train_iter_t = iter(train_loader_t)
            #     train_imgs_tar, _, _ = next(train_iter_t)
                
            # # Sending to device
            # train_imgs_src, train_imgs_tar = train_imgs_src.to(device), train_imgs_tar.to(device)
            
            # # Generate source features
            # source_feats = source_model(train_imgs_src)
            
            # # Generate target features
            # target_feats = target_model(train_imgs_tar)

            # # Train Discriminator
            # optimizer_D.zero_grad()

            # source_output, target_output = discriminator(source_feats.detach(), target_feats.detach())
            # source_loss = adversarial_loss(source_output, torch.ones_like(source_output))
            # target_loss = adversarial_loss(target_output, torch.zeros_like(target_output))
            # # d_loss = (source_loss + target_loss) / 2
            # d_loss = 0.75 * source_loss + 0.25 * target_loss

            # d_loss.backward()
            # optimizer_D.step()

            # # Train target model
            # optimizer_G.zero_grad()
            
            # _, target_output = discriminator(source_feats, target_feats)
            # g_loss = adversarial_loss(target_output, torch.ones_like(target_output))

            # g_loss.backward()
            # optimizer_G.step()

            # train_loss_list_d.append(d_loss.item())
            # train_loss_list_g.append(g_loss.item())

            # if 'current_k_fold' in train_parameters:
            #     nept_run[f'training_loss_{train_parameters["current_k_fold"]}'].log(train_loss)
            # else:
            #     nept_run['training_d_loss'].log(d_loss)
            #     nept_run['training_g_loss'].log(g_loss)
            
            # # Updating optimizer_G
            # optimizer_G.step()

            # Validation (don't want it to influence gradients in network)
            with torch.no_grad():
                # This turns off any dropout in the network 
                target_model.eval()
                
                try:
                    val_imgs, val_masks, _ = next(val_iter)
                except StopIteration:                        
                    val_iter = iter(valid_loader)
                    val_imgs, val_masks, _ = next(val_iter)
                    
                val_imgs = val_imgs.to(device)
                val_masks = val_masks.to(device)

                # Predicting on the validation images
                val_preds = target_model(val_imgs)
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

            # Saving target_model if current i is a multiple of "save_step"
            # Also generating example output segmentation and uploading that to Neptune
            if i%save_step == 0:
                torch.save(target_model.state_dict(),model_dir+f'Collagen_Seg_Model_Latest.pth')

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
        torch.save(target_model.state_dict(),model_dir+f'Collagen_Seg_Model_Latest.pth')

    # Determine the maximum length among the lists
    max_length = max(len(train_loss_list_d), len(train_loss_list_g), len(val_loss_list))

    # Pad shorter lists with NaN values to make them equal in length
    train_loss_list_d += [np.nan] * (max_length - len(train_loss_list_d))
    train_loss_list_g += [np.nan] * (max_length - len(train_loss_list_g))
    val_loss_list += [np.nan] * (max_length - len(val_loss_list))

    loss_df = pd.DataFrame(data = {'TrainingLoss_d':train_loss_list_d,
                                   'TrainingLoss_g':train_loss_list_g,
                                   'ValidationLoss':val_loss_list})
    loss_df.to_csv(output_dir+'Training_Validation_Loss.csv')
    return model_dir+f'Collagen_Seg_Model_Latest.pth'



