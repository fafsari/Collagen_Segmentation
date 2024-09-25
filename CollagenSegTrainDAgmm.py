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
import torchvision as tv
# import medicaltorch.transforms as mt_transforms
# import medicaltorch.datasets as mt_datasets

import albumentations

from skimage.transform import resize

from Augmentation_Functions import *
from CollagenSegUtils import resize_special
# from CollagenModels import EnsembleModel, EnsembleModelMIT, EnsembleModelMAnet, EnsembleModelSE, create_model
from CollagenModels import create_model
from CollagenLoss import *

import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
from tqdm import tqdm
import sys
import os
import time
import gc

from CollagenSegUtils import visualize_continuous, set_requires_grad, loop_iterable
# from CollagenModels import Discriminator

# Calculate MMD between source and target features
def compute_mmd(source_features, target_features):
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)
    mmd_loss = torch.norm(source_mean - target_mean, p=2)
    return mmd_loss

def get_current_consistency_weight(weight, epoch, rampup):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * sigmoid_rampup(epoch, rampup)

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def update_target_variables(source_model, target_model, alpha, global_step):
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    p = 0.5
    alpha = min(1 - 1 / (global_step + 1)**p, alpha)
    # alpha = 0.6
    one_minus_alpha = 1.0 - alpha    
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):    
        # target_param.data.mul_(alpha).add_(one_minus_alpha, source_param.data)
        target_param = alpha * target_param.data + one_minus_alpha * source_param.data

def linked_batch_augmentation(input_batch, preds_unsup):
    
    teacher_transforms = ComposeDouble([
            AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
            AlbuSeg2d(albumentations.RandomBrightnessContrast(p=0.2)),
            AlbuSeg2d(albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5,rotate_limit=45,interpolation=1,p=0.1)),
            AlbuSeg2d(albumentations.VerticalFlip(p=0.5)),
            FunctionWrapperDouble(np.moveaxis, input = True, target = True, source = -1, destination = 0)
        ])

    input_batch_size = input_batch.size(0)

    input_batch_cpu = input_batch.cpu().detach().numpy()
    preds_unsup_cpu = preds_unsup.cpu().detach().numpy()

    samples_linked_aug = []
    for sample_idx in range(input_batch_size):
        x = np.moveaxis(input_batch_cpu[sample_idx], 0, -1)  # Move channels to last dimension
        y = np.moveaxis(preds_unsup_cpu[sample_idx], 0, -1)  # Move channels to last dimension

        # Apply transformations
        augmented_x, augmented_y = teacher_transforms(x, y)

        samples_linked_aug.append((augmented_x, augmented_y))
            
    # Separate the augmented samples into inputs and predictions
    augmented_inputs, augmented_preds = zip(*samples_linked_aug)

    # Convert lists to numpy arrays
    augmented_inputs = np.array(augmented_inputs)
    augmented_preds = np.array(augmented_preds)

    # Convert numpy arrays back to PyTorch tensors
    augmented_inputs_tensor = torch.tensor(augmented_inputs, dtype=torch.float32)
    augmented_preds_tensor = torch.tensor(augmented_preds, dtype=torch.float32)

    return augmented_inputs_tensor, augmented_preds_tensor

def TrainingADDA_Loop(model_path, dataset_train_s, dataset_valid_s, dataset_train_t, dataset_valid_t, train_parameters, nept_run):
    
    model_details = train_parameters['model_details']

    if not model_details['architecture'] == 'DUnet':
        encoder = model_details['encoder']
        encoder_weights = model_details['encoder_weights']
        dropout_rate = model_details['dropout_rate']
        pretrained = True if model_details['pretrained']==1 else False
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

    in_channels = int(model_details['in_channels'])
    
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
    
    batch_size = int(train_parameters['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Is training on GPU available? : {torch.cuda.is_available()}')
    print(f'Device is : {device}')
    print(f'Torch Cuda version is : {torch.version.cuda}')    

    nept_run['Architecture'] = model_details['architecture']
    nept_run['Loss'] = train_parameters['loss']
    nept_run['output_type'] = output_type
    nept_run['lr'] = train_parameters['lr']
    
    source_model = create_model(model_details, n_classes)
    target_model = create_model(model_details, n_classes)
    
    source_model.to(device)
    target_model.to(device)
    
    if pretrained:
        print(f"Loading pre-trained model from {model_path}")            
        source_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)        

        # Update target model's state_dict with source model's parameters
        source_state_dict = source_model.state_dict()
        target_state_dict = target_model.state_dict()
        target_state_dict.update(source_state_dict)

        # Load the updated state_dict into target model
        target_model.load_state_dict(target_state_dict, strict=False)

    dummy_input = torch.zeros(batch_size, in_channels, image_size[0], image_size[1]).to(device)
    _ = target_model(dummy_input)
    for param in target_model.parameters():
        param.detach_()
    
    torch.cuda.empty_cache()
    # discriminator = torch.nn.Sequential(
    #     torch.nn.Linear(image_size[0] * image_size[1] , 512),
    #     torch.nn.Dropout(0.5),  # Add dropout before the linear layer with dropout probability of 0.5
    #     torch.nn.LeakyReLU(0.2, inplace=True),
    #     torch.nn.Linear(512, 1)
    # )
    
    # Sending loss to current device ('cuda','cuda:0','cuda:1',or 'cpu')    
    loss = loss.to(device) 
                
    optimizer = torch.optim.Adam([
            dict(params = source_model.parameters(), lr = train_parameters['lr'], weight_decay = 0.0001)
            ])
    # Define the learning rate scheduler with adjusted patience    
    # lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=250)
    lr_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=250)
    
    train_loader_s = DataLoader(dataset_train_s, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loader_t = DataLoader(dataset_train_t, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    train_loader_s_iter = iter(train_loader_s)
    train_loader_t_iter = iter(train_loader_t)

    valid_loader_s = DataLoader(dataset_valid_s, batch_size=64, shuffle=True, pin_memory=True)
    val_iter_s = iter(valid_loader_s)
    
    valid_loader_t = DataLoader(dataset_valid_t, batch_size=64, shuffle=True, pin_memory=True)
    val_iter_t = iter(valid_loader_t)
    
    # Maximum number of epochs defined here as well as how many steps between model saves and example outputs
    step_num = train_parameters['step_num']
    save_step = train_parameters['save_step']

    train_loss = 0
    consistency_loss = 0
    seg_loss = 0
    # d_loss = 0
    # g_loss = 0
    val_loss_s = 0
    val_loss_t = 0
    best_val_loss = float('inf')

    # Recording training and validation loss
    train_loss_list = []
    train_loss_list_seg = []
    train_loss_list_consistency = []
    # train_loss_list_d = []
    # train_loss_list_g = []
    val_loss_list_s = []
    val_loss_list_t = []
    
           
    # for global_step in tqdm(range(1, step_num + 1), desc="Epochs"):
    with tqdm(total = step_num, position = 0, leave = True, file = sys.stdout) as pbar:
        start_time = time.time()
        
        global_step = 0
        weight_step = 0    
        while global_step < step_num:
            torch.cuda.empty_cache()
            if global_step % 50 == 0:
                consistency_weight = get_current_consistency_weight(
                    train_parameters['cons_weight'], 
                    weight_step, 
                    train_parameters['consistency_rampup'])
                weight_step += 1
            
            # Controlling the progress bar, printing training and validation losses
            if global_step == 1: 
                pbar.set_description(f'Step: {global_step}/{step_num}')
                pbar.update(global_step)
            elif global_step%5 == 0:
                if global_step == 0:
                    pbar.set_description(f'Step: {global_step}/{step_num}, Train Loss: {train_loss:.4f} / Val Loss Source: ,{val_loss_s:.4f} / Val Loss Target: ,{val_loss_t:.4f}')
                else:                    
                    pbar.set_description(f'Step: {global_step}/{step_num}, Train Loss: {train_loss:.4f} / Val Loss Source: ,{val_loss_s:.4f} / Val Loss Target: ,{val_loss_t:.4f}')
                pbar.update(5)                    
                        
            source_model.train()
            target_model.train()
            
            try:
                source_train_batch = next(train_loader_s_iter)
            except StopIteration:
                train_loader_s_iter = iter(train_loader_s)
                source_train_batch = next(train_loader_s_iter)

            source_train_inputs, source_train_masks, _ = source_train_batch
            source_train_inputs, source_train_masks = source_train_inputs.to(device), source_train_masks.to(device)
            
            source_train_preds = source_model(source_train_inputs)
            
            # seg_loss = loss(source_train_preds,source_train_masks)
            seg_loss = structure_loss(source_train_preds,source_train_masks)
            
            try:
                target_adapt_batch = next(train_loader_t_iter)
            except StopIteration:
                train_loader_t_iter = iter(train_loader_t)
                target_adapt_batch = next(train_loader_t_iter)

            target_adapt_inputs, _, _ = target_adapt_batch
            target_adapt_inputs = target_adapt_inputs.to(device)
            
            # Teacher forward
            with torch.no_grad():
                teacher_preds_unsup = target_model(target_adapt_inputs)  
            
            # print("Before:", target_adapt_inputs.shape, teacher_preds_unsup.shape)            
            adapt_input_batch, teacher_preds_unsup_aug = linked_batch_augmentation(target_adapt_inputs, teacher_preds_unsup)
            adapt_input_batch, teacher_preds_unsup_aug = adapt_input_batch.to(device), teacher_preds_unsup_aug.to(device)
            # print("After:", adapt_input_batch.shape, teacher_preds_unsup_aug.shape)
                
            # Student forward
            student_preds_unsup = source_model(adapt_input_batch)

            # consistency loss calculates the inconsistency between predictions of unaugmented target batch
            #  and augmneted target batch by target and source models, respectively
            consistency_loss = torch.nn.functional.mse_loss(student_preds_unsup, teacher_preds_unsup_aug)
            
            # MMD loss calculated shift gap between mean of source_batch and target_batch
            # print(teacher_preds_unsup_aug.shape)
            # marginal_loss = torch.nn.functional.mse_loss(source_train_preds, teacher_preds_unsup)
            # class_conditional_loss = class_conditional_mmd(source_train_preds, teacher_preds_unsup_aug)
            # class_conditional_loss = class_conditional_mmd(student_preds_unsup, teacher_preds_unsup_aug)
            
            # train_loss = seg_loss + (consistency_weight * ((consistency_loss + marginal_loss + 2 * class_conditional_loss) / 3))
            train_loss = seg_loss + consistency_weight * consistency_loss
            # train_loss = seg_loss + (consistency_weight * (2*consistency_loss + marginal_loss + 2*class_conditional_loss))
            # train_loss = seg_loss + (consistency_weight * (consistency_loss + 2 * class_conditional_loss))
            
            optimizer.zero_grad()
            train_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(source_model.parameters(), 0.2)
            optimizer.step()
            
            train_loss_list.append(train_loss.item())
            train_loss_list_consistency.append(consistency_loss.item())
            train_loss_list_seg.append(seg_loss.item())
            
            global_step += 1
            
            # updating target model's params using its previous step and source model params balancing by alpha
            if global_step < train_parameters['target_late_epoch']:
                update_target_variables(source_model, target_model, train_parameters['target_alpha'], global_step)
            else:
                update_target_variables(source_model, target_model, train_parameters["target_alpha_late"], global_step)
                
            if 'current_k_fold' in train_parameters:
                nept_run[f'training_loss_{train_parameters["current_k_fold"]}'].log(train_loss)
            else:            
                nept_run['training_loss'].log(train_loss)
                nept_run['segmentation_loss'].log(seg_loss)
                nept_run['consistencey_loss'].log(consistency_loss)
                nept_run['consistency_weight'].log(consistency_weight)
                # nept_run['marginal_loss'].log(marginal_loss)
                # nept_run['class_conditional_loss'].log(class_conditional_loss)                

            # Validation (don't want it to influence gradients in network)
            with torch.no_grad():
                # This turns off any dropout in the network 
                source_model.eval()
                target_model.eval()
                
                # # Initialize cumulative loss and counters
                # total_val_loss_s = 0
                # total_val_loss_t = 0
                # num_batches_s = 0
                # num_batches_t = 0

                # # Evaluate source model on entire validation set
                # for val_imgs_s, val_masks_s, _ in valid_loader_s:
                try:
                    val_batch = next(val_iter_s)
                except StopIteration:
                    val_iter_s = iter(valid_loader_s)
                    val_batch = next(val_iter_s)

                val_imgs_s, val_masks_s, _ = val_batch
                val_imgs_s = val_imgs_s.to(device)
                val_masks_s = val_masks_s.to(device)

                # Predicting on the validation images
                val_preds_s = source_model(val_imgs_s)
                # Finding validation loss
                val_loss_s = loss(val_preds_s, val_masks_s)
                
                val_loss_list_s.append(val_loss_s.item())
                
                # total_val_loss_s += val_loss_s.item()
                # num_batches_s += 1

                # # Calculate average validation loss for source model
                # avg_val_loss_s = total_val_loss_s / num_batches_s
                # val_loss_list_s.append(avg_val_loss_s)
                    
                # Evaluate target model on entire validation set
                # for val_imgs_t, val_masks_t, _ in valid_loader_t:
                try:
                    val_batch = next(val_iter_t)
                except StopIteration:
                    val_iter_t = iter(valid_loader_t)
                    val_batch = next(val_iter_t)

                val_imgs_t, val_masks_t, _ = val_batch
                val_imgs_t = val_imgs_t.to(device)
                val_masks_t = val_masks_t.to(device)

                # Predicting on the validation images
                val_preds_t = target_model(val_imgs_t)
                # Finding validation loss
                val_loss_t = loss(val_preds_t, val_masks_t)
                
                val_predt_bySource = source_model(val_imgs_t)
                val_loss_t_bySource = loss(val_predt_bySource, val_masks_t)

                # total_val_loss_t += val_loss_t.item()
                # num_batches_t += 1

                # # Calculate average validation loss for target model
                # avg_val_loss_t = total_val_loss_t / num_batches_t
                val_loss_list_t.append(val_loss_t.item())                
                
                # Stepping the learning rate plateau with the current validation loss
                #scheduler.step()
                lr_plateau.step((val_loss_s + val_loss_t) / 2)                
                # lr_plateau.step(val_loss_t)
                                    
                if 'current_k_fold' in train_parameters:
                    nept_run[f'validation_loss_Src_{train_parameters["current_k_fold"]}'].log(val_loss_s)
                    nept_run[f'validation_loss_Tar_{train_parameters["current_k_fold"]}'].log(val_loss_t)
                    nept_run[f'Learning rate_{train_parameters["current_k_fold"]}'].log(optimizer.param_groups[0]['lr'])
                else:                    
                    nept_run['validation_loss_Src'].log(val_loss_s)
                    nept_run['validation_loss_Tar'].log(val_loss_t)
                    nept_run['validation_loss_Tar_bySource'].log(val_loss_t_bySource)
                    nept_run['Learning rate'].log(optimizer.param_groups[0]['lr'])

                # if val_loss_t < best_val_loss:
                #     best_val_loss = val_loss_t
                #     torch.save(source_model.state_dict(),model_dir+f'Collagen_Seg_Model_Latest.pth')
                #     torch.save(target_model.state_dict(),model_dir+f'Collagen_Seg_Model_T_Latest.pth')
                    
                # Saving target_model if current global_step is a multiple of "save_step"
                # Also generating example output segmentation and uploading that to Neptune
                if global_step % save_step == 0:
                    torch.save(source_model.state_dict(),model_dir+f'Collagen_Seg_Model_Latest.pth')
                    torch.save(target_model.state_dict(),model_dir+f'Collagen_Seg_Model_T_Latest.pth')
                    
                    # for val_imgs, val_masks in [zip(val_imgs_s, val_masks_s), zip(val_imgs_t, val_masks_t)]:
                        
                    if batch_size==1:
                        current_img_s = val_imgs_s.cpu().numpy()
                        current_img_t = val_imgs_t.cpu().numpy()
                        current_gt = np.concatenate((val_masks_s.cpu().numpy(), val_masks_t.cpu().numpy()),axis=-2)
                        current_pred = np.concatenate((val_preds_s.cpu().numpy(), val_preds_t.cpu().numpy()),axis=-2)
                        # current_gt = val_masks_s.cpu().numpy()
                        # current_pred = val_preds_s.cpu().numpy()
                    else:
                        current_img_s = val_imgs_s[0].cpu().numpy()
                        current_img_t = val_imgs_t[0].cpu().numpy()
                        current_gt = np.concatenate((val_masks_s[0].cpu().numpy(), val_masks_t[0].cpu().numpy()),axis=-2)
                        current_pred = np.concatenate((val_preds_s[0].cpu().numpy(), val_preds_t[0].cpu().numpy()),axis=-2)
                        # current_gt = val_masks_s[0].cpu().numpy()
                        # current_pred = val_preds_s[0].cpu().numpy()

                    """
                    if target_type=='binary':
                        current_pred = current_pred.round()
                    """
                    
                    if type(in_channels)==int:
                        if in_channels == 6:
                            current_img_s = np.concatenate((current_img_s[0:3,:,:],current_img_s[3:6,:,:]),axis=-2)
                            current_img_t = np.concatenate((current_img_t[0:3,:,:],current_img_t[3:6,:,:]),axis=-2)
                        elif in_channels==4:
                            current_img = np.concatenate((np.stack((current_img[0,:,:],)*3,axis=1),current_img[0:3,:,:]),axis=-2)
                        elif in_channels==2:
                            current_img_s = np.concatenate((current_img_s[0,:,:],current_img_s[1,:,:]),axis=-2)
                            current_img_t = np.concatenate((current_img_t[0,:,:],current_img_t[1,:,:]),axis=-2)
                    elif type(in_channels)==list:
                        if sum(in_channels)==6:                            
                            current_img_s = np.concatenate((current_img_s[0,:,:],current_img_s[1,:,:]),axis=-2)
                            current_img_s = np.concatenate((current_img_t[0,:,:],current_img_t[1,:,:]),axis=-2)                            
                        elif sum(in_channels)==2:
                            current_img_s = np.concatenate((current_img_s[0,:,:][None,:,:],current_img_s[1,:,:][None,:,:]),axis=-2)
                            current_img_t = np.concatenate((current_img_t[0,:,:][None,:,:],current_img_t[1,:,:][None,:,:]),axis=-2)

                    img_dict = {
                        'CGPL Image': current_img_s, 
                        'PATH Image': current_img_t,
                        'Pred_Mask': current_pred,
                        'Ground_Truth': current_gt
                        }

                    fig = visualize_continuous(img_dict, output_type)

                    # Different process for saving comparison figures vs. only predictions
                    if output_type == 'comparison':
                        fig.savefig(output_dir+f'/Training_Step_{global_step}_Example.png')
                        nept_run[f'Example_Output_{global_step}'].upload(output_dir+f'/Training_Step_{global_step}_Example.png')
                    elif output_type == 'prediction':

                        im = Image.fromarray(fig.astype(np.uint8))
                        im.save(output_dir+f'/Training_Step_{global_step}_Example.tif')
                        nept_run[f'Example_Output_{global_step}'].upload(output_dir+f'/Training_Step_{global_step}_Example.tif')

    if (not global_step%save_step == 0) and (val_loss_t < best_val_loss):
        print("Line 561: not global_step'%'save_step == 0")
        torch.save(source_model.state_dict(),model_dir+f'Collagen_Seg_Model_Latest.pth')
        torch.save(target_model.state_dict(),model_dir+f'Collagen_Seg_Model_T_Latest.pth')

    # Determine the maximum length among the lists
    max_length = max(len(train_loss_list), len(train_loss_list_seg), len(train_loss_list_consistency), len(val_loss_list_s), len(val_loss_list_t))
    
    # Pad shorter lists with NaN values to make them equal in length
    train_loss_list_seg += [np.nan] * (max_length - len(train_loss_list_seg))
    train_loss_list_consistency += [np.nan] * (max_length - len(train_loss_list_consistency))
    train_loss_list += [np.nan] * (max_length - len(train_loss_list))
    val_loss_list_s += [np.nan] * (max_length - len(val_loss_list_s))
    val_loss_list_t += [np.nan] * (max_length - len(val_loss_list_t))

    loss_df = pd.DataFrame(data = {'TrainingLoss':train_loss_list,
                                   'SegmentationLoss': train_loss_list_seg,
                                   'ConsistencyLoss': train_loss_list_consistency,
                                   'ValidationLossSrc': val_loss_list_s,
                                   'ValidationLossTar':val_loss_list_t})
    loss_df.to_csv(output_dir+'Training_Validation_Loss.csv')
    return model_dir+f'Collagen_Seg_Model_T_Latest.pth'