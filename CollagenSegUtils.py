"""

Utilities included here for collagen segmentation task.  

This includes:

output figure generation, 
metrics calculation,
etc.


"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from PIL import Image

from Segmentation_Metrics_Pytorch.metric import BinaryMetrics
from skimage.transform import resize
from skimage.color import rgb2gray, rgb2lab, lab2rgb

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

    if n_classes==2:
        image = img[:,:,1]
    else:
        image = img[:,:,0]
        for cl in range(1,n_classes):
            image = np.concatenate((image, img[:,:,cl]),axis = 1)

    return image

def visualize_multi_task(images,output_type):
    
    n = len(images)

    if output_type=='comparison':
        fig = plt.figure(constrained_layout = True)
        subfigs = fig.subfigures(1,3)
        image_keys = list(images.keys())
        for outer_ind,subfig in enumerate(subfigs.flat):
            
            current_key = image_keys[outer_ind]

            subfig.suptitle(current_key)

            if len(images[current_key].shape)==4:
                img = images[current_key][0,:,:,:]
            else:
                img = images[current_key]
            
            if np.shape(img)[0]<np.shape(img)[-1]:
                img = np.moveaxis(img,source=0,destination=-1)
            img = np.float32(img)

            if image_keys[outer_ind]=='Image':
                img_ax = subfig.add_subplot(1,1,1)
                img_ax.imshow(img)
            else:
                neg_img = np.uint8(255*np.round(img[:,:,0]))
                coll_img = np.uint8(255*img[:,:,1])


                axs = subfig.subplots(1,2)
                titles = ['Continuous','Binary']
                sub_imgs = [coll_img,neg_img]
                cmaps = ['jet','jet']
                for innerind,ax in enumerate(axs.flat):
                    ax.set_title(current_key+'_'+titles[innerind])
                    ax.set_xticks([])
                    ax.set_yticks([])

                    ax.imshow(sub_imgs[innerind],cmap=cmaps[innerind])

    elif output_type=='prediction':
        pred_mask = images['Pred_Mask']

        if len(np.shape(pred_mask))==4:
            pred_mask = pred_mask[0,:,:,:]

        pred_mask = np.float32(pred_mask)

        if np.shape(pred_mask)[0]<np.shape(pred_mask)[-1]:
            pred_mask = np.moveaxis(pred_mask,source=0,destination = -1)

        neg_output = 255*np.round(pred_mask[:,:,0])
        coll_output = 255*pred_mask[:,:,1]

        #print(f'Collagen min/max: {np.min(coll_output)},{np.max(coll_output)}')
        #print(f'Negative image min/max: {np.min(neg_output)},{np.max(neg_output)}')

        fig = [coll_output,neg_output]

    return fig

def visualize_continuous(images,output_type):

    if output_type=='comparison':
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

            img = np.float32(img)

            if np.shape(img)[0]<np.shape(img)[-1]:
                img = np.moveaxis(img,source=0,destination=-1)

            if key == 'Pred_Mask' or key == 'Ground_Truth':
                img = apply_colormap(img)

                plt.imshow(img,cmap='jet')
            else:

                plt.imshow(img)
        output_fig = plt.gcf()

    elif output_type=='prediction':
        pred_mask = images['Pred_Mask']

        if len(np.shape(pred_mask))==4:
            pred_mask = pred_mask[0,:,:,:]

        pred_mask = np.float32(pred_mask)

        if np.shape(pred_mask)[0]<np.shape(pred_mask)[-1]:
            pred_mask = np.moveaxis(pred_mask,source=0,destination = -1)

        output_fig = apply_colormap(pred_mask)


    return output_fig

def get_metrics(pred_mask,ground_truth,img_name,calculator,target_type):

    metrics_row = {}

    if target_type=='binary':
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
    elif target_type == 'nonbinary':
        square_diff = (ground_truth.numpy()-pred_mask.numpy())**2
        mse = np.mean(square_diff)

        norm_mse = (square_diff-np.min(square_diff))/np.max(square_diff)
        norm_mse = np.mean(norm_mse)

        metrics_row['MSE'] = [round(mse,4)]
        metrics_row['Norm_MSE']=[round(norm_mse,4)]

    elif target_type == 'multi_task':
        bin_gt = ground_truth[:,0,:,:]
        bin_gt = torch.squeeze(bin_gt)
        bin_pred = pred_mask[0,:,:]
        

        acc, dice, precision, recall, sensitivity = calculator(bin_gt,torch.round(bin_pred))
        metrics_row['Accuracy'] = [round(acc.numpy().tolist(),4)]
        metrics_row['Dice'] = [round(dice.numpy().tolist(),4)]
        metrics_row['Precision'] = [round(precision.numpy().tolist(),4)]
        metrics_row['Recall'] = [round(recall.numpy().tolist(),4)]
        metrics_row['Specificity'] = [round(specificity.numpy().tolist(),4)]
        metrics_row['Sensitivity'] = [round(sensitivity.numpy().tolist(),4)]

        reg_gt = ground_truth[:,1,:,:]
        reg_gt = torch.squeeze(reg_gt)
        reg_pred = pred_mask[1,:,:]

        square_diff = (reg_gt.numpy()-reg_pred.numpy())**2
        mse = np.mean(square_diff)

        norm_mse = (square_diff-np.min(square_diff))/np.max(square_diff)
        norm_mse = np.mean(norm_mse)

        metrics_row['MSE'] = [round(mse,4)]
        metrics_row['Norm_MSE'] = [round(norm_mse,4)]


    metrics_row['ImgLabel'] = img_name

    return metrics_row

# Function to resize and apply any condensing transform like grayscale conversion
def resize_special(img,output_size,transform):

    # multi-image input transform
    if 'multi_input' in transform:
        if transform=='multi_input_rgb':
            img = resize(img,output_shape=(output_size))
        elif transform =='multi_input_invbf':
            # Inverting brightfield channels
            img = resize(img,output_shape=(output_size))

            f_img = img[:,:,0:3]
            f_img = f_img/np.sum(f_img,axis=-1)[:,:,None]
            b_img = 255-img[:,:,2:5]
            b_img = b_img/np.sum(b_img,axis=-1)[:,:,None]

            img = np.concatenate((f_img,b_img),axis=-1)
        elif transform =='multi_input_green_invbf':
            # Green channels, inverting bf
            #img = resize(img, output_shape = (output_size))

            f_img = img[:,:,1]
            f_img = (f_img - np.min(f_img))/np.ptp(f_img)
            
            b_img = 255-img[:,:,4]
            b_img = (b_img - np.min(b_img))/np.ptp(b_img)

            img = np.concatenate((f_img[:,:,None],b_img[:,:,None]),axis=-1)
            img = resize(img,output_shape = (output_size))
        elif transform == 'multi_input_mean_invbf':

            # Mean of color channels, inverting bf
            f_img = np.mean(img[:,:,0:3],axis=-1)
            f_img = (f_img - np.min(f_img))/np.ptp(f_img)

            b_img = 255-np.mean(img[:,:,2:5],axis=-1)
            b_img = (b_img - np.min(b_img))/np.ptp(b_img)

            img = np.concatenate((f_img[:,:,None],b_img[:,:,None]),axis=-1)
            img = resize(img, output_shape = (output_size))
    else:
        if transform=='mean':
            img = resize(img,output_size)

            img = np.mean(img,axis = -1)
            img = img[:,:,np.newaxis]

        elif transform in ['red','green','blue']:

            color_list = ['red','green','blue']
            img = img[:,:,color_list.index(transform)]
            img = img[:,:,np.newaxis]
            img = resize(img,output_size)

        elif transform == 'rgb2gray':
            img = resize(img,output_size)

            img = rgb2gray(img)
            img = img[:,:,np.newaxis]

        elif transform == 'rgb2lab':
            img = resize(img,output_size)

            img = rgb2lab(img)
        
        elif type(transform)==dict:

            # Determining non-tissue regions to mask out prior to scaling/conversion
            # For BF images the non-tissue regions are closer to white whereas with fluorescence images they 
            # are closer to black
            img = resize(img,output_size)

            lab_img = rgb2lab(img)
            scaled_img = (lab_img-np.nanmean(lab_img))/np.nanstd(lab_img)

            for i in range(3):
                scaled_img[:,:,i] = scaled_img[:,:,i]*transform['norm_std'][i]+transform['norm_mean'][i]
            
            # converting back to rgb
            img = (scaled_img-np.nanmean(scaled_img))/np.nanstd(scaled_img)

        elif transform == 'invert_bf_intensity':
            # Grabbing the green channel from both the fluorescence and brightfield images
            f_green_img = img[:,:,1]
            f_green_img = np.divide(f_green_img,np.sum(img[:,:,0:3],axis=-1),where=(np.sum(img[:,:,0:3],axis=-1)!=0))
            # Inverting brightfield channels
            b_green_inv_img = 255-img[:,:,3]
            b_green_inv_img = np.divide(b_green_inv_img,np.sum(255-img[:,:,2:5],axis=-1),where=(np.sum(255-img[:,:,2:5],axis=-1)!=0))
            img = np.concatenate((f_green_img[:,:,None],b_green_inv_img[:,:,None]),axis=-1)

        elif transform == 'invert_bf_01norm':

            inv_bf = 255-img[:,:,2:5]
            inv_bf_norm = np.divide(inv_bf,np.sum(inv_bf,axis=-1)[:,:,None],where=(np.sum(inv_bf,axis=-1)[:,:,None]!=0))
            f_img = img[:,:,0:3]
            f_norm = np.divide(f_img,np.sum(f_img,axis=-1)[:,:,None],where=(np.sum(f_img,axis=-1)[:,:,None]!=0))

            img = np.concatenate((f_norm,inv_bf_norm),axis=-1)

    img = resize(img,output_size)

    return img













