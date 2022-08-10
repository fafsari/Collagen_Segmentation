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
from skimage.color import rgb2gray



def back_to_reality(tar):
    
    # Getting target array into right format
    classes = np.shape(tar)[-1]
    dummy = np.zeros((np.shape(tar)[0],np.shape(tar)[1]))
    for value in range(classes):
        mask = np.where(tar[:,:,value]!=0)
        dummy[mask] = value

    return dummy

def apply_colormap(img):

    #print(f'Size of image: {np.shape(img)}')
    #print(f'Min:{np.min(img)}, Max: {np.max(img)}, Type: {img.dtype}')
    n_classes = np.shape(img)[-1]

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

            #print(f'Image shape: {img.shape}')

            if image_keys[outer_ind]=='Image':
                img_ax = subfig.add_subplot(1,1,1)
                img_ax.imshow(img)
            else:
                #coll_img = apply_colormap(img[:,:,0])
                #print(f'Pre Rounding min: {np.min(img[:,:,0])}, max: {np.max(img[:,:,0])}, mean: {np.mean(img[:,:,0])}, std: {np.std(img[:,:,0])}')
                neg_img = np.uint8(255*np.round(img[:,:,0]))
                coll_img = np.uint8(255*img[:,:,1])
                #neg_img = back_to_reality(img[:,:,1])
                #print(f'On: {current_key}')
                #print(f'Collagen min/max: {np.min(coll_img)},{np.max(coll_img)}')
                #print(f'Negative image min/max: {np.min(neg_img)},{np.max(neg_img)}')
                #print(f'mean:{np.mean(neg_img)}, std:{np.std(neg_img)}')

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

"""   
 # This visualization function can be used for binary outputs
def visualize(images,output_type):
    
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
        #print(key)
        if key == 'Pred_Mask' or key == 'Ground_Truth':
            if output_type=='binary' or key == 'Ground_Truth':
                #print('using back_to_reality')
                img = back_to_reality(img)

                plt.imshow(img)
            if output_type == 'continuous' and not key == 'Ground_Truth':
                #print('applying colormap')
                img = apply_colormap(img)

                plt.imshow(img,cmap='jet')
        else:
            plt.imshow(img)

    return plt.gcf()
    """

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

    if output_size[-1]==1:
        img = resize(img,output_size,preserve_range=True,order=0,anti_aliasing=False)
    else:
        img = resize(img,output_size)

        # Setting default size to 256,256,n_channels
        if transform=='mean':

            img = np.mean(img,axis = -1)
            img = img[:,:,np.newaxis]

        elif transform in ['red','green','blue']:
            color_list = ['red','green','blue']
            img = img[:,:,color_list.index(transform)]
            img = img[:,:,np.newaxis]
        elif transform == 'rgb2gray':

            img = rgb2gray(img)
            img = img[:,:,np.newaxis]


    return img













