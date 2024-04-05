# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:28:08 2021

@author: spborder


CS Testing best model

from: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb


"""

from typing import Type
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from tqdm import tqdm
import os
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import pandas as pd

import neptune
import umap

import plotly.express as px

from Segmentation_Metrics_Pytorch.metric import BinaryMetrics
from CollagenSegUtils import visualize_continuous, get_metrics
from CollagenCluster import Clusterer
from CollagenSegTrain import EnsembleModel
from tifffile import imsave
    
        
def Test_Network(model_path, dataset_valid, nept_run, test_parameters):

    model_details = test_parameters['model_details']

    if 'scaler_means' not in model_details:
        test_parameters['model_details']['scaler_means'] = None
        
    encoder = model_details['encoder']
    encoder_weights = model_details['encoder_weights']

    # Loading clusterer to cluster latent features
    clusterer = Clusterer(test_parameters)

    ann_classes = model_details['ann_classes']
    active = model_details['active']
    target_type = model_details['target_type']
    output_dir = test_parameters['output_dir']

    if active == 'None':
        active = None

    if target_type=='binary':
        n_classes = len(ann_classes)
    elif target_type == 'nonbinary':
        n_classes = 1

    in_channels = int(test_parameters['preprocessing']['image_size'].split(',')[-1])
    output_type = 'prediction'

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

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

    if torch.cuda.is_available:
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    with torch.no_grad():

        # Evaluating model on test set
        #test_dataloader = DataLoader(dataset_valid)

        test_output_dir = output_dir+'/Testing_Output/'
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
            
        if dataset_valid.testing_metrics:
            if target_type=='binary':
                metrics_calculator = BinaryMetrics()
                testing_metrics_df = pd.DataFrame(data = {'Dice':[],'Accuracy':[],'Recall':[],'Precision':[],'Specificity':[]})
            elif target_type =='nonbinary':
                metrics_calculator = []
                testing_metrics_df = pd.DataFrame(data = {'MSE':[],'Norm_MSE':[]})

        # Setting up iterator to generate images from the validation dataset
        data_iterator = iter(dataset_valid)
        all_latent_features = None
        clustering_labels = []

        if dataset_valid.patch_batch:
            print('Using patch prediction pipeline')
        else:
            print('Images are the same size as the model inputs')

        with tqdm(range(len(dataset_valid)),desc='Testing') as pbar:
            for i in range(0,len(dataset_valid.images)):
                
                # Initializing combined mask from patched predictions
                if dataset_valid.patch_batch:

                    save_name = dataset_valid.cached_item_names[i].split(os.sep)[-1]
                    og_file_ext = save_name.split('.')[-1]
                    save_name = save_name.replace('.'+og_file_ext,'_prediction.tif')

                    # Getting original image dimensions from test_dataloader
                    original_image, _ = dataset_valid.images[i]
                    original_image_size = np.shape(original_image)
                    final_pred_mask = np.zeros((original_image_size[0],original_image_size[1]))
                    overlap_mask = np.zeros_like(final_pred_mask)

                    patch_size = [int(i) for i in test_parameters['preprocessing']['image_size'].split(',')[0:-1]]
                    
                    # Now getting the number of patches needed for the current image
                    n_patches = len(dataset_valid.cached_data[i])
                    image_name = dataset_valid.cached_item_names[i]
                    #print(f'image name: {image_name}')

                    # Grabbing list of data at once:
                    image_list, _, input_name_list = next(data_iterator)
                    for image, input_name in zip(image_list, input_name_list):

                        #for n in range(0,n_patches):

                        #image, _, input_name = next(data_iterator)
                        input_name = ''.join(input_name).split(os.sep)[-1]
                        #print(f'input_name: {input_name}')

                        pred_mask = model(image.to(device))
                        if not test_parameters['model_details']['scaler_means'] is None:
                            pred_latent_features = clusterer.cluster_in_loop(model,image.to(device))

                            if all_latent_features is None:
                                all_latent_features = pred_latent_features.cpu().numpy()
                                clustering_labels.append({'Full_Image_Name':image_name,'Patch_Name':input_name})
                            else:
                                pred_latent_features = pred_latent_features.cpu().numpy()
                                all_latent_features = np.concatenate((all_latent_features,pred_latent_features),axis=0)
                                clustering_labels.append({'Full_Image_Name':image_name,'Patch_Name':input_name})

                        # Detaching prediction from gradients, converting to numpy array
                        pred_mask_img = pred_mask.detach().cpu().numpy()

                        # Getting patch locations from input_name
                        row_start = int(input_name.split('_')[-2])
                        col_start = int(input_name.split('_')[-1].split('.')[0])

                        """
                        fig = visualize_continuous(
                            images = {'Pred_Mask':pred_mask_img},
                            output_type = 'prediction'
                        )
                        """
                        
                        # Adding to final_pred_mask and overlap_mask
                        #final_pred_mask[row_start:row_start+patch_size[0],col_start:col_start+patch_size[1]] += (fig*255).astype(np.uint8)
                        final_pred_mask[row_start:row_start+patch_size[0],col_start:col_start+patch_size[1]] += np.squeeze(pred_mask_img*255)

                        overlap_mask[row_start:row_start+patch_size[0],col_start:col_start+patch_size[1]] += np.ones((patch_size[0],patch_size[1]))
                        
                        #img_fig = Image.fromarray((fig*255).astype(np.uint8))
                        #img_fig.save(test_output_dir+input_name+'.png')
                        pbar.update(1)

                    # Scaling predictions by overlap (mean pixel prediction where there is overlap)
                    final_pred_mask = np.multiply(final_pred_mask,1/overlap_mask)

                    im = Image.fromarray((final_pred_mask).astype(np.uint8))
                    # Smoothing image to get rid of grid lines
                    im = im.filter(ImageFilter.SMOOTH_MORE)
                    im.save(test_output_dir+save_name)
                    #imsave(test_output_dir+save_name,final_pred_mask)

                    # Saving overlap mask
                    #overlap_mask = (overlap_mask-np.min(overlap_mask))/(np.max(overlap_mask))
                    #overlap_im = Image.fromarray((overlap_mask*255).astype(np.uint8))
                    #overlap_im.save(test_output_dir+'Overlap_Mask.tif')

                else:

                    try:
                        image, target, input_name = next(data_iterator)
                        input_name = ''.join(input_name)
                    except StopIteration:
                        data_iterator = iter(test_dataloader)
                        image, target, input_name = next(data_iterator)
                        input_name = ''.join(input_name)

                    input_name = input_name.split('/')[-1]

                    # Add something here so that it calculates perforance metrics and outputs
                    # raw values for 2-class segmentation(not binarized output masks)
                    #pred_mask = model.predict(image.to(device))
                    pred_mask = model(image.to(device))

                    if not test_parameters['model_details']['scaler_means'] is None:
                        pred_latent_features = clusterer.cluster_in_loop(model,image.to(device))
                        if all_latent_features is None:
                            all_latent_features = pred_latent_features.cpu().numpy()
                            clustering_labels.append({'Patch_Name':input_name})
                        else:
                            pred_latent_features = pred_latent_features.cpu().numpy()
                            all_latent_features = np.concatenate((all_latent_features,pred_latent_features),axis=0)
                            clustering_labels.append({'Patch_Name':input_name})

                    if target_type=='binary':        
                        target_img = target.cpu().numpy().round()
                        pred_mask_img = pred_mask.detach().cpu().numpy()

                        if dataset_valid.testing_metrics:
                            testing_metrics_df = pd.concat([testing_metrics_df,pd.DataFrame(get_metrics(pred_mask.detach().cpu(),target.cpu(), input_name, metrics_calculator,target_type))],ignore_index=True)
                        # Outputting the prediction as a continuous mask even though running binary metrics

                    elif target_type=='nonbinary':
                        pred_mask_img = pred_mask.detach().cpu().numpy()
                        target_img = target.cpu().numpy()

                        if dataset_valid.testing_metrics:
                            testing_metrics_df = pd.concat([testing_metrics_df,pd.DataFrame(get_metrics(pred_mask.detach().cpu(),target.cpu(), input_name, metrics_calculator,target_type))],ignore_index=True)

                    image = image.cpu().numpy()
                    if type(in_channels)==int:
                        if in_channels==6:
                            image = np.concatenate((image[:,0:3,:,:],image[:,2:5,:,:]),axis=2)
                        elif in_channels == 4:
                            image = np.concatenate((np.stack((image[:,0,:,:],)*3,axis=1),image[:,0:3,:,:]),axis=2)
                        elif in_channels == 2:
                            image = np.concatenate((image[:,0,:,:],image[:,1,:,:]),axis=-1)
                    elif type(in_channels)==list:
                        if sum(in_channels)==6:
                            image = np.concatenate((image[:,0:3,:,:],image[:,2:5,:,:]),axis=2)
                        elif sum(in_channels)==2:
                            image = np.concatenate((image[:,0,:,:][None,:,:],image[:,1,:,:][None,:,:]),axis=2)

                    img_dict = {'Image':image,'Pred_Mask':pred_mask_img,'Ground_Truth':target_img}

                    fig = visualize_continuous(img_dict,output_type)

                    pbar.update(1)

                    # Different process for saving comparison figures vs. only predictions
                    if output_type=='comparison':
                        fig.savefig(test_output_dir+'Test_Example_'+input_name)
                        #nept_run['testing/Testing_Output_'+input_name].upload(test_output_dir+'Test_Example_'+input_name)
                    elif output_type=='prediction':
                        
                        im = Image.fromarray((fig*255).astype(np.uint8))
                        im.save(test_output_dir+'Test_Example_'+input_name.replace('.jpg','.tif'))

                # Used during hyperparameter optimization to compute objective value
                if dataset_valid.testing_metrics:
                    testing_metrics_df.to_csv(test_output_dir+'Test_Metrics.csv')
                
                    if not 'current_k_fold' in test_parameters:
                        nept_run['Test_Image_metrics'].upload(neptune.types.File.as_html(testing_metrics_df))

                        for met in testing_metrics_df.columns.values.tolist():
                            try:
                                print(f'{met} value: {testing_metrics_df[met].mean()}')
                                nept_run[met] = testing_metrics_df[met].mean()
                            except TypeError:
                                print(f'Number of samples: {testing_metrics_df.shape[0]}')
                    else:
                        current_k_fold = test_parameters['current_k_fold']
                        nept_run[f'Test_Image_metrics_{current_k_fold}'].upload(neptune.types.File.as_html(testing_metrics_df))

                        for met in testing_metrics_df.columns.values.tolist():
                            try:
                                print(f'{met}: value: {testing_metrics_df[met].mean()}')
                                nept_run[met+f'_{current_k_fold}'] = testing_metrics_df[met].mean()
                            except TypeError:
                                print(f'Number of samples: {testing_metrics_df.shape[0]}')


        # Scaling according to reference training set scaler values
        if not test_parameters['model_details']['scaler_means'] is None:
            
            # Rows are samples, columns are "features"
            all_latent_features = all_latent_features - test_parameters['model_details']['scaler_means'][None,:]
            all_latent_features = all_latent_features / test_parameters['model_details']['scaler_var'][None,:]
            print(np.shape(all_latent_features))
            
            # Removing NaN's, if present
            all_latent_features = all_latent_features[~np.isnan(all_latent_features).any(axis=1)]
            print(np.shape(all_latent_features))

            if not np.shape(all_latent_features)[0]==0:
                umap_reducer = test_parameters['model_details']['umap_reducer']
                embeddings = umap_reducer.fit_transform(all_latent_features)

                cluster_df = pd.DataFrame.from_records(clustering_labels)

                cluster_df['umap1'] = embeddings[:,0]
                cluster_df['umap2'] = embeddings[:,1]
                
                if dataset_valid.patch_batch:

                    # UMAP with the full image name as the label
                    umap_scatter = px.scatter(
                        data_frame = cluster_df,
                        x='umap1',
                        y='umap2',
                        color='Full_Image_Name',
                        title = 'UMAP of latent features, Testing Only'
                    )
                else:

                    # UMAP scatter plot
                    umap_scatter = px.scatter(
                        data_frame = cluster_df,
                        x='umap1',
                        y='umap2',
                        title = 'UMAP of latent features, Testing Only'
                    )
                
                # Saving UMAP coordinates and plots
                umap_scatter.write_image(test_parameters['output_dir']+'Test_UMAP.png')
                umap_scatter.write_html(test_parameters['output_dir']+'Test_UMAP.html')

                cluster_df.to_csv(test_parameters['output_dir']+'Test_UMAP_Coordinates.csv')
            else:
                print('Oops! All NaNs!')



