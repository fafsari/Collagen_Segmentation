"""

Summarizing extracted features. 

Saving brightfield, DUET, collagen mask (thresholded) as well as merged images

For minimum, maximum, and median for each feature

"""


import os
import sys

import numpy as np
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd



def main():

    base_dir = 'D:\\Collagen_Segmentation\\Same_Training_Set_Data\\'
    model_dir = f'{base_dir}Results\\All_Results_and_Models\\Ensemble_RGB\\'

    features = f'{model_dir}Collagen_Quantification\\Patch_Collagen_Features.csv'
    collagen_masks = f'{model_dir}Testing_Output\\'
    bf_imgs = f'{base_dir}B\\'
    f_imgs = f'{base_dir}F\\'

    # Reading in the features
    features_df = pd.read_csv(features,index_col=0)
    feature_names = features_df.columns.tolist()
    feature_names = [i for i in feature_names if not i=='Image Names']

    for f in feature_names:
        print(f'On feature: {f}')
        # Creating feature save directory
        feature_save_dir = f'{features.replace("Patch_Collagen_Features.csv",f)}\\'
        if not os.path.exists(feature_save_dir):
            os.makedirs(feature_save_dir)

        # Accessing values for this particular features
        feature_values = features_df[f].values
        feature_min = np.min(feature_values)
        feature_max = np.max(feature_values)
        feature_median = np.median(feature_values)

        # Getting the image name associated with each of those values
        image_min = features_df[features_df[f]==feature_min]['Image Names'].tolist()[0]
        #print(f'image_min: {image_min}, {feature_min}')
        image_max = features_df[features_df[f]==feature_max]['Image Names'].tolist()[0]
        #print(f'image_max: {image_max}, {feature_max}')
        try:
            image_median = features_df[features_df[f]==feature_median]['Image Names'].tolist()[0]
            #print(f'image_median: {image_median}, {feature_median}')
        except IndexError:
            # median doesn't really mean there has to be an actual instance of this value. lame.
            feature_diff = abs(feature_values-feature_median)
            min_diff = np.argmin(feature_diff)
            image_median = features_df['Image Names'].tolist()[min_diff]
            #print(f'image_median: {image_median}, diff: {feature_diff[min_diff]}')
    
        # Reading collagen masks
        mask_min = np.array(Image.open(collagen_masks+image_min))>(255*0.1)
        mask_max = np.array(Image.open(collagen_masks+image_max))>(255*0.1)
        mask_median = np.array(Image.open(collagen_masks+image_median))>(255*0.1)

        # editing the name so it matches original naming and file extension
        min_name = image_min.replace('Test_Example_','').replace('tif','jpg')
        max_name = image_max.replace('Test_Example_','').replace('tif','jpg')
        median_name = image_median.replace('Test_Example_','').replace('tif','jpg')

        # Reading bf images
        bf_min = np.array(Image.open(bf_imgs+min_name))
        bf_max = np.array(Image.open(bf_imgs+max_name))
        bf_median = np.array(Image.open(bf_imgs+median_name))

        # Reading f images
        f_min = np.array(Image.open(f_imgs+min_name))
        f_max = np.array(Image.open(f_imgs+max_name))
        f_median = np.array(Image.open(f_imgs+median_name))

        # Merging bf and collagen mask
        merged_bf_min = mask_min[:,:,None]*(255-bf_min)
        merged_f_min = mask_min[:,:,None]*(f_min)
        merged_bf_max = mask_max[:,:,None]*(255-bf_max)
        merged_f_max = mask_max[:,:,None]*(f_max)
        merged_bf_median = mask_median[:,:,None]*(255-bf_median)
        merged_f_median = mask_median[:,:,None]*(f_median)

        # Creating side-by-side views of each image type
        # BF, F, Mask, Merged BF, Merged F
        merged_min = np.concatenate((bf_min,f_min,np.repeat(255*mask_min[:,:,None],repeats=3,axis=-1),merged_bf_min,merged_f_min),axis=1)
        merged_max = np.concatenate((bf_max,f_max,np.repeat(255*mask_max[:,:,None],repeats=3,axis=-1),merged_bf_max,merged_f_max),axis=1)
        merged_median = np.concatenate((bf_median,f_median,np.repeat(255*mask_median[:,:,None],repeats=3,axis=-1),merged_bf_median,merged_f_median),axis=1)

        # Creating plots
        min_plot = px.imshow(
            img = Image.fromarray(np.uint8(merged_min)),
            title = min_name
        )
        min_plot.update_layout(
            margin = {'b':0,'l':0,'r':0}
        )
        max_plot = px.imshow(
            img = Image.fromarray(np.uint8(merged_max)),
            title = max_name
        )
        max_plot.update_layout(
            margin = {'b':0,'l':0,'r':0}
        )
        median_plot = px.imshow(
            img = Image.fromarray(np.uint8(merged_median)),
            title = median_name
        )
        median_plot.update_layout(
            margin = {'b':0,'l':0,'r':0}
        )

        # Saving to the feature_save_dir
        min_plot.write_image(feature_save_dir+'Minimum.png')
        max_plot.write_image(feature_save_dir+'Maximum.png')
        median_plot.write_image(feature_save_dir+'Median.png')


if __name__=='__main__':
    main()












