"""

Quantifying collagen content in each predicted patch

Metrics:
    - Binarized:
        - Total area collagen (thresholded (thresheld?))
        - Total area not-collagen (same threshold)
        - Area ratio collagen
        - Distance Transform:
            - Mean
            - Median
            - Standard Deviation
            - Maximum
            - Minimum
    - Continuous:
        - Channel-wise? Grayscale?
            - Mean intensity
            - Median intensity
            - Std. Dev intensity
            - Max intensity
            - Texture

"""

import os
import sys
import numpy as np
import pandas as pd
import argparse

from glob import glob

import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import binary_fill_holes
from PIL import Image

from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go


class Quantifier:
    def __init__(self,
                 bf_image_dir:str,
                 f_image_dir:str,
                 mask_dir:str,
                 output_dir:str,
                 threshold:float):
        
        self.bf_image_dir = bf_image_dir
        self.f_image_dir = f_image_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.threshold = threshold

        # Getting predicted image patches from self.image_dir
        # Should all have .tif extension
        self.mask_paths = sorted(glob(self.mask_dir+'*.tif'))
        self.f_image_paths = [i.replace(self.mask_dir,self.f_image_dir).replace('Test_Example_','').replace('_prediction','').replace('.tif','.jpg') for i in self.mask_paths]
        self.bf_image_paths = [i.replace(self.f_image_dir,self.bf_image_dir) for i in self.f_image_paths]
        print(f'--------------On: {self.output_dir.split("/")[-3]} -------------------------')
        print(f'------------------Found: {len(self.mask_paths)} Images!---------------------')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_file_path = self.output_dir+'/Patch_Collagen_Features.csv'

        # This will be a list of features for all images, list of dictionaries
        all_feature_list = []
        for img_idx,img in tqdm(enumerate(self.mask_paths),total = len(self.mask_paths)):

            # Reading the image:
            og_pred_image = np.array(Image.open(img))
            bin_image = self.binarize(og_pred_image)

            if np.sum(bin_image)>0:

                bf_image = np.mean(255-np.array(Image.open(self.bf_image_paths[img_idx])),axis=-1)
                f_image = np.mean(np.array(Image.open(self.f_image_paths[img_idx])),axis=-1)

                # Verifying image name alignment
                #print(f'Prediction name: {img.split(os.sep)[-1]}')
                #print(f'BF image name: {self.bf_image_paths[img_idx].split(os.sep)[-1]}')
                #print(f'F image name: {self.f_image_paths[img_idx].split(os.sep)[-1]}')

                masked_bf_image = np.uint8(bin_image*bf_image)
                masked_f_image = np.uint8(bin_image*f_image)

                # Global and distance transform features can still be using the prediction
                # The merged image features are the intensity and texture features
                glob_features = self.global_features(bin_image)
                dt_features = self.distance_transform_features(bin_image)

                # BF merged intensity and texture features
                bf_int_features = self.intensity_features(masked_bf_image)
                bf_text_features = self.texture_features(masked_bf_image)

                bf_features = {**bf_int_features,**bf_text_features}

                # F merged intensity and texture features
                f_int_features = self.intensity_features(masked_f_image)
                f_text_features = self.texture_features(masked_f_image)

                f_features = {**f_int_features,**f_text_features}

                # Renaming keys in each set of dictionaries
                renamed_bf_features = {}
                for b in bf_features:
                    renamed_bf_features[f'BF {b}'] = bf_features[b]

                renamed_f_features = {}
                for f in f_features:
                    renamed_f_features[f'F {f}'] = f_features[f]

                # Merging dictionaries (3.9<=)
                #image_features = glob_features | int_features | text_features | dt_features
                # Merging dictionaries (3.5<=)
                image_features = {**glob_features, **dt_features, **renamed_bf_features, **renamed_f_features}
                image_features['Image Names'] = img.split('/')[-1]

                all_feature_list.append(image_features)

        # Saving combined feature dataframe
        feature_df = pd.DataFrame.from_records(all_feature_list)
        feature_df.to_csv(self.output_file_path)

        self.plot_feature(feature_df)


    def binarize(self,image):
        # Take in a continuous image prediction (2D) and return binarized image

        bin_img = image.copy()
        bin_img[bin_img<(255*self.threshold)] = 0.0
        bin_img[bin_img>=(255*self.threshold)] = 1.0

        bin_img = np.uint8(bin_img)

        return bin_img

    def global_features(self,bin_image):

        # Takes as input the binary image and returns global features
        total_patch_area = np.shape(bin_image)[0]*np.shape(bin_image)[1]
        collagen_area = np.sum(bin_image)
        if collagen_area>0:
            other_area = binary_fill_holes(bin_image)
            area_ratio = collagen_area/other_area
        else:
            other_area = total_patch_area
            area_ratio = 0.0

        feature_dict = {
            'Total Patch Area':total_patch_area,
            'Filled Area': other_area,
            'Collagen Area':collagen_area,
            'Collagen Area Ratio':area_ratio
        }

        return feature_dict

    def distance_transform_features(self,bin_image):
        
        # Takes as input the binarized collagen image and returns distance transform features
        # Should already be uint8 type
        # Using L2 for distance, mask_size = 5 
        dist_transform_img = cv2.distanceTransform(bin_image,cv2.DIST_L2,5)
        # Setting 0 values to nan for statistics
        dist_transform_img[dist_transform_img==0] = np.nan

        sum_distance = np.nansum(dist_transform_img)
        mean_distance = np.nanmean(dist_transform_img)
        max_distance = np.nanmax(dist_transform_img)
        std_distance = np.nanstd(dist_transform_img)
        med_distance = np.nanmedian(dist_transform_img)

        feature_dict = {
            'Sum Distance Transform': sum_distance,
            'Mean Distance Transform': mean_distance,
            'Max Distance Transform': max_distance,
            'Standard Deviation Distance Transform': std_distance,
            'Median Distance Transform': med_distance
        }

        return feature_dict
    
    def intensity_features(self,gray_image):

        # Take as input the grayscale image and return a dictionary of intensity features
        intensity_image = gray_image.copy().astype(float)
        # Ignoring non-collagen regions (defined using threshold)
        intensity_image[intensity_image<(255*self.threshold)] = np.nan

        mean_int = np.nanmean(intensity_image)
        med_int = np.nanmedian(intensity_image)
        std_int = np.nanstd(intensity_image)
        max_int = np.nanmax(intensity_image)

        feature_dict = {
            'Mean Intensity': mean_int,
            'Median Intensity': med_int,
            'Standard Deviation Intensity': std_int,
            'Maximum Intensity': max_int
        }

        return feature_dict

    def texture_features(self,gray_image):

        # Taking the grayscale collagen image and returning dictionary of texture features
        texture_image = gray_image.copy()
        # Ignoring non-collagen regions (using self.threshold)
        texture_image[texture_image<(255*self.threshold)] = 0

        texture_matrix = graycomatrix(texture_image,[1],[0],levels=256, symmetric=True, normed=True)
        
        texture_feature_names = ['Contrast','Homogeneity','Correlation','Energy']
        feature_dict = {}
        for i,t_name in enumerate(texture_feature_names):
            t_feat_value = graycoprops(texture_matrix,t_name.lower())
            feature_dict[t_name] = t_feat_value[0][0]

        return feature_dict

    def plot_feature(self,feature_df):

        for feat in feature_df.columns.tolist():

            if not feat=='Image Names':

                feat_bar = px.violin(
                    data_frame=feature_df,
                    y = feat
                )

                feat_bar.update_layout(
                    yaxis = {
                        'title':{
                            'text':f'<b>{feat}</b>'
                        }
                    },
                    title = {
                        'text': f'<b>Violin plot of {feat}<br>across predicted image patches'
                    }
                )

                feat_bar.write_image(self.output_dir+f'{feat}.png')



def main(args):

    feature_extractor = Quantifier(
        bf_image_dir = args.bf_image_dir,
        f_image_dir = args.f_image_dir,
        mask_dir = args.test_image_path,
        output_dir = args.output_dir,
        threshold = args.threshold
    )


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description = 'Collagen segmentation evaluation argument parser'
    )

    parser.add_argument('--test_image_path',type=str, help='Path to predicted collagen masks')
    parser.add_argument('--bf_image_dir',type=str,help='Path to brightfield image patches')
    parser.add_argument('--f_image_dir',type=str,help='Path to fluorescence image patches')
    parser.add_argument('--output_dir',type=str, default='Evaluation_Metrics',help='If you want to save the output to another path, specify here. (no / needed at the end).')
    parser.add_argument('--threshold',type=float,default = 0.1, help = 'Value used to threshold grayscale images to make them binary')

    main(parser.parse_args())
