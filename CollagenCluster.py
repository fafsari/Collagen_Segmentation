"""

Performing dimensional reduction from latent features extracted from image patches 


"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn

import plotly.express as px

from sklearn.preprocessing import StandardScaler
import umap

import segmentation_models_pytorch as smp


class Clusterer:
    def __init__(self,
                 output_dir,
                 plot_labels = None,
                 save_scaler_properties = True,
                 save_latent_features = False,
                 save_umap_coordinates = True):
        
        self.plot_labels = plot_labels

        self.output_folder = output_dir

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        # Post-processing features (spatial averaging and flattening)
        self.feature_post_extract = nn.Sequential(
            nn.AvgPool2d(kernel_size=15),
            nn.Flatten(start_dim=1)
        )
        self.feature_post_extract.to(self.device)

        self.save_latent_features = save_latent_features
        self.save_umap_coordinates = save_umap_coordinates
        self.save_scaler_properties = save_scaler_properties

    def run_clustering_iterator(self,model_file,dataset):
        
        self.load_model(model_file)

        # extracting features and labels
        extracted_features, labels = self.extract_feature_loop(dataset)
        print('Done extracting Features!')

        if self.save_latent_features:
            self.save_features(extracted_features,labels,None,file_name = 'Extracted_Latent_Features.csv')
            print('Done Saving Features!')
        
        # UMAP reduction
        reduced_features = self.reduce_features(extracted_features)
        print('Done Reducing Features')

        if self.save_umap_coordinates:
            self.save_features(reduced_features,['UMAP1','UMAP2'], labels, file_name = 'Extracted_UMAP_Coordinates.csv')
            print('Done saving UMAP coordinates')

        # Make some plot from the reduced features and labels
        if self.plot_labels is None:
            plot = self.gen_plot(reduced_features)
            print('Done Generating Plot!')

        else:
            plot = self.gen_plot(reduced_features,labels)

    def load_model(self,model_file):

        model_details = self.parameters['model_details']
        encoder = model_details['encoder']
        encoder_weights = model_details['encoder_weights']
        target_type = model_details['target_type']
        ann_classes = model_details['ann_classes']
        active = model_details['active']


        if target_type=='binary':
            n_classes = len(ann_classes)
        else:
            n_classes = 1

        in_channels = int(self.parameters['preprocessing']['image_size'].split(',')[-1])

        if model_details['architecture']=='Unet++':
            self.model = smp.UnetPlusPlus(
                encoder_name = encoder,
                encoder_weights = encoder_weights,
                in_channels = in_channels,
                classes = n_classes,
                activation = active
            )

        self.model.load_state_dict(torch.load(model_file))
        self.model.to(self.device)
        self.model.eval()

    def extract_feature_loop(self,dataset):

        with torch.no_grad():

            test_dataloader = DataLoader(dataset)
            print(f'length of dataset: {len(dataset)}')
            labels = []
            all_features = None

            data_iterator = iter(test_dataloader)
            with tqdm(range(len(dataset.images)),desc='Extracting Features') as pbar:
                for i in range(0,len(dataset.images)):
                    
                    if dataset.patch_batch:

                        n_patches = len(dataset.cached_data[i])
                        image_name = dataset.cached_item_names[i]

                        for n in range(0,n_patches):
                            
                            image, _, input_name = next(data_iterator)
                            input_name = ''.join(input_name).split(os.sep)[-1]

                            feature_maps = self.model.encoder(image.to(self.device))[-1]
                            features = self.feature_post_extract(feature_maps)

                            if all_features is None:
                                all_features = features.cpu().numpy()
                            else:
                                features = features.cpu().numpy()
                                all_features = np.concatenate((all_features,features),axis=0)

                            labels.append(input_name)
                            pbar.update(1)
                    else:

                        try:
                            image, _, input_name = next(data_iterator)
                        except StopIteration:
                            data_iterator = iter(data_iterator)
                            image, _, input_name = next(data_iterator)
                            
                        input_name = ''.join(input_name).split(os.sep)[-1]
                        
                        feature_maps = self.model.encoder(image.to(self.device))[-1]
                        features = self.feature_post_extract(feature_maps)

                        if all_features is None:
                            all_features = features.cpu().numpy()
                        else:
                            features = features.cpu().numpy()
                            all_features = np.concatenate((all_features,features),axis=0)

                        labels.append(input_name)
                        pbar.update(1)
            
            print(f'shape of features: {np.shape(all_features)}')
                            
        return all_features, labels

    def save_features(self,features,col_labels,row_labels,file_name):

        features_df = pd.DataFrame(data = features, columns = col_labels, index = row_labels)
        features_df.to_csv(self.output_folder+file_name)

    def reduce_features(self,features):

        scaler = StandardScaler().fit(features)
        # Saving the scaler information to the output directory
        scaler_means = scaler.mean_
        scaler_vars = scaler.var_

        if self.save_scaler_properties:
            output_mean_file = self.output_folder+'/scaler_means.npy'
            with open(output_mean_file, 'wb') as f:
                np.save(f, scaler_means)
            
            output_var_file = self.output_folder+'/scaler_var.npy'
            with open(output_var_file,'wb') as f:
                np.save(f, scaler_vars)

        scaled_data = scaler.transform(features)

        umap_reducer = umap.UMAP()
        embedding = umap_reducer.fit_transform(scaled_data)

        return embedding

    def gen_plot(self,umap_features,labels = None):

        if self.plot_labels is None:
            scatter_plot = px.scatter(
                data_frame = pd.DataFrame(data=umap_features,columns = ['umap1','umap2'],index=labels),
                x = 'umap1',
                y='umap2',
                title = "UMAP of latent features"
            )

            output_umap_plots = self.output_folder+'UMAP_Plots/'
            if not os.path.exists(output_umap_plots):
                os.makedirs(output_umap_plots)

            self.save_plot(scatter_plot,'UMAP_Plots/UMAP_Plot.png')
            
        else:
            # self.plot_labels is a pandas DataFrame containing an "Image_Names" column that aligns with the "labels" passed to this function
            column_labels = [i for i in self.plot_labels.columns.tolist() if not i=='Image_Names']
            umap_df = pd.DataFrame(data=umap_features,columns=['umap1','umap2'])
            umap_df['Image_Names'] = labels

            merged_df = pd.merge(umap_df,self.plot_labels,on='Image_Names')
            print(f'shape of merged: {merged_df.shape}')

            # Making output directory
            output_umap_plots = self.output_folder+'UMAP_Plots/'
            if not os.path.exists(output_umap_plots):
                os.makedirs(output_umap_plots)

            for f in column_labels:
                
                scatter_plot = px.scatter(
                    data_frame = merged_df,
                    x='umap1',
                    y='umap2',
                    color=f,
                    title = f'UMAP of latent features labeled with: {f}'
                )

                self.save_plot(scatter_plot,f'UMAP_Plots/UMAP_Plot_{f}.png')

        return scatter_plot

    def save_plot(self,plot,filename):

        plot.write_image(self.output_folder+'/'+filename)

    def cluster_in_loop(self,model,image):

        # Extract latent features given an image, return latent features
        feature_maps = model.encoder(image.to(self.device))[-1]
        features = self.feature_post_extract(feature_maps)

        return features
    











