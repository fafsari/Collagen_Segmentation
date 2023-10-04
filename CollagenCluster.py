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
                 dataset,
                 model_file,
                 parameters,
                 plot_labels = None,
                 save_scaler_properties = True,
                 save_latent_features = False,
                 save_umap_coordinates = True):
        
        self.dataset = dataset
        self.model_file = model_file
        self.plot_labels = plot_labels

        self.parameters = parameters
        self.output_folder = self.parameters['output_dir']

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        self.load_model()

        # Post-processing features (spatial averaging and flattening)
        self.feature_post_extract = nn.Sequential(
            nn.AvgPool2d(kernel_size=15),
            nn.Flatten(start_dim=1)
        )
        self.feature_post_extract.to(self.device)

        self.save_latent_features = save_latent_features
        self.save_umap_coordinates = save_umap_coordinates
        self.save_scaler_properties = save_scaler_properties

        # extracting features and labels
        extracted_features, labels = self.extract_feature_loop()
        print('Done extracting Features!')

        if self.save_latent_features:
            self.save_features(extracted_features,labels,file_name = 'Extracted_Latent_Features.csv')
            print('Done Saving Features!')
        
        # UMAP reduction
        reduced_features = self.reduce_features(extracted_features)
        print('Done Reducing Features')

        if self.save_umap_coordinates:
            self.save_features(reduced_features,labels, file_name = 'Extracted_UMAP_Coordinates.csv')
            print('Done saving UMAP coordinates')

        # Make some plot from the reduced features and labels
        plot = self.gen_plot(reduced_features)
        print('Done Generating Plot!')
        
        self.save_plot(plot)
        print('Plot saved!')

    def load_model(self):

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

        self.model.load_state_dict(torch.load(self.model_file))
        self.model.to(self.device)
        self.model.eval()

    def extract_feature_loop(self):

        with torch.no_grad():

            test_dataloader = DataLoader(self.dataset)
            labels = []
            all_features = None

            data_iterator = iter(test_dataloader)
            with tqdm(range(len(self.dataset)),desc='Extracting Features') as pbar:
                for i in range(0,len(self.dataset.images)):
                    
                    if 'patch_batch' in dir(self.dataset):

                        n_patches = len(self.dataset.cached_data[i])
                        image_name = self.dataset.cached_item_names[i]

                        for n in range(0,n_patches):

                            image, _, input_name = next(data_iterator)
                            input_name = ''.join(input_name).split(os.sep)[-1]

                            feature_maps = self.model.encoder(image.to(self.device))[-1]
                            features = torch.squeeze(self.feature_post_extract(feature_maps))

                            if all_features is None:
                                all_features = features.cpu().numpy()[:,None]
                            else:
                                features = features.cpu().numpy()[:,None]
                                all_features = np.concatenate((all_features,features),axis=-1)

                            labels.append(input_name)
                            pbar.update(1)
                    else:

                        image, _, input_name = next(data_iterator)
                        input_name = ''.join(input_name).split(os.sep)[-1]
                        
                        feature_maps = self.model.encoder(image)[-1]
                        features = torch.squeeze(self.feature_post_extract(feature_maps))

                        if all_features is None:
                            all_features = features.cpu().numpy()
                        else:
                            features = features.cpu().numpy()
                            all_features = np.concatenate((all_features,features),axis=-1)

                        labels.append(input_name)
                            
        return all_features, labels

    def save_features(self,features,labels,file_name):

        features_df = pd.DataFrame(data = features, columns = labels)
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

        scatter_plot = px.scatter(
            data_frame = pd.DataFrame(data=umap_features,columns = ['umap1','umap2']),
            x = 'umap1',
            y='umap2',
            color = labels,
            title = "UMAP of latent features"
        )

        return scatter_plot

    def save_plot(self,plot):

        plot.write_image(self.output_folder+'/Output_UMAP_Plot.png')

        