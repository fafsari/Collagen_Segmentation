# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 08:39:49 2021

@author: spborder

Organizing inputs (JSON file) and running either training or testing

Currently works for image patches (training) and either image patches or larger (not WSI) testing

"""

import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from math import floor
import json

from sklearn.model_selection import KFold

import neptune

from Input_Pipeline import *
from CollagenSegTrain import Training_Loop
from CollagenSegTest import Test_Network
from CollagenCluster import Clusterer

# Class to use if no neptune information is provided without having to add logic every time nept_run is called
class FakeNeptune:
    def __init__(self):
        pass
    def assign(self):
        pass
    def log(self):
        pass
    def upload(self):
        pass

def main():
    # Changing up from sys.argv to reading a specific set of input parameters
    parameters_file = sys.argv[1]

    parameters = json.load(open(parameters_file))

    # Getting input parameters and neptune-specific parameters (if specified)
    input_parameters = parameters['input_parameters']

    if 'neptune' in input_parameters:
        nept_params = input_parameters['neptune']

        nept_api_token = os.environ.get('NEPTUNE_API_TOKEN')
        nept_run = neptune.init_run(
            project = nept_params['project'],
            source_files = nept_params['source_files'],
            api_token = nept_api_token,
            tags = nept_params['tags']
            )

        nept_run['input_parameters'] = input_parameters
    else:
        nept_run = None

    if input_parameters['phase']=='train':

        training_parameters = parameters['train_parameters']
        # Path to folder containing ground truth images
        if os.path.isdir(input_parameters['label_dir']):
            label_paths = sorted(glob(input_parameters['label_dir']+'*'))
        elif os.path.isfile(input_parameters['label_dir']):
            label_paths = sorted(pd.read_csv(input_parameters['label_dir'])['Paths'].tolist())

        input_image_type = list(input_parameters['image_dir'].keys())

        if input_parameters['type']=='multi':
            
            # Getting multi-input image paths
            image_paths_base = []
            for inp_type in input_image_type:
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'*')))
                if os.path.isfile(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(pd.read_csv(input_parameters['image_dir'][inp_type])['Paths'].tolist()))

            # Getting the image paths as lists of lists (each input type)
            image_paths = [list(i) for i in zip(*image_paths_base)]
        
        elif input_parameters['type']=='single':

            # Getting either DUET or Brightfield images
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'*'))
            elif os.path.isfile(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist())

        # Now determining which images are used for training and which are used for testing
        if 'k_folds' in training_parameters['train_test_set']:

            kf = KFold(n_splits = int(training_parameters['train_test_set']['k_folds']),shuffle=True)
            k_count = 1

            # Iteratively training and validating model using k-fold CV
            for train_idx, test_idx in kf.split(image_paths):

                print(f'On k-fold: {k_count}')

                train_idx = list(train_idx.astype(int))
                test_idx = list(test_idx.astype(int))

                train_images = [image_paths[i] for i in train_idx]
                test_images = [image_paths[i] for i in test_idx]
                train_labels = [label_paths[i] for i in train_idx]
                test_labels = [label_paths[i] for i in test_idx]

                #k_train_set = pd.DataFrame(data = {'Training_Image_Paths':train_images})
                #k_test_set = pd.DataFrame(data = {'Testing_Image_Paths':test_images})

                nept_run.assign({f'{k_count}_Training_Set':train_images})
                nept_run.assign({f'{k_count}_Testing_Set':test_images})

                train_dataset, validation_dataset = make_training_set(
                    phase = 'train',
                    train_img_paths=train_images,
                    train_tar=train_labels,
                    valid_img_paths=test_images,
                    valid_tar=test_labels,
                    parameters = training_parameters
                )

                model = Training_Loop(
                    dataset_train = train_dataset,
                    dataset_valid = validation_dataset,
                    train_parameters = training_parameters,
                    nept_run=nept_run
                )

                Test_Network(model,validation_dataset,nept_run,training_parameters)

                k_count+=1

        elif 'split' in training_parameters['train_test_split']:

            # Random train/test split with 'split' proportion assigned to training data
            shuffle_idx = np.random.permutation(len(image_paths))

            split_pct = training_parameters['train_test_split']['split']
            train_idx = shuffle_idx[0:floor(split_pct*len(image_paths))]
            val_idx = shuffle_idx[floor(split_pct*len(image_paths)):len(image_paths)]
            
            train_images = [image_paths[i] for i in train_idx]
            test_images = [image_paths[i] for i in val_idx]
            train_labels = [label_paths[i] for i in train_idx]
            test_labels = [label_paths[i] for i in val_idx]

            dataset_train, dataset_valid = make_training_set(
                'train',
                train_img_paths=train_images,
                train_tar = train_labels,
                valid_img_paths=test_images,
                valid_tar = test_labels,
                parameters=training_parameters
            )

            model = Training_Loop(dataset_train, dataset_valid, training_parameters,nept_run)
            Test_Network(model, dataset_valid, nept_run, training_parameters)

        elif 'training' in training_parameters['train_test_split']:
            
            # Passing a file containing the names of files in the training and testing sets
            # Adding the image_dir afterwards (dependent on whether it's a multi or single input)
            train_list = pd.read_csv(training_parameters['train_test_split']['training'])['Image_Names'].tolist()
            test_list = pd.read_csv(training_parameters['train_test_split']['testing'])['Image_Names'].tolist()

            if input_parameters['type']=='multi':
                train_images = []
                for tr in train_list:
                    # Getting each type of input
                    tr_list = []
                    for in_type in input_image_type:
                        tr_list.append(input_parameters['image_dir'][in_type]+tr)
                    # Adding lists of inputs to main train images list
                    train_images.append(tr_list)
                
                test_images = []
                for te in test_list:
                    te_list = []
                    for in_type in input_image_type:
                        te_list.append(input_parameters['image_dir'][in_type]+te)
                    
                    test_images.append(te_list)

            elif input_parameters['type']=='single':
                train_images = [input_parameters['image_dir'][input_image_type[0]]+i for i in train_list]
                test_images = [input_parameters['image_dir'][input_image_type[0]]+i for i in test_list]

            train_labels = [input_parameters['label_dir']+i for i in train_list]
            test_labels = [input_parameters['label_dir']+i for i in test_list]

            dataset_train, dataset_valid = make_training_set(
                'train',
                train_img_paths = train_images,
                train_tar = train_labels,
                valid_img_paths=test_images,
                valid_tar=test_labels,
                parameters= training_parameters
            )

            model = Training_Loop(dataset_train,dataset_valid,training_parameters,nept_run)

            Test_Network(model,dataset_valid,nept_run,training_parameters)

    elif input_parameters['phase']=='retrain':

        # Retraining existing model
        # Create new model version and download previous model 

        training_parameters = parameters['train_parameters']
        # Path to folder containing ground truth images
        if os.path.isdir(input_parameters['label_dir']):
            label_paths = sorted(glob(input_parameters['label_dir']+'*'))
        elif os.path.isfile(input_parameters['label_dir']):
            label_paths = sorted(pd.read_csv(input_parameters['label_dir'])['Paths'].tolist())

        input_image_type = list(input_parameters['image_dir'].keys())

        if input_parameters['type']=='multi':
            
            # Getting multi-input image paths
            image_paths_base = []
            for inp_type in input_image_type:
                image_path_in_type = input_parameters['image_dir'][inp_type]
                if os.path.isdir(image_path_in_type):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'*')))
                elif os.path.isfile(image_path_in_type):
                    image_paths_base.append(sorted(pd.read_csv(image_path_in_type)['Paths'].tolist())) 

            # Getting the image paths as lists of lists (each input type)
            image_paths = [list(i) for i in zip(*image_paths_base)]
        
        elif input_parameters['type']=='single':

            # Getting either DUET or Brightfield images
            image_path_in_type = input_parameters['image_dir'][input_image_type[0]]
            if os.path.isdir(image_path_in_type):
                image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'*'))
            elif os.path.isfile(image_path_in_type):
                image_paths = sorted(pd.read_csv(image_path_in_type)['Paths'].tolist())

        # Now determining which images are used for training and which are used for testing
        if 'k_folds' in training_parameters['train_test_set']:

            kf = KFold(n_splits = int(training_parameters['train_test_set']['k_folds']),shuffle=True)
            k_count = 1

            # Iteratively training and validating model using k-fold CV
            for train_idx, test_idx in kf.split(image_paths):

                print(f'On k-fold: {k_count}')

                train_idx = list(train_idx.astype(int))
                test_idx = list(test_idx.astype(int))

                train_images = [image_paths[i] for i in train_idx]
                test_images = [image_paths[i] for i in test_idx]
                train_labels = [label_paths[i] for i in train_idx]
                test_labels = [label_paths[i] for i in test_idx]

                #k_train_set = pd.DataFrame(data = {'Training_Image_Paths':train_images})
                #k_test_set = pd.DataFrame(data = {'Testing_Image_Paths':test_images})

                nept_run.assign({f'{k_count}_Training_Set':train_images})
                nept_run.assign({f'{k_count}_Testing_Set':test_images})

                train_dataset, validation_dataset = make_training_set(
                    phase = 'train',
                    train_img_paths=train_images,
                    train_tar=train_labels,
                    valid_img_paths=test_images,
                    valid_tar=test_labels,
                    parameters = training_parameters
                )

                model = Training_Loop(
                    dataset_train = train_dataset,
                    dataset_valid = validation_dataset,
                    train_parameters = training_parameters,
                    nept_run=nept_run
                )

                Test_Network(model,validation_dataset,nept_run,training_parameters)

                k_count+=1

        elif 'split' in training_parameters['train_test_split']:

            # Random train/test split with 'split' proportion assigned to training data
            shuffle_idx = np.random.permutation(len(image_paths))

            split_pct = training_parameters['train_test_split']['split']
            train_idx = shuffle_idx[0:floor(split_pct*len(image_paths))]
            val_idx = shuffle_idx[floor(split_pct*len(image_paths)):len(image_paths)]
            
            train_images = [image_paths[i] for i in train_idx]
            test_images = [image_paths[i] for i in val_idx]
            train_labels = [label_paths[i] for i in train_idx]
            test_labels = [label_paths[i] for i in val_idx]

            dataset_train, dataset_valid = make_training_set(
                'train',
                train_img_paths=train_images,
                train_tar = train_labels,
                valid_img_paths=test_images,
                valid_tar = test_labels,
                parameters=training_parameters
            )

            model = Training_Loop(dataset_train, dataset_valid, training_parameters,nept_run)
            Test_Network(model, dataset_valid, nept_run, training_parameters)

        elif 'training' in training_parameters['train_test_split']:
            
            # Passing a file containing the names of files in the training and testing sets
            # Adding the image_dir afterwards (dependent on whether it's a multi or single input)
            train_list = pd.read_csv(training_parameters['train_test_split']['training'])['Image_Names'].tolist()
            test_list = pd.read_csv(training_parameters['train_test_split']['testing'])['Image_Names'].tolist()

            if input_parameters['type']=='multi':
                train_images = []
                for tr in train_list:
                    # Getting each type of input
                    tr_list = []
                    for in_type in input_image_type:
                        tr_list.append(input_parameters['image_dir'][in_type]+tr)
                    # Adding lists of inputs to main train images list
                    train_images.append(tr_list)
                
                test_images = []
                for te in test_list:
                    te_list = []
                    for in_type in input_image_type:
                        te_list.append(input_parameters['image_dir'][in_type]+te)
                    
                    test_images.append(te_list)

            elif input_parameters['type']=='single':
                train_images = [input_parameters['image_dir'][input_image_type[0]]+i for i in train_list]
                test_images = [input_parameters['image_dir'][input_image_type[0]]+i for i in test_list]

            train_labels = [input_parameters['label_dir']+i for i in train_list]
            test_labels = [input_parameters['label_dir']+i for i in test_list]

            dataset_train, dataset_valid = make_training_set(
                'train',
                train_img_paths = train_images,
                train_tar = train_labels,
                valid_img_paths=test_images,
                valid_tar=test_labels,
                parameters= training_parameters
            )

            model = Training_Loop(dataset_train,dataset_valid,training_parameters,nept_run)

            Test_Network(model,dataset_valid,nept_run,training_parameters)

    elif input_parameters['phase']=='test':

        input_image_type = list(input_parameters['image_dir'].keys())

        if 'label_dir' in input_parameters:
            if os.path.isdir(input_parameters['label_dir']):
                label_paths = glob(input_parameters['label_dir']+'*')
            elif os.path.isfile(input_parameters['label_dir']):
                label_paths = sorted(pd.read_csv(input_parameters['label_dir'])['Paths'].tolist())
        else:
            label_paths = []
        
        if input_parameters['type']=='multi':

            image_paths_base = []
            for inp_type in input_image_type:
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'*')))
                elif os.path.isfile(input_parameters['image_dir'][inp_type]):
                    image_paths_base = sorted(pd.read_csv(input_parameters['image_dir'][inp_type])['Paths'].tolist())

            image_paths = [list(i) for i in zip(*image_paths_base)]
        elif input_parameters['type']=='single':
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'*'))
            elif os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist())

        # Loading model and data
        if 'neptune' in input_parameters:
            model_version = neptune.init_model(
                project = nept_params['project'],
                with_id = input_parameters['model'],
                mode = 'async',
                api_token = nept_api_token
            )

            if 'model_file' not in input_parameters:
                # Download the model from the model version
                if not os.path.exists(input_parameters['output_dir']+'/model/'):
                    os.makedirs(input_parameters['output_dir']+'/model/')

                model_version['model_file'].download(input_parameters['output_dir']+'/model')
                model_file = os.listdir(input_parameters['output_dir']+'/model/')[0]
            else:
                model_file = input_parameters['model_file']

            all_model_metadata = model_version.get_structure()
            prep_keys = all_model_metadata['preprocessing'].keys()
            preprocessing = {}
            for p in prep_keys:
                preprocessing[p] = model_version[f'preprocessing/{p}'].fetch()

            model_deets_keys = all_model_metadata['model_details'].keys()
            model_details = {}
            for m in model_deets_keys:
                model_details[m] = model_version[f'model_details/{m}'].fetch()

        else:

            model_file = input_parameters['model_file']
            preprocessing = input_parameters['preprocessing']
            model_details = input_parameters['model_details']

        input_parameters['model_details'] = model_details
        input_parameters['preprocessing'] = preprocessing

        nothin, dataset_test = make_training_set(
            'test',
            None,
            None,
            image_paths,
            label_paths,
            input_parameters
        )

        Test_Network(model_file, dataset_test, nept_run, input_parameters)

    elif input_parameters['phase']=='cluster':

        # Inputs are the same as testing but instead of predicting, just generating some clustering/relative clustering of latent features
        input_image_type = list(input_parameters['image_dir'].keys())
        label_paths = []
        
        if input_parameters['type']=='multi':

            image_paths_base = []
            for inp_type in input_image_type:
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'*')))
                elif os.path.isfile(input_parameters['image_dir'][inp_type]):
                    image_paths_base = pd.read_csv(input_parameters['image_dir'][inp_type])['Paths'].tolist()

            image_paths = [list(i) for i in zip(*image_paths_base)]
        elif input_parameters['type']=='single':
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'*'))
            elif os.path.isfile(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist()


        # Loading model and data
        if 'neptune' in input_parameters:
            model_version = neptune.init_model(
                project = nept_params['project'],
                with_id = input_parameters['model'],
                mode = 'async',
                api_token = nept_api_token
            )

            if 'model_file' not in input_parameters:
                # Download the model from the model version
                if not os.path.exists(input_parameters['output_dir']+'/model/'):
                    os.makedirs(input_parameters['output_dir']+'/model/')

                model_version['model_file'].download(input_parameters['output_dir']+'/model')
                model_file = os.listdir(input_parameters['output_dir']+'/model/')[0]
            else:
                model_file = input_parameters['model_file']

            all_model_metadata = model_version.get_structure()
            prep_keys = all_model_metadata['preprocessing'].keys()
            preprocessing = {}
            for p in prep_keys:
                preprocessing[p] = model_version[f'preprocessing/{p}'].fetch()

            model_deets_keys = all_model_metadata['model_details'].keys()
            model_details = {}
            for m in model_deets_keys:
                model_details[m] = model_version[f'model_details/{m}'].fetch()

        else:

            model_file = input_parameters['model_file']
            preprocessing = input_parameters['preprocessing']
            model_details = input_parameters['model_details']

        input_parameters['model_details'] = model_details
        input_parameters['preprocessing'] = preprocessing

        nothin, dataset_test = make_training_set(
            'test',
            None,
            None,
            image_paths,
            label_paths,
            input_parameters
        )

        cluster_object = Clusterer(dataset_test,model_file,input_parameters)

        # Uploading data to model_version metadata (if specified)
        upload_to_model_version = True
        if upload_to_model_version:
            model_version['UMAP_coordinates'].upload(input_parameters['output_dir']+'Extracted_UMAP_Coordinates.csv')
            model_version['UMAP_Plot'].upload(input_parameters['output_dir']+'Output_UMAP_Plot.png')


if __name__=='__main__':
    main()
