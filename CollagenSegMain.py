# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 08:39:49 2021

@author: spborder


DGCS main file

1) Input pipeline
    1a) Splitting up training/validation/testing
    1b) Option for 5-fold CV
2) Initializing training loop
    2a) Hyperparameter tuning?
    2b) Performance on validation set
3) Performance on test set
4) Visualizations

"""

import os
import sys
import pandas as pd
import numpy as np
from glob import glob
from math import floor
import json
from MultiTaskModel import make_multi_training_set
from Organize_Inputs import organize_parameters

from sklearn.model_selection import KFold

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

from Input_Pipeline import *
from CollagenSegTrain import Training_Loop
from CollagenSegTest import Test_Network


nept_run = neptune.init(
    project = 'spborder/AssortedSegmentations',
    source_files = ['**/*.py','*.json'],
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMjE3MzBiOS1lOGMwLTRmZjAtOGUyYS0yMGFlMmM4ZTRkMzMifQ==')


# Changing up from sys.argv to reading a specific set of input parameters
parameters_file = sys.argv[1]

parameters = json.load(open(parameters_file))

train_parameters, test_parameters, phase = organize_parameters(parameters,nept_run)

if phase == 'train':

    if 'image_dir' not in train_parameters:

        f_image_paths_base = glob(train_parameters['f_image_dir']+'*')
        b_image_paths_base = glob(train_parameters['b_image_dir']+'*')

        label_paths_base = glob(train_parameters['label_dir']+'*')

        f_img_names = [i.split('/')[-1] for i in f_image_paths_base]
        b_img_names = [i.split('/')[-1] for i in b_image_paths_base]
        label_names = [i.split('/')[-1] for i in label_paths_base]

        # Sorting by f image names (arbitrarily)
        image_paths = []
        label_paths = []
        for idx,f in enumerate(f_img_names):
            try:
                b_path = b_image_paths_base[b_img_names.index(f)]

                image_paths.append([f_image_paths_base[idx],b_path])
                l_path = label_paths_base[label_names.index(f)]
                label_paths.append(l_path)
            except ValueError:
                continue
    else:
        image_paths = glob(train_parameters['image_dir']+'*')

        label_paths_base = glob(train_parameters['label_dir']+'*')
        
        # For when image and label paths don't line up
        label_paths = []
        label_names = [i.split('/')[-1] for i in label_paths_base]
        for j in range(0,len(image_paths)):
            image_name = image_paths[j].split('/')[-1]
            label_paths.append(label_paths_base[label_names.index(image_name)])


    # Determining whether or not doing k-fold CV and proceeding to training loop
    if int(train_parameters['k_folds'])==1:
        
        # If a specific set of files is mentioned for training and testing
        if 'train_set' in train_parameters:
            train_df = pd.read_csv(train_parameters['train_set'])
            test_df = pd.read_csv(train_parameters['test_set'])
            if 'Training_Image_Paths' in train_df.columns.values.tolist():
                train_img_paths = train_df['Training_Image_Paths'].tolist()
                valid_img_paths = test_df['Testing_Image_Paths'].tolist()
            elif 'F_Training_Image_Paths' in train_df.columns.values.tolist():
                train_img_paths = (train_df['F_Training_Image_Paths']+train_df['B_Training_Image_Paths']).tolist()
                valid_img_paths = (test_df['F_Training_Image_Paths']+test_df['B_Training_Image_Paths']).tolist()
                
            train_tar = [i.replace(train_parameters['image_dir'],train_parameters['label_dir']) for i in train_img_paths]
            valid_tar = [i.replace(train_parameters['image_dir'],train_parameters['label_dir']) for i in valid_img_paths]

            nept_run['Training_Set'].upload(neptune.types.File.as_html(train_df))
            nept_run['Testing_Set'].upload(neptune.types.File.as_html(test_df))

        else:
            # shuffling image and target paths
            shuffle_idx = np.random.permutation(len(image_paths))
            
            train_idx = shuffle_idx[0:floor(0.8*len(image_paths))]    
            val_idx = shuffle_idx[floor(0.8*len(image_paths)):len(image_paths)]
                
            train_img_paths = [image_paths[i] for i in train_idx]
            valid_img_paths = [image_paths[i] for i in val_idx]

            train_tar = [label_paths[i] for i in train_idx]
            valid_tar = [label_paths[i] for i in val_idx]

        nept_run['N_Training'] = len(train_img_paths)
        nept_run['N_Valid'] = len(valid_img_paths)
        

        dataset_train, dataset_valid = make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar, train_parameters)
        dataset_train.normalize_cache(np.mean(dataset_train.image_means,axis=0),np.mean(dataset_train.image_stds,axis=0))
        dataset_valid.normalize_cache(np.mean(dataset_train.image_means,axis=0),np.mean(dataset_train.image_stds,axis=0))

        model = Training_Loop(dataset_train, dataset_valid, train_parameters, nept_run)
        
        Test_Network(model, dataset_valid, nept_run, test_parameters)
    
    else:
        
        # Splitting dataset into k-folds
        kf = KFold(n_splits = int(train_parameters['k_folds']), shuffle = True)
        k_count = 1
        for train_idx, test_idx in kf.split(image_paths):
            
            print('\n------------------------------------------------------')
            print('------------------------------------------------------')
            print('On k-fold #: {}'.format(k_count))
            print('-------------------------------------------------------')
            print('-------------------------------------------------------\n')

            train_idx = list(train_idx.astype(int))
            test_idx = list(test_idx.astype(int))
            
            X_train = [image_paths[i] for i in train_idx]
            X_test = [image_paths[i] for i in test_idx]
            y_train = [label_paths[i] for i in train_idx]
            y_test = [label_paths[i] for i in test_idx]

            test_parameters['current_k_fold'] = k_count
            train_parameters['current_k_fold'] = k_count

            # Saving k_fold training and testing set
            k_training_set = pd.DataFrame(data = {'Training_Image_Paths':X_train})
            k_testing_set = pd.DataFrame(data = {'Testing_Image_Paths':X_test})

            nept_run[f'{k_count}_Training_Set'].upload(neptune.types.File.as_html(k_training_set))
            nept_run[f'{k_count}_Testing_Set'].upload(neptune.types.File.as_html(k_testing_set))
    

            dataset_train, dataset_valid = make_training_set(phase, X_train, y_train, X_test, y_test,train_parameters)
            
            model = Training_Loop(dataset_train, dataset_valid, train_parameters, nept_run)
            
            Test_Network(model, dataset_valid, nept_run, test_parameters)
            k_count += 1

elif phase == 'test':
    
    if 'image_dir' in test_parameters:
        image_paths = glob(test_parameters['image_dir']+'*')
        if 'label_dir' in test_parameters:
            label_paths = glob(test_parameters['label_dir']+'*')
        else:
            label_paths = []
    else:
        f_image_paths_base = glob(test_parameters['f_image_dir']+'*')
        b_image_paths_base = glob(test_parameters['b_image_dir']+'*')

        if 'label_dir' in test_parameters:
            label_paths_base = glob(test_parameters['label_dir']+'*')
            label_names = [i.split('/')[-1] for i in label_paths_base]

        else:
            label_paths_base = []

        f_img_names = [i.split('/')[-1] for i in f_image_paths_base]
        b_img_names = [i.split('/')[-1] for i in b_image_paths_base]

        # Sorting by f image names (arbitrarily)
        image_paths = []
        label_paths = []
        for idx,f in enumerate(f_img_names):
            try:
                b_path = b_image_paths_base[b_img_names.index(f)]

                image_paths.append([f_image_paths_base[idx],b_path])

                if len(label_paths_base)>0:
                    l_path = label_paths_base[label_names.index(f)]
                    label_paths.append(l_path)
            except ValueError:
                print(f'{f} not found')

    valid_img_paths = image_paths
    valid_tar = label_paths
    
    nothin, dataset_test = make_training_set(phase, None, None, valid_img_paths, valid_tar,test_parameters)

    # Applying normalization if values are provided
    if 'image_means' in test_parameters:
        dataset_test.normalize_cache(test_parameters['image_means'],test_parameters['image_stds'])
    
    Test_Network(test_parameters['model_file'], dataset_test, nept_run, test_parameters)
