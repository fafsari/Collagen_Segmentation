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

from sklearn.model_selection import KFold

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import optuna

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
input_parameters = parameters['input_parameters']

phase = input_parameters['phase']
if 'image_dir' in input_parameters:
    image_dir = input_parameters['image_dir']+'/'
elif 'f_image_dir' in input_parameters:
    f_image_dir = input_parameters['f_image_dir']+'/'
    b_image_dir = input_parameters['b_image_dir']+'/'

print(f'Phase is: {phase}')

if phase == 'train':
    train_parameters = parameters['training_parameters']
    test_parameters = parameters['testing_parameters']

    if 'color_transform' in input_parameters:
        train_parameters['color_transform'] = input_parameters['color_transform']
        test_parameters['color_transform'] = input_parameters['color_transform']
        train_parameters['norm_mean'] = input_parameters['norm_mean']
        test_parameters['norm_mean'] = input_parameters['norm_mean']
        train_parameters['norm_std'] = input_parameters['norm_std']
        test_parameters['norm_std'] = input_parameters['norm_std']
    else:
        test_parameters['color_transform'] = []
        test_parameters['norm_mean'] = []
        test_parameters['norm_std'] = []
        train_parameters['color_transform'] = []
        train_parameters['norm_mean'] = []
        train_parameters['norm_std'] = []

    if train_parameters['multi_task']:
        label_bin_dir = input_parameters['label_bin_dir']+'/'
        label_reg_dir = input_parameters['label_reg_dir']+'/'
    else:
        label_dir = input_parameters['label_dir']+'/'

    k_folds = input_parameters['k_folds']

    ann_classes = train_parameters['ann_classes']

elif phase == 'test':
    model_file = input_parameters['model_file']
    test_parameters = parameters['testing_parameters']

    # Adding parameters for Reinhard color normalization
    if 'color_transform' in input_parameters:
        test_parameters['color_transform'] = input_parameters['color_transform']
        test_parameters['norm_mean'] = input_parameters['norm_mean']
        test_parameters['norm_std'] = input_parameters['norm_std']
    else:
        test_parameters['color_transform'] = []
        test_parameters['norm_mean'] = []
        test_parameters['norm_std'] = []

    if test_parameters['multi_task']:
        label_bin_dir = input_parameters['label_bin_dir']+'/'
        label_reg_dir = input_parameters['label_reg_dir']+'/'
    elif 'label_dir' in input_parameters:
        label_dir = input_parameters['label_dir']
    
    ann_classes = test_parameters['ann_classes']

# Adding this option here for if using binary input masks or probabilistic (grayscale)
target_type = input_parameters['target_type']

output_dir = input_parameters['output_dir']

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

model_dir = output_dir+'/models/'
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

try:
    nept_run['image_dir'] = image_dir
except:
    nept_run['f_image_dir'] = f_image_dir
    nept_run['b_image_dir'] = b_image_dir
try:
    if 'label_dir' in input_parameters:
        nept_run['label_dir'] = label_dir
except:
    nept_run['label_bin_dir'] = label_bin_dir
    nept_run['label_reg_dir'] = label_reg_dir
nept_run['output_dir'] = output_dir
nept_run['model_dir'] = model_dir
nept_run['Classes'] = ann_classes
nept_run['Target_Type'] = target_type

if phase == 'train':

    if 'image_dir' not in input_parameters:

        f_image_paths_base = glob(f_image_dir+'*')
        b_image_paths_base = glob(b_image_dir+'*')

        label_paths_base = glob(label_dir+'*')

        f_img_names = [i.split('/')[-1] for i in f_image_paths_base]
        b_img_names = [i.split('/')[-1] for i in b_image_paths_base]
        label_names = [i.split('/')[-1] for i in label_paths_base]

        # Sorting by f image names (arbitrarily)
        image_paths = []
        label_paths = []
        for idx,f in enumerate(f_img_names):
            b_path = b_image_paths_base[b_img_names.index(f)]

            image_paths.append([f_image_paths_base[idx],b_path])
            l_path = label_paths_base[label_names.index(f)]
            label_paths.append(l_path)

    else:
        image_paths = glob(image_dir+'*')

        if not train_parameters['multi_task']:
            label_paths_base = glob(label_dir+'*')
            
            # For when image and label paths don't line up
            label_paths = []
            label_names = [i.split('/')[-1] for i in label_paths_base]
            for j in range(0,len(image_paths)):
                image_name = image_paths[j].split('/')[-1]
                label_paths.append(label_paths_base[label_names.index(image_name)])
        else:
            label_bin_paths_base = glob(label_bin_dir+'*')
            label_reg_paths_base = glob(label_reg_dir+'*')

            label_bin_paths = []
            label_bin_names = [i.split('/')[-1] for i in label_bin_paths_base]
            label_reg_paths = []
            label_reg_names = [i.split('/')[-1] for i in label_reg_paths_base]

            for j in range(0,len(image_paths)):
                image_name = image_paths[j].split('/')[-1]
                label_bin_paths.append(label_bin_paths_base[label_bin_names.index(image_name)])
                label_reg_paths.append(label_reg_paths_base[label_reg_names.index(image_name)])

    # Determining whether or not doing k-fold CV and proceeding to training loop
    if int(k_folds)==1:
        
        # If a specific set of files is mentioned for training and testing
        if 'train_set' in input_parameters:
            train_df = pd.read_csv(input_parameters['train_set'])
            test_df = pd.read_csv(input_parameters['test_set'])
            train_img_paths = train_df['Training_Image_Paths'].tolist()
            valid_img_paths = test_df['Testing_Image_Paths'].tolist()

            train_tar = [i.replace(input_parameters['image_dir'],input_parameters['label_dir']) for i in train_img_paths]
            valid_tar = [i.replace(input_parameters['image_dir'],input_parameters['label_dir']) for i in valid_img_paths]

            nept_run['Training_Set'].upload(neptune.types.File.as_html(train_df))
            nept_run['Testing_Set'].upload(neptune.types.File.as_html(test_df))

        else:
            # shuffling image and target paths
            shuffle_idx = np.random.permutation(len(image_paths))
            
            train_idx = shuffle_idx[0:floor(0.8*len(image_paths))]    
            val_idx = shuffle_idx[floor(0.8*len(image_paths)):len(image_paths)]
                
            train_img_paths = [image_paths[i] for i in train_idx]
            valid_img_paths = [image_paths[i] for i in val_idx]
        
            if train_parameters['multi_task']:
                train_bin_tar = [label_bin_paths[i] for i in train_idx]
                train_reg_tar = [label_reg_paths[i] for i in train_idx]
                valid_bin_tar = [label_bin_paths[i] for i in val_idx]
                valid_reg_tar = [label_reg_paths[i] for i in val_idx]

            else:
                train_tar = [label_paths[i] for i in train_idx]
                valid_tar = [label_paths[i] for i in val_idx]

        nept_run['N_Training'] = len(train_img_paths)
        nept_run['N_Valid'] = len(valid_img_paths)
        
        if train_parameters['multi_task']:
            # slightly different inputs into the multi-task training set function compared to the regular
            dataset_train, dataset_valid = make_multi_training_set(phase, train_img_paths, train_bin_tar, train_reg_tar, valid_img_paths, valid_bin_tar, valid_reg_tar)
        else:
            dataset_train, dataset_valid = make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar,target_type,train_parameters)
        
        model = Training_Loop(ann_classes, dataset_train, dataset_valid,model_dir, output_dir, target_type, train_parameters, nept_run)
        
        Test_Network(ann_classes, model, dataset_valid, output_dir, nept_run, test_parameters, target_type)
    
    else:
        
        # Splitting dataset into k-folds
        kf = KFold(n_splits = int(k_folds), shuffle = True)
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
    
            if train_parameters['multi_task']:
                dataset_train, dataset_valid = make_multi_training_set(phase, X_train, y_train, X_test, y_test, target_type, train_parameters)
            else:
                dataset_train, dataset_valid = make_training_set(phase, X_train, y_train, X_test, y_test,target_type,train_parameters)
            
            model = Training_Loop(ann_classes, dataset_train, dataset_valid,model_dir, output_dir,target_type, train_parameters, nept_run)
            
            Test_Network(ann_classes, model, dataset_valid, output_dir,nept_run, test_parameters, target_type)
            k_count += 1

elif phase == 'test':
    
    if 'image_dir' in input_parameters:
        image_paths = glob(image_dir+'*')
        if 'label_dir' in input_parameters:
            label_paths = glob(label_dir+'*')
    else:
        f_image_paths_base = glob(f_image_dir+'*')
        b_image_paths_base = glob(b_image_dir+'*')

        label_paths_base = glob(label_dir+'*')

        f_img_names = [i.split('/')[-1] for i in f_image_paths_base]
        b_img_names = [i.split('/')[-1] for i in b_image_paths_base]
        label_names = [i.split('/')[-1] for i in label_paths_base]

        # Sorting by f image names (arbitrarily)
        image_paths = []
        label_paths = []
        for idx,f in enumerate(f_img_names):
            b_path = b_image_paths_base[b_img_names.index(f)]

            image_paths.append([f_image_paths_base[idx],b_path])
            l_path = label_paths_base[label_names.index(f)]
            label_paths.append(l_path)


    valid_img_paths = image_paths
    valid_tar = image_paths
    
    nothin, dataset_test = make_training_set(phase, None, None, valid_img_paths, valid_tar, target_type,test_parameters)
    
    Test_Network(ann_classes, model_file, dataset_test, output_dir,nept_run, test_parameters, target_type)
