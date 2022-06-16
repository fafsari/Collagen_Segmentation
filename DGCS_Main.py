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

from sklearn.model_selection import KFold

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import optuna

from Input_Pipeline import *
from DGCS_Train import Training_Loop
from DGCS_Test import Test_Network


# Changing up from sys.argv to reading a specific set of input parameters
parameters_file = sys.argv[1]

parameters = json.load(open(parameters_file))
input_parameters = parameters['input_parameters']

phase = input_parameters['phase']
image_dir = input_parameters['image_dir']+'/'

if phase == 'train':
    label_dir = input_parameters['label_dir']+'/'
    k_folds = input_parameters['k_folds']
    train_parameters = parameters['training_parameters']
    test_parameters = parameters['testing_parameters']

    ann_classes = train_parameters['ann_classes']

elif phase == 'test':
    model_file = input_parameters['model_file']
    test_parameters = parameters['testing_parameters']
    
    ann_classes = test_parameters['ann_classes']

elif phase == 'optimize':
    label_dir = input_parameters['label_dir']+'/'
    train_parameters = parameters['training_parameters']
    test_parameters = parameters['testing_parameters']

    ann_classes = train_parameters['ann_classes']

    hyp_opts = train_parameters['hyp_opts']

# Adding this option here for if using binary input masks or probabilistic (grayscale)
target_type = input_parameters['target_type']

output_dir = input_parameters['output_dir']

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

model_dir = output_dir+'/models/'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

nept_run = neptune.init(
    project = 'spborder/AssortedSegmentations',
    source_files = ['**/*'],
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMjE3MzBiOS1lOGMwLTRmZjAtOGUyYS0yMGFlMmM4ZTRkMzMifQ==')

nept_run['image_dir'] = image_dir
nept_run['label_dir'] = label_dir
nept_run['output_dir'] = output_dir
nept_run['model_dir'] = model_dir
nept_run['Classes'] = ann_classes
nept_run['Target_Type'] = target_type


if phase == 'train':
    if not os.path.isdir(image_dir):
        
        all_paths = pd.read_csv(image_dir)
        image_paths = all_paths[0].tolist()
        label_paths = all_paths[1].tolist()
    else:
        image_paths = glob(image_dir+'*')
        label_paths_base = glob(label_dir+'*')
        
        # For when image and label paths don't line up
        label_paths = []
        label_names = [i.split('/')[-1] for i in label_paths_base]
        for j in range(0,len(image_paths)):
            image_name = image_paths[j].split('/')[-1]
            label_paths.append(label_paths_base[label_names.index(image_name)])

    # Determining whether or not doing k-fold CV and proceeding to training loop
    if int(k_folds)==1:
        
        # shuffling image and target paths
        shuffle_idx = np.random.permutation(len(image_paths))
        
        train_idx = shuffle_idx[0:floor(0.8*len(image_paths))]    
        val_idx = shuffle_idx[floor(0.8*len(image_paths)):len(image_paths)]
            
        train_img_paths = [image_paths[i] for i in train_idx]
        valid_img_paths = [image_paths[i] for i in val_idx]
        
        train_tar = [label_paths[i] for i in train_idx]
        valid_tar = [label_paths[i] for i in val_idx]
        

        print(f'Training image paths: {train_img_paths[0:4]}')
        print(f'Training mask paths: {train_tar[0:4]}')

        nept_run['N_Training'] = len(train_img_paths)
        nept_run['N_Valid'] = len(valid_img_paths)
        
        dataset_train, dataset_valid = make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar,target_type)
        
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
    
            dataset_train, dataset_valid = make_training_set(phase, X_train, y_train, X_test, y_test,target_type)
            
            model = Training_Loop(ann_classes, dataset_train, dataset_valid,model_dir, output_dir,target_type, train_parameters, nept_run)
            
            Test_Network(ann_classes, model, dataset_valid, output_dir,nept_run, test_parameters, target_type)
            k_count += 1

elif phase == 'test':
    
    image_paths = glob(image_dir+'*')
    label_paths = glob(label_dir+'*')

    model = torch.load(model_file)
    model = model.to(torch.device('cuda') if torch.cuda.is_available() else 'cpu')

    valid_img_paths = image_paths
    valid_tar = image_paths
    
    nothin, dataset_test = make_training_set(phase, None, None, valid_img_paths, valid_tar, target_type)
    
    Test_Network(ann_classes, model, dataset_test, output_dir,nept_run, test_parameters, target_type)
    
elif phase == 'optimize':
    if not os.path.isdir(image_dir):
        
        all_paths = pd.read_csv(image_dir)
        image_paths = all_paths[0].tolist()
        label_paths = all_paths[1].tolist()
    else:
        image_paths = glob(image_dir+'*')
        label_paths_base = glob(label_dir+'*')
        
        # For when image and label paths don't line up
        label_paths = []
        label_names = [i.split('/')[-1] for i in label_paths_base]
        for j in range(0,len(image_paths)):
            image_name = image_paths[j].split('/')[-1]
            label_paths.append(label_paths_base[label_names.index(image_name)])

    model_params = {
        'architecture':{'type',str,['Unet++','Unet','MAnet','DeepLabV3','DeepLabV3+']},
        'lr':{'type':float,'min':0.00001,'max':0.001},
        'epoch_num':{'type':int,'min':70,'max':200},
        'loss':{'type':str,'opts':['MSE','L1','L2']}
    }

    def objective(trial, training_parameters, model_params, model_dir):

        # Iterating through model_params
        for key,val in model_params.items():
            if val['type']==int:
                training_parameters[key] = trial.suggest_int(key,val['min'],val['max'])
            elif model_params[key]['type']==float:
                training_parameters[key] = trial.suggest_float(key,val['min'],val['max'])
            elif model_params[key]['type']==str:
                training_parameters[key] = trial.suggest_categorical(key,val['opts'])

        if not os.path.exists(model_dir+'/Trial_Parameters.csv'):
            prev_trials = pd.DataFrame(training_parameters)
            prev_trials.to_csv(model_dir+'/Trial_Parameters.csv')
        else:
            prev_trials = pd.read_csv(model_dir+'/Trial_Parameters.csv',index_col = 0)
            prev_trials = pd.concat([prev_trials,pd.DataFrame(training_parameters)],axis = 0)

            prev_trials.to_csv(model_dir+'/Trial_Parameters.csv')

        # shuffling image and target paths
        shuffle_idx = np.random.permutation(len(image_paths))
        
        train_idx = shuffle_idx[0:floor(0.8*len(image_paths))]    
        val_idx = shuffle_idx[floor(0.8*len(image_paths)):len(image_paths)]
            
        train_img_paths = [image_paths[i] for i in train_idx]
        valid_img_paths = [image_paths[i] for i in val_idx]
        
        train_tar = [label_paths[i] for i in train_idx]
        valid_tar = [label_paths[i] for i in val_idx]

        #print(f'Training image paths: {train_img_paths[0:4]}')
        #print(f'Training mask paths: {train_tar[0:4]}')

        nept_run['N_Training'] = len(train_img_paths)
        nept_run['N_Valid'] = len(valid_img_paths)
        
        dataset_train, dataset_valid = make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar,target_type)
        
        model = Training_Loop(ann_classes, dataset_train, dataset_valid,model_dir, output_dir, target_type, train_parameters, nept_run)
        
        Test_Network(ann_classes, model, dataset_valid, output_dir, nept_run, test_parameters, target_type)

        # Reading the metrics
        performance_vals = pd.read_csv(output_dir+'Testing_Output/Test_Metrics.csv')
        performance_means = performance_vals.mean(axis = 0)
        performance_stds = performance_vals.std(axis = 0)
        
        # If the target type is 'nonbinary' then there will only be one metric recorded
        if target_type=='nonbinary':
            performance_sum = performance_means+performance_stds
        else:
            performance_sum = performance_means.sum()+performance_stds.sum()

        return performance_sum


    if target_type=='nonbinary':
        study = optuna.create_study(direction = 'minimize', study_name = 'Collagen_Segmentation')
    else:
        study = optuna.create_study(direction = 'maximize',study_name = 'Collagen_Segmentation')
    
    neptune_callback = optuna_utils.NeptuneCallback(nept_run)
    study.optimize(lambda trial: objective(trial,train_parameters,model_params,model_dir),n_trials = 500, callbacks=[neptune_callback])
    nept_run.stop()
    

