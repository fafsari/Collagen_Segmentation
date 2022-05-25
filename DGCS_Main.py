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

from sklearn.model_selection import KFold

import neptune.new as neptune

from Input_Pipeline import *
from DGCS_Train import Training_Loop
from DGCS_Test import Test_Network


# Order of inputs = 
# Train ('train') or Test ('test') 
# Full image directory (relative to where this script is located or change base_dir variable)
# Labeled images directory (all images don't have to be annotated)

## Either of these can also be subbed out for '.csv' files listing paths to specific images/labels

# Number of CV (1 = normal train/test split (0.8,0.2 resp.), >1 = k-fold CV)

# output is defaulted to /reports/
# models saved to /models/

phase = sys.argv[1]

if phase == 'train':
    
    image_dir = sys.argv[2]+'/'
    label_dir = sys.argv[3]+'/'
    k_folds = sys.argv[4]

else:
    image_dir = sys.argv[2]+'/'
    label_dir = sys.argv[3]+'/'
    
# Adding this option here for if using binary input masks or probabilistic (grayscale)
if len(sys.argv)==4:
    target_type = sys.argv[4]
else:
    target_type = 'binary'


# Modify classes according to annotations
#ann_classes = ['background','u_space','tuft']
#ann_classes = ['background','mesangium','luminal_space','nuclei']
ann_classes = ['background','collagen']


# Comment out to use main project directory as base
#base_dir = ''

# Defining input directories
if 'base_dir' not in locals():
    base_dir = os.getcwd()
    
model_dir = '/'.join(base_dir.split('/')[0:-1])+'/models/'
data_dir = '/'.join(base_dir.split('/')[0:-1])+'/data/'
output_dir = '/'.join(base_dir.split('/')[0:-1])+'/reports/'


nept_run = neptune.init(
    project = 'spborder/AssortedSegmentations',
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwMjE3MzBiOS1lOGMwLTRmZjAtOGUyYS0yMGFlMmM4ZTRkMzMifQ==')

nept_run['image_dir'] = data_dir+image_dir
nept_run['label_dir'] = data_dir+label_dir
nept_run['output_dir'] = output_dir
nept_run['Classes'] = ann_classes
nept_run['Target_Type'] = target_type


if phase == 'train':
    if not os.path.isdir(data_dir+image_dir):
        
        all_paths = pd.read_csv(data_dir+image_dir)
        image_paths = all_paths[0].tolist()
        label_paths = all_paths[1].tolist()
    else:
        image_paths = glob(data_dir+image_dir+'*')
        label_paths = glob(data_dir+label_dir+'*')
        

        """
        if image_paths != label_paths:
            # Optional replacement for labeled images that don't have the same name as the corresponding images
            label_to_img_paths = [i.replace('_ann','') for i in label_paths]
            # Making sure the same files are in each list
            label_filenames = [i.split('/')[-1] for i in label_to_img_paths]
            image_filenames = [i.split('/')[-1] for i in image_paths]
        
        
            image_paths = [image_paths[image_filenames.index(i)] for i in label_filenames]
        """
        
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
        
        nept_run['N_Training'] = len(train_img_paths)
        nept_run['N_Valid'] = len(valid_img_paths)
        
        dataset_train, dataset_valid = make_training_set(phase,train_img_paths, train_tar, valid_img_paths, valid_tar,ann_classes,target_type)
        
        model = Training_Loop(ann_classes, dataset_train, dataset_valid,model_dir, output_dir, nept_run)
        
        Test_Network(ann_classes, model_dir, dataset_valid, output_dir, nept_run)
    
    else:
        
        # Splitting dataset into k-folds
        kf = KFold(n_splits = int(k_folds), shuffle = True)
        k_count = 0
        for train_idx, test_idx in kf.split(image_paths):
            
            print('\n------------------------------------------------------')
            print('------------------------------------------------------')
            print('On k-fold #: {}'.format(k_count+1))
            print('-------------------------------------------------------')
            print('-------------------------------------------------------\n')
            
            X_train, X_test = image_paths[train_idx], image_paths[test_idx]
            y_train, y_test = label_paths[train_idx], label_paths[test_idx]
    
            dataset_train, dataset_valid = make_training_set(phase, X_train, y_train, X_test, y_test,ann_classes,target_type)
            
            model = Training_Loop(ann_classes, dataset_train, dataset_valid,model_dir, output_dir,target_type,nept_run)
            
            Test_Network(ann_classes, model_dir, dataset_valid, output_dir,target_type,nept_run)
    
elif phase == 'test':
    
    image_paths = glob(data_dir+image_dir+'*')
    label_paths = glob(data_dir+label_dir+'*')
    """
    # Optional replacement for labeled images that don't have the same name as the corresponding images
    label_to_img_paths = [i.replace('_ann','') for i in label_paths]
    # Making sure the same files are in each list
    label_filenames = [i.split('/')[-1] for i in label_to_img_paths]
    image_filenames = [i.split('/')[-1] for i in image_paths]
    
    image_paths = [image_paths[image_filenames.index(i)] for i in label_filenames]
    
    """
    valid_img_paths = image_paths
    valid_tar = label_paths
    
    nothin, dataset_test = make_training_set(phase, None, None, valid_img_paths, valid_tar, ann_classes, target_type)
    
    Test_Network(ann_classes, model_dir, dataset_test, output_dir,nept_run)
    
    

