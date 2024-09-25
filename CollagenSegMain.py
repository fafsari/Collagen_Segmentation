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
import joblib
from math import ceil
from sklearn.model_selection import KFold
import torch
import time
import neptune

from Input_Pipeline import *
from CollagenSegTrain import Training_Loop
from CollagenSegTrainDAgmm import TrainingADDA_Loop
from CollagenSegTrain import Training_Loop
from CollagenSegTest import Test_Network
from CollagenCluster import Clusterer
# from utils import save_train_test

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

def check_image_bytes(image_path_list,lower_threshold = None, upper_threshold = None):
    # Checking if a path contains an image with greater than a set threshold of bytes
    # In this case it looks like all the images with less than ~40kb just contain background

    passed_list = []
    if lower_threshold is None and upper_threshold is None:
        return image_path_list

    for i in image_path_list:
        file_size = os.path.getsize(i)
        if not lower_threshold is None:
            if file_size>=lower_threshold:
                if not upper_threshold is None:
                    if file_size<=upper_threshold:
                        passed_list.append(i)
                else:
                    passed_list.append(i)
        else:
            if not upper_threshold is None:
                if file_size<=upper_threshold:
                    passed_list.append(i)
            else:
                passed_list.append(i)

    return passed_list

def check_duplicate(image_path_list,output_path):

    passed_list = []
    for i in image_path_list:
        image_name = i.split(os.sep)[-1]
        if not os.path.exists(output_path+f"Testing_Output/{image_name.replace('.jpg','_prediction.tif')}"):
            passed_list.append(i)
    
    return passed_list


def main():
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    if device == 'cuda':
        # Total memory reserved by the allocator (in bytes)
        reserved_memory = torch.cuda.memory_reserved(device)
        
        # Total memory allocated by tensors (in bytes)
        allocated_memory = torch.cuda.memory_allocated(device)
        
        print(f"Reserved memory: {reserved_memory / 1024**2:.2f} MB")
        print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
        
    # Changing up from sys.argv to reading a specific set of input parameters
    parameters_file = sys.argv[1]

    parameters = json.load(open(parameters_file))

    # Getting input parameters and neptune-specific parameters (if specified)
    input_parameters = parameters['input_parameters']
    
    # Get mode
    mode = parameters["mode"]

    if 'neptune' in input_parameters:
        nept_params = input_parameters['neptune']

        try:
            nept_api_token = os.environ.get('NEPTUNE_API_TOKEN')
            print('NEPTUNE_API_TOKEN was set successfully!', nept_api_token)
        except:
            print('Error! export NEPTUNE_API_TOKEN as an environmetn variable.')
            exit(-1)
        
        nept_run = neptune.init_run(
            project = nept_params['project'],
            source_files = nept_params['source_files'],
            # api_token = nept_api_token,
            api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhODE0Mzc0MC03MmQ5LTQ0NjYtOGU2OC1jZDQ1NGZhODUxNDAifQ==",
            tags = nept_params['tags']
            )

        nept_run['input_parameters'] = input_parameters
    else:
        nept_run = FakeNeptune()

    if input_parameters['phase']=='train':

        training_parameters = parameters['train_parameters']
        # Path to folder containing ground truth images    
        if os.path.isdir(input_parameters['label_dir']):
            # label_paths = sorted(glob(input_parameters['label_dir']+'*'))
            label_paths = sorted([os.path.join(input_parameters['label_dir'], f) for f in os.listdir(input_parameters['label_dir'])])
            training_parameters['supervision'] = 'full'

        elif os.path.isfile(input_parameters['label_dir']):
            label_paths = sorted(pd.read_csv(input_parameters['label_dir'])['Paths'].tolist())
            training_parameters['supervision'] = 'full'
        else:
            print('Initial training of a model requires labeled targets')
        
        if mode == "ADDA":         
            if os.path.isdir(input_parameters['target_label_dir']):
                target_label_paths = sorted(glob(input_parameters['target_label_dir']+'*'))
                training_parameters['supervision'] = 'full'
            else:
                print('Initial training of a model requires labeled targets')

        input_image_type = list(input_parameters['image_dir'].keys())

        if not os.path.exists(input_parameters['output_dir']):
            os.makedirs(input_parameters['output_dir'])
        
        training_parameters['output_dir']=input_parameters['output_dir']

        if input_parameters['type']=='multi':
            
            # Getting multi-input image paths
            image_paths_base = []
            if mode == "ADDA":
                t_image_paths_base = []
                
            for inp_type in input_image_type:
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'*')))
                elif os.path.isfile(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(pd.read_csv(input_parameters['image_dir'][inp_type])['Paths'].tolist()))
                if mode == "ADDA":
                    if os.path.isdir(input_parameters['target_dir'][inp_type]):
                        t_image_paths_base.append(sorted(glob(input_parameters['target_dir'][inp_type]+'*')))
                    elif os.path.isfile(input_parameters['target_dir'][inp_type]):
                        image_paths_base.append(sorted(pd.read_csv(input_parameters['target_dir'][inp_type])['Paths'].tolist()))

            # Getting the image paths as lists of lists (each input type)
            image_paths = []
            for i in range(len(image_paths_base[0])):
                image_paths.append([image_paths_base[0][i], image_paths_base[1][i]])
            if mode == "ADDA":
                t_image_paths = []
                for i in range(len(t_image_paths_base[0])):
                    t_image_paths.append([t_image_paths_base[0][i], t_image_paths_base[1][i]])

        elif input_parameters['type']=='single':

            # Getting either DUET or Brightfield images
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                # image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'*'))
                image_paths = sorted([os.path.join(input_parameters['image_dir'][input_image_type[0]], f) for f in os.listdir(input_parameters['image_dir'][input_image_type[0]])])
            elif os.path.isfile(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist())
        
        # Now determining which images are used for training and which are used for testing
        if 'k_folds' in training_parameters['train_test_split']:

            kf = KFold(n_splits = int(training_parameters['train_test_split']['k_folds']),shuffle=True)
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
            train_exist = False
            valid_exist = False
            if 'train_test_source' in training_parameters['train_test_split']:
                train_exist = os.path.isfile(f"{training_parameters['train_test_split']['train_test_source']}/source_train_idx.npy")
                valid_exist = os.path.isfile(f"{training_parameters['train_test_split']['train_test_source']}/source_valid_idx.npy")
            if train_exist and valid_exist:
                train_idx_file =  f"{training_parameters['train_test_split']['train_test_source']}/source_train_idx.npy"                
                valid_idx_file =  f"{training_parameters['train_test_split']['train_test_source']}/source_valid_idx.npy"
                try:                        
                    train_idx = np.load(train_idx_file)
                    val_idx   = np.load(valid_idx_file)
                    print("Preveiuosly saved source indices loaded successfully!")
                except FileNotFoundError as e:
                    print(f"File not found: {e.filename}")
            else:
                # Random train/test split with 'split' proportion assigned to training data
                shuffle_idx = np.random.permutation(len(image_paths))

                split_pct = training_parameters['train_test_split']['split']
                train_idx = shuffle_idx[0:floor(split_pct*len(image_paths))]
                val_idx = shuffle_idx[floor(split_pct*len(image_paths)):len(image_paths)]
                
                # save train_val idx
                np.save(f"{training_parameters['train_test_split']['train_test_source']}/source_train_idx.npy", train_idx)
                np.save(f"{training_parameters['train_test_split']['train_test_source']}/source_valid_idx.npy", val_idx)
                                    
            train_images = [image_paths[i] for i in train_idx]
            test_images = [image_paths[i] for i in val_idx]
            train_labels = [label_paths[i] for i in train_idx]
            test_labels = [label_paths[i] for i in val_idx]
            
            # # Create dataframes only once
            # train_df = pd.DataFrame({
            #     'train_images': train_images,
            #     'train_labels': train_labels
            # })

            # test_df = pd.DataFrame({
            #     'test_images': test_images,
            #     'test_labels': test_labels
            # })
                        
            # # Create a Pandas Excel writer using openpyxl as the engine
            # with pd.ExcelWriter(f'{input_parameters["output_dir"]}/source_train_test.xlsx', engine='openpyxl') as writer:
            #     # Write each DataFrame to a different worksheet
            #     train_df.to_excel(writer, sheet_name='Train', index=False)
            #     test_df.to_excel(writer, sheet_name='Test', index=False)
                                    
            dataset_train, dataset_valid = make_training_set(
                'train',
                train_img_paths=train_images,
                train_tar = train_labels,
                valid_img_paths=test_images,
                valid_tar = test_labels,
                parameters=training_parameters
            )
            if mode == "None":
                model = Training_Loop(dataset_train, dataset_valid, training_parameters,nept_run)
                Test_Network(model, dataset_valid, nept_run, training_parameters)
            elif mode == "ADDA":
                if 'train_test_target' in training_parameters['train_test_split']:
                    train_idx_file =  f"{training_parameters['train_test_split']['train_test_target']}/target_train_idx{input_parameters['overlapRate']}.npy"
                    valid_idx_file =  f"{training_parameters['train_test_split']['train_test_target']}/target_valid_idx{input_parameters['overlapRate']}.npy"
                    try:
                        train_idx = np.load(train_idx_file)
                        val_idx   = np.load(valid_idx_file)
                        print("Preveiuosly saved target indices loaded successfully!")
                    except FileNotFoundError as e:
                        print(f"File not found: {e.filename}")
                    
                else:
                    # Random train/test split with 'split' proportion assigned to training data                
                    shuffle_idx_t = np.random.permutation(len(t_image_paths))

                    split_pct = training_parameters['train_test_split']['split']
                    train_idx = shuffle_idx_t[0:floor(split_pct*len(t_image_paths))]
                    val_idx = shuffle_idx_t[floor(split_pct*len(t_image_paths)):len(t_image_paths)]
                    
                    # save train_val idx 
                    np.save("/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH/NormalizedF/Results/"+f"target_train_idx{input_parameters['overlapRate']}.npy", train_idx)
                    np.save("/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH/NormalizedF/Results/"+f"target_valid_idx{input_parameters['overlapRate']}.npy", val_idx)
                                                
                train_images = [t_image_paths[i] for i in train_idx]
                test_images  = [t_image_paths[i] for i in val_idx]
                train_labels = [target_label_paths[i] for i in train_idx]
                test_labels  = [target_label_paths[i] for i in val_idx]
                
                # # # Create dataframes only once
                # # train_df = pd.DataFrame({
                # #     'train_images': train_images,
                # #     'train_labels': train_labels
                # # })

                # # test_df = pd.DataFrame({
                # #     'test_images': test_images,
                # #     'test_labels': test_labels
                # # })
                            
                # # Create a Pandas Excel writer using openpyxl as the engine
                # with pd.ExcelWriter('/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH/NormalizedF/Results/'+'target_train_test25.xlsx', engine='openpyxl') as writer:
                #     # Write each DataFrame to a different worksheet
                #     train_df.to_excel(writer, sheet_name='Train', index=False)
                #     test_df.to_excel(writer, sheet_name='Test', index=False)
                                                            
                dataset_train_t, dataset_valid_t = make_training_set(
                    'train',
                    train_img_paths=train_images,
                    train_tar = train_labels,
                    valid_img_paths=test_images,
                    valid_tar = test_labels,
                    parameters=training_parameters
                )
                
                modelADDA = TrainingADDA_Loop(input_parameters["model_file"], dataset_train, dataset_valid,
                                              dataset_train_t, dataset_valid_t, 
                                              training_parameters, nept_run)
                Test_Network(modelADDA, dataset_valid_t, nept_run, training_parameters)

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
            start_time = time.time()
            model = Training_Loop(dataset_train,dataset_valid,training_parameters,nept_run)
        
            end_time = time.time()
            spent_time = end_time - start_time
            print("Time spent to test is:", int(spent_time // 60), "minutes and", int(spent_time % 60), "seconds")
        
            Test_Network(model,dataset_valid,nept_run,training_parameters)

    elif input_parameters['phase']=='retrain':

        # Retraining existing model
        # Create new model version and download previous model 

        training_parameters = parameters['train_parameters']
        # Path to folder containing ground truth images
        if os.path.isdir(input_parameters['label_dir']):
            label_paths = sorted(glob(input_parameters['label_dir']+'*'))
            training_parameters['supervision'] = 'full'

        elif os.path.isfile(input_parameters['label_dir']):
            label_paths = sorted(pd.read_csv(input_parameters['label_dir'])['Paths'].tolist())
            training_parameters['supervision'] = 'full'

        else:
            label_paths = []
            training_parameters['supervision'] = 'semi'

        input_image_type = list(input_parameters['image_dir'].keys())

        if not os.path.exists(input_parameters['output_dir']):
            os.makedirs(input_parameters['output_dir'])
        training_parameters['output_dir'] = input_parameters['output_dir']

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
        if 'k_folds' in training_parameters['train_test_split']:

            kf = KFold(n_splits = int(training_parameters['train_test_split']['k_folds']),shuffle=True)
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
            # Test_Network(model, dataset_valid, nept_run, training_parameters)

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

            # Test_Network(model,dataset_valid,nept_run,training_parameters)

    elif input_parameters['phase']=='test':

        input_image_type = list(input_parameters['image_dir'].keys())

        if not os.path.exists(input_parameters['output_dir']):
            os.makedirs(input_parameters['output_dir'])

        # Whether to test with labels (calculate metrics) or not
        if 'label_dir' in input_parameters:
            if os.path.isdir(input_parameters['label_dir']):
                # label_paths = glob(input_parameters['label_dir']+'*')
                label_paths = [os.path.join(input_parameters['label_dir'], f)
                               for f in os.listdir(input_parameters['label_dir'])]
            elif os.path.isfile(input_parameters['label_dir']):
                label_paths = sorted(pd.read_csv(input_parameters['label_dir'])['Paths'].tolist())
        else:
            label_paths = []
        
        if input_parameters['type']=='multi':
            
            image_paths_base = []
            for inp_type in input_image_type:
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    # image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'*')))
                    image_paths_base.append([os.path.join(input_parameters['image_dir'][inp_type], f) 
                                             for f in os.listdir(input_parameters['image_dir'][inp_type])])
                elif os.path.isfile(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(pd.read_csv(input_parameters['image_dir'][inp_type])['Paths'].tolist()))

            # Hack for only predicting on images which contain minimal background
            # Defined threshold (40kb) for brightfield images
            passed_images = check_image_bytes(image_paths_base[1],lower_threshold = None, upper_threshold = None)
            print(f'{len(passed_images)} passed the bytes check')

            if 'skip_duplicates' in input_parameters:
                if input_parameters['skip_duplicates']:
                    passed_images = check_duplicate(passed_images,input_parameters['output_dir'])
                    print(f'{len(passed_images)} passed the duplicates check!')

            passed_image_idxes = [image_paths_base[1].index(i) for i in passed_images]
            image_paths_base = [
                [image_paths_base[0][i] for i in passed_image_idxes],
                passed_images
            ]

            # Getting the image paths as lists of lists (each input type)
            image_paths = []
            for i in range(len(image_paths_base[0])):
                image_paths.append([image_paths_base[0][i],image_paths_base[1][i]])

        elif input_parameters['type']=='single':
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                # image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'*'))
                image_paths = sorted([os.path.join(input_parameters['image_dir'][input_image_type[0]], f) 
                                  for f in os.listdir(input_parameters['image_dir'][input_image_type[0]])])
            else:
                image_paths = sorted(pd.read_excel(input_parameters['image_dir'][input_image_type[0]], sheet_name='Test')['test_images'])
        elif os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist())                

        # Loading model and data
        if 'neptune' in input_parameters:
            model_version = neptune.init_model(
                project = nept_params['project'],
                # with_id = input_parameters['model'],
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

            # Grabbing clustering scaler and training set coordinates
            try:
                model_version['UMAP_Scaler/scaler_means'].download(input_parameters['output_dir']+'/scaler_means.npy')
                model_version['UMAP_Scaler/scaler_var'].download(input_parameters['output_dir']+'/scaler_var.npy')
                model_version['UMAP_Scaler/umap_reducer'].download(input_parameters['output_dir']+'/umap_reducer.sav')

                model_details['scaler_means'] = np.load(input_parameters['output_dir']+'/scaler_means.npy')
                model_details['scaler_var'] = np.load(input_parameters['output_dir']+'/scaler_var.npy')
                model_details['umap_reducer'] = joblib.load(input_parameters['output_dir']+'/umap_reducer.sav')
            except:
                print('no umap info for this model')
        else:

            model_file = input_parameters['model_file']
            preprocessing = input_parameters['preprocessing']
            model_details = input_parameters['model_details']

            model_details['scaler_means'] = None
            model_details['scaler_var'] = None

        input_parameters['model_details'] = model_details
        input_parameters['preprocessing'] = preprocessing

        # This is a hack for running on large sets of large images
        if len(image_paths) >= 10:
            image_set_size = len(image_paths) // 1
        else:
            image_set_size = len(image_paths)
        run_throughs = ceil(len(image_paths)/image_set_size)
        start_time = time.time()
        for run in range(run_throughs):
            
            if not int((run+1)*image_set_size)>=len(image_paths):
                print(f'Running on images: {int(run*image_set_size)} to {int((run+1)*image_set_size)}')
                run_paths = image_paths[int(run*image_set_size):int((run+1)*image_set_size)]
            else:
                print(f'Running on images: {int(run*image_set_size)} to {len(image_paths)}')
                run_paths = image_paths[int(run*image_set_size):len(image_paths)]
            
            # print(run_paths)
                
            nothin, dataset_test = make_training_set(
                'test',
                None,
                None,
                run_paths,
                [],
                input_parameters
            )

            Test_Network(model_file, dataset_test, nept_run, input_parameters)
        end_time = time.time()
        spent_time = end_time - start_time
        print("Time spent to test is:", int(spent_time // 60), "minutes and", int(spent_time % 60), "seconds")

    elif input_parameters['phase']=='cluster':

        # Inputs are the same as testing but instead of predicting, just generating some clustering/relative clustering of latent features
        input_image_type = list(input_parameters['image_dir'].keys())
        label_paths = []

        if not os.path.exists(input_parameters['output_dir']):
            os.makedirs(input_parameters['output_dir'])

        
        if input_parameters['type']=='multi':

            image_paths_base = []
            for inp_type in input_image_type:
                if os.path.isdir(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(glob(input_parameters['image_dir'][inp_type]+'*')))
                elif os.path.isfile(input_parameters['image_dir'][inp_type]):
                    image_paths_base.append(sorted(pd.read_csv(input_parameters['image_dir'][inp_type])['Paths'].tolist()))
                else:
                    print(input_parameters['image_dir'][inp_type])
                    image_paths_base = []

            image_paths = [list(i) for i in zip(*image_paths_base)]

        elif input_parameters['type']=='single':
            if os.path.isdir(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(glob(input_parameters['image_dir'][input_image_type[0]]+'*'))
            elif os.path.isfile(input_parameters['image_dir'][input_image_type[0]]):
                image_paths = sorted(pd.read_csv(input_parameters['image_dir'][input_image_type[0]])['Paths'].tolist())
            else:
                print(input_parameters['image_dir'][input_image_type[0]])
                image_paths = []
        
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

        # Getting labels from model_version
        model_version['Merged_Results_Table'].download(input_parameters['output_dir']+'/Merged_Results_Table.csv')
        plot_labels = pd.read_csv(input_parameters['output_dir']+'Merged_Results_Table.csv',header = 0)

        cluster_object = Clusterer(input_parameters,plot_labels = plot_labels)
        cluster_object.run_clustering_iterator(model_file,dataset_test)

        # Uploading data to model_version metadata (if specified)
        upload_to_model_version = True
        if upload_to_model_version:
            model_version['UMAP_coordinates'].upload(input_parameters['output_dir']+'Extracted_UMAP_Coordinates.csv')
            
            umap_plots = os.listdir(input_parameters['output_dir']+'UMAP_Plots/')
            for u in umap_plots:
                model_version[f'UMAP_Plots/{u}'].upload(input_parameters['output_dir']+f'UMAP_Plots/{u}')

        model_version['UMAP_Scaler/scaler_means'].upload(input_parameters['output_dir']+f'scaler_means.npy')
        model_version['UMAP_Scaler/scaler_var'].upload(input_parameters['output_dir']+f'scaler_var.npy')
        model_version['UMAP_Scaler/umap_reducer'].upload(input_parameters['output_dir']+f'umap_reducer.sav')

if __name__=='__main__':
    main()
