"""

Batch extraction of clustering data for latent features from each training set


"""

import json
import os
import sys
from time import sleep
import subprocess


base_model_dir = '/blue/pinaki.sarder/samuelborder/Same_Training_Set/'
bf_train_data_df = 'Brightfield_Training.csv'
bf_test_data_df = 'Brightfield_Testing.csv'
du_train_data_df = 'DUET_Training.csv'
du_test_data_df = 'DUET_Testing.csv'

model_dict_list = [
    {
        'model': 'DEDU-MCRGB',
        'type': 'multi',
        'tags': ['MultiChannel_RGB Clustering'],
        'model_file': f'{base_model_dir}MultiChannel_RGB/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-MCG',
        'type':'multi',
        'tags':['MultiChannel_G Clustering'],
        'model_file': f'{base_model_dir}MultiChannel_Green/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-FRGB',
        'type':'single',
        'tags':['Fluorescence_RGB Clustering'],
        'model_file':f'{base_model_dir}Fluorescence_RGB/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-FG',
        'type':'single',
        'tags':['Fluorescence_G Clustering'],
        'model_file':f'{base_model_dir}Fluorescence_G/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-BFRGB',
        'type':'single',
        'tags':['Brightfield_RGB Clustering'],
        'model_file':f'{base_model_dir}Brightfield_RGB/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-BFG',
        'type':'single',
        'tags':['Brightfield_G Clustering'],
        'model_file':f'{base_model_dir}Brightfield_G/models/Collagen_Seg_Model_Latest.pth'
    }
]

cluster_inputs = {
    "input_parameters":{
        "phase":"cluster",
        "type":"",
        "image_dir":{},
        "output_dir":"",
        "model":"",
        "model_file":"",
        "neptune":{
            "project":"samborder/Deep-DUET",
            "source_files":["*.py","**/*.py"],
            "tags":[]
        }
    }
}

inputs_file_path = './batch_inputs/cluster_inputs.json'

if not os.path.exists('./batch_inputs/'):
    os.makedirs('./batch_inputs/')

count = 0

for model in model_dict_list:

    # Generating new cluster_inputs
    cluster_inputs['input_parameters']['type'] = model['type']
    if model['type']=='multi':
        cluster_inputs['input_parameters']['image_dir'] = {
            "DUET":du_train_data_df,
            'Brightfield':bf_train_data_df
        }
    elif model['type']=='single':
        if 'B' in model['model']:
            cluster_inputs['input_parameters']['image_dir'] = {
                'Brightfield':bf_train_data_df
            }
        else:
            cluster_inputs['input_parameters']['image_dir'] = {
                'DUET':du_train_data_df
            }
    
    output_dir = model['model_file'].replace('/model/Collagen_Seg_Model_Latest.pth','')
    cluster_inputs['input_parameters']['output_dir'] = output_dir
    cluster_inputs['input_parameters']['model'] = model['model']
    cluster_inputs['input_parameters']['model_file'] = model['model_file']

    with open(inputs_file_path.replace('.json',f'{count}.json'),'w') as f:
        json.dump(cluster_inputs,f,ensure_ascii=False)
        f.close()
    
    #if not os.path.exists(output_dir):
    process = subprocess.Popen(['python3', 'Collagen_Segmentation/CollagenSegMain.py', f'./batch_inputs/cluster_inputs{count}.json'])
    process.wait()

    exit_code = process.returncode
    print(f'Return code of process was: {exit_code}')
    #else:
    #    print('Already run, skipping')

    count+=1



























