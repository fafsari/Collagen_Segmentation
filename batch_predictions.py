"""

Batch prediction with all datasets and all models

"""

import json
import os
from time import sleep
import subprocess

base_model_dir = '/blue/pinaki.sarder/samuelborder/Same_Training_Set/'
model_dict_list = [
    {
        'model': 'DEDU-MCRGB',
        'type': 'multi',
        'tags': ['MultiChannel_RGB Predictions'],
        'model_file': f'{base_model_dir}MultiChannel_RGB/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-MCG',
        'type':'multi',
        'tags':['MultiChannel_G Predictions'],
        'model_file': f'{base_model_dir}MultiChannel_Green/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-FRGB',
        'type':'single',
        'tags':['Fluorescence_RGB Predictions'],
        'model_file':f'{base_model_dir}Fluorescence_RGB/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-FG',
        'type':'single',
        'tags':['Fluorescence_G Predictions'],
        'model_file':f'{base_model_dir}Fluorescence_G/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-BFRGB',
        'type':'single',
        'tags':['Brightfield_RGB Predictions'],
        'model_file':f'{base_model_dir}Brightfield_RGB/models/Collagen_Seg_Model_Latest.pth'
    },
    {
        'model':'DEDU-BFG',
        'type':'single',
        'tags':['Brightfield_G Predictions'],
        'model_file':f'{base_model_dir}Brightfield_G/models/Collagen_Seg_Model_Latest.pth'
    }
]

model_dict_list = [model_dict_list[2]]
base_data_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/DUET UCD PATH vs CGPL/'
dataset_list = os.listdir(base_data_dir)
#dataset_list = ['Same_Training_Set_Data']

print(f'Iterating through {len(model_dict_list)} models on {len(dataset_list)} datasets')

test_inputs = {
    "input_parameters":{
        "phase":"test",
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

inputs_file_path = './batch_inputs/test_inputs.json'

if not os.path.exists('./batch_inputs/'):
    os.makedirs('./batch_inputs/')

count = 0

for dataset in dataset_list:
    for model in model_dict_list:

        # Generating new test_inputs
        test_inputs['input_parameters']['type'] = model['type']
        if model['type']=='multi':
            test_inputs['input_parameters']['image_dir'] = {
                "DUET":f'{base_data_dir}{dataset}/F/',
                'Brightfield':f'{base_data_dir}{dataset}/B/'
            }
        elif model['type']=='single':
            if 'B' in model['model']:
                test_inputs['input_parameters']['image_dir'] = {
                    'Brightfield':f'{base_data_dir}{dataset}/B/'
                }
            else:
                test_inputs['input_parameters']['image_dir'] = {
                    'DUET':f'{base_data_dir}{dataset}/F/'
                }
        
        output_dir = f'{base_data_dir}{dataset}/Results/{model["tags"][0].split(" ")[0]}/'
        test_inputs['input_parameters']['output_dir'] = output_dir
        test_inputs['input_parameters']['model'] = model['model']
        test_inputs['input_parameters']['model_file'] = model['model_file']
        test_inputs['input_parameters']['neptune']['tags'] = model['tags'][0].replace(' ',f' {dataset} ')

        print(test_inputs)
        with open(inputs_file_path.replace('.json',f'{count}.json'),'w') as f:
            json.dump(test_inputs,f,ensure_ascii=False)
            f.close()
        
        #if not os.path.exists(output_dir):
        process = subprocess.Popen(['python3', 'Collagen_Segmentation/CollagenSegMain.py', f'./batch_inputs/test_inputs{count}.json'])
        process.wait()

        exit_code = process.returncode
        print(f'Return code of process was: {exit_code}')
        #else:
        #    print('Already run, skipping')

        count+=1


