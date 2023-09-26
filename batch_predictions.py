"""

Batch prediction with all datasets and all models

"""

import json
import os
from time import sleep

base_model_dir = '/blue/pinaki.sarder/samuelborder/Same_Training_set/'
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
        'model_file': f'{base_model_dir}MultiChannel_Grayscale/models/Collagen_Seg_Model_Latest.pth'
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
        'model_file':f'{base_model_dir}Fluorescence_Grayscale/models/Collagen_Seg_Model_Latest.pth'
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

dataset_list = ['Vetpath','CGPL','JHU']

print(f'Iterating through {len(model_dict_list)} models on {len(dataset_list)} datasets')
base_data_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/10 SET/'

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
            "source_files":["*.py","**/.py"],
            "tags":[]
        }
    }
}

inputs_file_path = './test_inputs.json'

dataset_list = [dataset_list[0]]
model_dict_list = [model_dict_list[0]]

count = 0

for dataset in dataset_list:
    for model in model_dict_list:

        # Generating new test_inputs
        test_inputs['input_parameters']['type'] = model['type']
        if model['type']=='multi':
            test_inputs['input_parameters']['image_dir'] = {
                "DUET":f'{base_data_dir}{dataset}/Fluorescence/',
                'Brightfield':f'{base_data_dir}{dataset}/Brightfield/'
            }
        elif model['type']=='single':
            if 'B' in model['model']:
                test_inputs['input_parameters']['image_dir'] = {
                    'Brightfield':f'{base_data_dir}{dataset}/Brightfield/'
                }
            else:
                test_inputs['input_parameters']['image_dir'] = {
                    'DUET':f'{base_data_dir}{dataset}/Fluorescence/'
                }
        
        test_inputs['output_dir'] = f'{base_data_dir}{dataset}/Results/{model["tags"][0].split(" ")[0]}/'
        test_inputs['model'] = model['model']
        test_inputs['model_file'] = model['model_file']
        test_inputs['input_parameters']['neptune']['tags'] = model['tags'][0].replace(' ',f' {dataset} ')

        print(test_inputs)
        with open(inputs_file_path.replace('.json',f'{count}.json'),'w') as f:
            json.dump(test_inputs,f,ensure_ascii=False)
            f.close()
        
        # The only time this usage is acceptable
        os.system(f'python3 Collagen_Segmentation/CollagenSegMain.py test_inputs{count}.json')
        sleep(50)
        count+=1


