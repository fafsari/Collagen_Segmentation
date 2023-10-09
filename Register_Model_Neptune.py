"""

Registering models to Neptune

This includes uploading the actual model file (.pth) as well as associated metadata


"""

import os
import neptune

from glob import glob
import pandas as pd

#neptune_api_token = os.environ.get('NEPTUNE_API_TOKEN')
"""
model_version = neptune.init_model_version(
    model = 'DEDU-BFG',
    project = 'samborder/Deep-DUET',
    mode = 'async',
    api_token=neptune_api_token
)
"""

models_list = ['MCRGB','MCG','FRGB','FG','BFRGB','BFG']
model_names = ['MultiChannel_RGB','MultiChannel_Grayscale','Fluorescence_RGB','Fluorescence_Grayscale','Brightfield_RGB','Brightfield_Grayscale']

for m_key,m_name in zip(models_list,model_names):

    model_version = neptune.init_model(
        project = 'samborder/Deep-DUET',
        with_id = 'DEDU-'+m_key,
        mode='async'
    )

    model_name = m_name
    path_to_model = f'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\Results\\{model_name}\\models\\Collagen_Seg_Model_Latest.pth'
    """
    #model_version['model_file'].upload(path_to_model)

    #model_color_transform = 'green'
    #model_image_means = '0.65390968'
    #model_image_stds = '0.1818624'
    if m_name.split('_')[-1]=='RGB':
        if m_name.split('_')[0]=='MultiChannel':
            image_size = '512,512,6'
        else:
            image_size = '512,512,3'
    else:
        if m_name.split('_')[0]=='MultiChannel':
            image_size = '512,512,2'
        else:
            image_size = '512,512,1'
    mask_size = '512,512,1'

    #model_version['preprocessing/color_transform'] = model_color_transform
    #model_version['preprocessing/image_means'] = model_image_means
    #model_version['preprocessing/image_stds'] = model_image_stds
    model_version['preprocessing/image_size'] = image_size
    model_version['preprocessing/mask_size'] = mask_size

    model_dict = {
        'encoder':'resnet34',
        'encoder_weights':'imagenet',
        'active':'sigmoid',
        'architecture':'Unet++',
        'loss':'L1',
        'lr':0.00005,
        'batch_size':2,
        'target_type':'nonbinary',
        'ann_classes':'background,collagen'
    }

    for m_p in model_dict:
        model_version[f'model_details/{m_p}'] = model_dict[m_p]
    """
    
    path_to_csvs = f'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\Results\\{model_name}\\Testing_Output\\'
    model_csv_files = glob(path_to_csvs+'*.csv')

    for m_csv in model_csv_files:
        file_name = m_csv.split('\\')[-1].split('.')[0]
        #data = pd.read_csv(m_csv,header=0)
        model_version[file_name].upload(m_csv)
    









