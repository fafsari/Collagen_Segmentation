"""

Batch quantification of collagen from multiple datasets

    - 

"""

import os
from time import sleep
import subprocess

datasets_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/020524_DUET_Patches/'
datasets_list = os.listdir(datasets_dir)
print(datasets_list)
print(f'---------Found {len(datasets_list)} models! Wow!----------------------')

output_dir = '/Collagen_Quantification/'
results_dir = '/Results/Ensemble_RGB/Testing_Output/'
b_dir = '/B/'
f_dir = '/F/'
threshold = 0.1

use_stitched = True


for dataset_name in datasets_list:

    # If only running for the ones that haven't been quantified yet
    #if not os.path.exists(f'{datasets_dir}{dataset_name}/{output_dir}'):

    process = subprocess.Popen(["python3", "Collagen_Segmentation/CollagenQuantify.py", "--test_image_path", f'{datasets_dir}{dataset_name}{results_dir}','--bf_image_dir',f'{datasets_dir}{dataset_name}{b_dir}','--f_image_dir',f'{datasets_dir}{dataset_name}{f_dir}','--output_dir',f'{datasets_dir}{dataset_name}{output_dir}','--threshold',f'{threshold}','--use_stitched',f'{use_stitched}'])
    process.wait()
