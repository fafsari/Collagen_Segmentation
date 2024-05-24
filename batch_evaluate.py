"""

Evaluate a bunch of segmentation models one after another

"""

import os
from time import sleep
import subprocess

models_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/Results/'
models_list = os.listdir(models_dir)
print(f'---------Found {len(models_list)} models! Wow!----------------------')

output_dir = 'Evaluation_Metrics'

for model_name in models_list:

    # If only running for the ones that haven't been evaluated yet
    #if not os.path.exists(f'{models_dir}{model_name}/{output_dir}'):

    # Checking if this model contains all predictions or just validation predictions.
    if len(os.listdir(f'{models_dir}{model_name}/Testing_Output/'))<=175:
        process = subprocess.Popen(["python3", "Collagen_Segmentation/CollagenEvaluate.py", "--test_model_path", f'{models_dir}{model_name}', "--label_path", "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/C"])
        process.wait()
    else:
        # These ones need extra specification for images which were originally in the training set and which were in the validation set
        # We only want to compare performance for images in the validation set.
        process = subprocess.Popen(["python3", "Collagen_Segmentation/CollagenEvaluate.py", "--train_test_names", f'{models_dir}{model_name}/Merged_Results_Table.csv',"--test_model_path", f'{models_dir}{model_name}', "--label_path", "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/C"])
        process.wait()
