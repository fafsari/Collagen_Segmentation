"""

WSI prediction main file, organize inputs into input pipeline and prediction script

"""

import os
import sys
import pandas as pd

import json

from WSI_Input_Pipeline import WSISegmentationDataSet
from WSI_Predict import Test_Network


# Changing up from sys.argv to reading a specific set of input parameters
parameters_file = sys.argv[1]

parameters = json.load(open(parameters_file))
input_parameters = parameters['input_parameters']
test_parameters = parameters['testing_parameters']

testing_dataset = WSISegmentationDataSet(test_parameters['target_type'],
                                        test_parameters['batch_size'],
                                        input_parameters['image_dir'])

model_path = input_parameters['model_file']

Test_Network(model_path,testing_dataset,test_parameters)
