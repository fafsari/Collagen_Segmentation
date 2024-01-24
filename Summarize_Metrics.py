"""

Summarize segmentation metrics from each folder

"""

import os
import sys
import pandas as pd

import numpy as np

from glob import glob

base_dir = 'D:\\Collagen_Segmentation\\Same_Training_Set_Data\\Results\\All_Results_and_Models\\'
save_dir = 'D:\\Collagen_Segmentation\\Same_Training_Set_Data\\Results\\All_Results_and_Models\\'

folders = os.listdir(base_dir)
print(f'Found: {len(folders)} folders')

metrics_list = [
    'Dice',
    'Accuracy',
    'Recall',
    'Precision',
    'Specificity',
    'AUC',
    'MSE'
]

model_names = []
model_means = []
model_stds = []
for f in folders:

    # Finding the metrics table
    path_to_table = f'{base_dir}\\{f}\\Evaluation_Metrics\\{f}_Segmentation_Metrics.csv'

    seg_metrics_df = pd.read_csv(path_to_table,index_col=0)

    seg_metrics_mean = seg_metrics_df.mean(axis=0,numeric_only=True).to_dict()
    seg_metrics_std = seg_metrics_df.std(axis=0,numeric_only=True).to_dict()

    print(seg_metrics_mean)
    model_names.append(f)
    model_means.append(seg_metrics_mean)
    model_stds.append(seg_metrics_std)

agg_mean_df = pd.DataFrame(data = model_means,index=model_names)
agg_std_df = pd.DataFrame(data = model_stds, index = model_names)

agg_mean_df.to_csv(save_dir+'Means.csv')
agg_std_df.to_csv(save_dir+'Stds.csv')



