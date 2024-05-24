"""

Summarize segmentation metrics from each folder

"""

import os
import sys
import pandas as pd

import numpy as np

from glob import glob

import plotly.express as px
import plotly.graph_objects as go


base_dir = 'D:\\Collagen_Segmentation\\Same_Training_Set_Data\\Results\\All_Results_and_Models\\'
save_dir = 'D:\\Collagen_Segmentation\\Same_Training_Set_Data\\Results\\All_Results_and_Models\\'

folders = os.listdir(base_dir)
folders = [i for i in folders if 'csv' not in i and 'png' not in i]
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

    model_names.append(f)
    model_means.append(seg_metrics_mean)
    model_stds.append(seg_metrics_std)

agg_mean_df = pd.DataFrame(data = model_means,index=model_names)
agg_std_df = pd.DataFrame(data = model_stds, index = model_names)

agg_mean_df.to_csv(save_dir+'Means.csv')
agg_std_df.to_csv(save_dir+'Stds.csv')



# Combining ROC Data into one ROC Plot
combine_models_list = [
    "Brightfield_RGB",
    "Fluorescence_RGB",
    "Ensemble_RGB"
]

combined_roc_fig = go.Figure()
# Adding random chance
combined_roc_fig.add_shape(
    type='line',
    line=dict(dash='dash'),
    x0=0,x1=1,y0=0,y1=1,
    name='Random Chance'
)

for c in combine_models_list:

    # Reading ROC data file
    roc_data = pd.read_csv(base_dir+f'{c}\\Evaluation_Metrics\\{c}_ROC_Data.csv')

    combined_roc_fig.add_trace(
        go.Scatter(
            x = roc_data['Mean FPR'].values,
            y = roc_data['Mean TPR'].values,
            mode = 'lines',
            name = c
        )
    )


combined_roc_fig.update_layout(
    xaxis = {
        'title':{
            'text': '<b>False Positive Rate (FPR)</b>',
        },
        'constrain':'domain'
    },
    yaxis = {
        'title':{
            'text':'<b>True Positive Rate (TPR)</b>'
        },
        'scaleanchor':'x',
        'scaleratio':1
    },
    title = {
        'text':'<b>Combined ROC Plot</b>'
    },
    legend = {
        'title':{
            'text':'<b>Model Type</b>'
        }
    },
    width=700,
    height=500
)

combined_roc_fig.write_image(save_dir+'Combined_ROC_Plot.png')

