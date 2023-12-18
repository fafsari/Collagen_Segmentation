"""

Generate segmentation metrics for selected model.

See: https://plotly.com/python/roc-and-pr-curves/ for more examples and plotting reference

Evaluation model folder layout:
- {Model_Name}/
    - model/Collagen_Seg_Model_Latest.pth
    - Testing_Output/*Testing_Example_{image_name}_prediction.tif

Generates new directory:
- Evaluation_Metrics/
    - {Model_Name}_ROC_Plot.png
    - {Model_Name}_Segmentation_Metrics.csv
"""

import os
import sys
import numpy as np
import pandas as pd

from PIL import Image

import argparse

import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, f1_score

def precision_score(y_true,y_pred):

    # Calculate precision given two flattened vectors of labels
    # precision = (tp)/(tp+fp)
    tp = np.sum(y_pred(np.argwhere(y_pred==y_true)))
    fp = np.sum(1-y_pred(np.argwhere(y_pred!=y_true)))

    return tp/(tp+fp)

def main(args):

    print('----------------------------')
    print('---Generating Metrics-------')
    print('----------------------------')
    print(f'test_model_path: {args.test_model_path}')
    print(f'label_path: {args.label_path}')
    print(f'output_dir: {args.output_dir}')
    print(f'train_test_names: {args.train_test_names}')
          
    
    model_name = args.test_model_path.split(os.sep)[-1]

    # Checking whether train_test_names is provided
    if args.train_test_names is None:

        # Checking contents of Testing_Output/ directory
        test_names = os.listdir(f'{args.test_model_path}/Testing_Output/')

    else:

        # Reading the file specified for train_test_names
        train_test_df = pd.read_csv(args.train_test_names)

        # Getting the "Test" image names
        test_names = train_test_df[train_test_df['Phase'].str.match('Test')]['Image_Names'].tolist()

    print(f'Found: {len(test_names)} images for metrics calculation')
    
    with tqdm(test_names) as pbar:
        pbar.set_description(f'Computing metrics: 0/{len(test_names)}')

        metrics_dict = {
            'Image_Name':[],
            'Dice':[],
            'Accuracy':[],
            'Recall':[],
            'Precision':[],
            'Specificity':[],
            'AUC':[],
            'MSE':[]
        }

        roc_curve_aggregate = []

        gt_names = os.listdir(args.label_path+'/')

        for t_idx,t in test_names:
            # Adjusting the test name to match the original image format (adjust as needed)
            adjusted_name = t.replace('Test_Example_','').replace('_prediction','').replace('.tif','.jpg')

            if adjusted_name in gt_names:

                test_image = np.array(Image.open(f'{args.test_model_path}/Testing_Output/{t}'))
                binary_test_image = np.uint8(np.where(test_image>0))

                gt_image = np.array(Image.open(f'{args.label_path}/{adjusted_name}'))
                binary_gt_image = np.uint8(np.where(gt_image>0))

                # Accuracy
                accuracy = accuracy_score(binary_gt_image.flatten(),binary_test_image.flatten())

                # Dice (F1)
                dice = f1_score(binary_gt_image.flatten(),binary_test_image.flatten())

                # Recall (Sensitivity/True Positive Rate)
                recall = recall_score(binary_gt_image.flatten(),binary_test_image.flatten())

                # Precision
                precision = precison_score(binary_gt_image.flatten(),binary_test_image.flatten())

                # Area Under the Curve (AUC)
                auc = roc_auc_score(binary_gt_image.flatten(),binary_test_image.flatten())

                # ROC and Specificity
                false_pos_rate, true_pos_rate, _ = roc_curve(binary_gt_image.flatten(),binary_test_image.flatten())
                specificity = np.nanmean(1-false_pos_rate)
                
                roc_curve_aggregate.append([false_pos_rate,true_pos_rate])

                # MSE
                mse = np.nanmean((test_image-gt_image)**(0.5))
                
                # Adding metrics to metrics_dict
                metrics_dict['Image_Name'].append(t)
                metrics_dict['Dice'].append(dice)
                metrics_dict['Accuracy'].append(accuracy)
                metrics_dict['Recall'].append(recall)
                metrics_dict['Precision'].append(precision)
                metrics_dict['Specificity'].append(specificity)
                metrics_dict['AUC'].append(auc)
                metrics_dict['MSE'].append(mse)

            pbar.update(t_idx)
            pbar.set_description(f'Computing metrics: {t_idx}/{len(test_names)}')

        pbar.close()

        print('-----------Done! Generating combined ROC plot----------')
        # Creating metrics dataframe
        metrics_df = pd.DataFrame(metrics_dict)

        # Finding min and max auc
        min_auc = np.nanmin(metrics_dict['AUC'])
        max_auc = np.nanmax(metrics_dict['AUC'])
        mean_auc = np.nanmean(metrics_dict['AUC'])
        median_auc = np.nanmedian(metrics_dict['AUC'])

        # Finding curves associated with min and max
        min_idx = metrics_dict['AUC'].index(min_auc)
        max_idx = metrics_dict['AUC'].index(max_auc)

        # Creating figure. Combined minimum and maximum ROC with fill between them. 
        fig = go.Figure()
        fig.add_shape(
            type = 'line',
            line = dict(dash='dash'),
            x0 = 0, x1 = 1, y0 = 0, y1=1,
            name = 'Random Chance'
        )
        fig.add_trace(
            go.Scatter(
                x = roc_curve_aggregate[min_idx][0],
                y = roc_curve_aggregate[min_idx][1],
                fill = None,
                mode = 'lines',
                line_color = 'red',
                name = f'Minimum AUC: {min_auc}'
            )
        )
        fig.add_trace(
            go.Scatter(
                x = roc_curve_aggregate[max_idx][0],
                y = roc_curve_aggregate[max_idx][1],
                fill = 'tonexty',
                mode = 'lines',
                line_color = 'blue',
                name = f'Maximum AUC: {max_auc}'
            )
        )

        fig.update_layout(
            xaxis_title = 'False Positive Rate (FPR)',
            yaxis_title = 'True Positive Rate',
            yaxis = dict(scaleanchor='x',scaleratio=1),
            xaxis = dict(constrain='domain'),
            width=700,
            height = 500,
            title = f'ROC for {model_name}, Mean AUC: {mean_auc}, Median AUC: {median_auc}'
        )

        # Saving outputs to output directory
        if args.output_dir=='Evaluation_Metrics':
            if not os.path.exists(f'{args.test_model_path}/Evaluation_Metrics/'):
                os.makedirs(f'{args.test_model_path}/Evaluation_Metrics/')

            metrics_df.to_csv(f'{args.test_model_path}/Evaluation_Metrics/{model_name}_Segmentation_Metrics.csv')
            fig.write_image(f'{args.test_model_path}/Evaluation_Metrics/{model_name}_ROC_Plot.png')

        else:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            metrics_df.to_csv(args.output_dir+f'/{model_name}_Segmentation_Metrics.csv')
            fig.write_image(args.output_dir+f'/{model_name}_ROC_Plot.png')



if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description = 'Collagen segmentation evaluation argument parser'
    )

    parser.add_argument('test_model_path',type=str,help='Parent directory for model to evaluate, see comment in CollagenEvaluate.py for folder structure.')
    parser.add_argument('label_path',type=str,help='Path to label masks (not one-hot)')
    parser.add_argument('output_dir',type=str,default='Evaluation_Metrics',help='If you want to save the output to another path, specify here. (no / needed at the end).')
    parser.add_argument('train_test_names',type=str,default=None,help='If you have predictions for both the training and testing/holdout set images in the same directory, you can specify the path to a csv file where one column has image name (Image_Names) and another column has whether it is "Train" or "Test" (Phase).')

    main(parser.parse_args())



