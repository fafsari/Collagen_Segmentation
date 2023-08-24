"""


Finding the images with the greatest change in performance after including DUET data



"""

import os
import sys
import numpy as np

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image

base_dir = 'C:\\Users\\Sam\\Desktop\\Collagen_Segmentation\\'
output_dir = base_dir+'Results\\Metrics_Comparisons\\'

gt_dir = base_dir+'second_round\\C White Excluded\\'
f_dir = base_dir+'second_round\\NF 0\\'
b_dir = base_dir+'second_round\\B_2nd\\'

b_g_results_dir = base_dir+'Results\\Brightfield_Grayscale\\Testing_Output\\'
f_g_results_dir = base_dir+'Results\\Fluorescence_Grayscale\\Testing_Output\\'
mc_g_results_dir = base_dir+'Results\\MultiChannel_Grayscale\\Testing_Output\\'

b_rgb_results_dir = base_dir+'Results\\Brightfield_RGB\\Testing_Output\\'
f_rgb_results_dir = base_dir+'Results\\Fluorescence_RGB\\Testing_Output\\'
mc_rgb_results_dir = base_dir+'Results\\MultiChannel_RGB\\Testing_Output\\'

results_dict = {
    'Brightfield_Grayscale':{'dir':b_g_results_dir},
    'Brightfield_RGB':{'dir':b_rgb_results_dir},
    'Fluorescence_Grayscale':{'dir':f_g_results_dir},
    'Fluorescence_RGB':{'dir':f_rgb_results_dir},
    'MultiChannel_Grayscale':{'dir':mc_g_results_dir},
    'MultiChannel_RGB':{'dir':mc_rgb_results_dir}
}

for r in results_dict:
    results_dict[r]['results'] = pd.read_csv(results_dict[r]['dir']+'Merged_Results_Table.csv')


# Specifying metric used to determine improvement (list can contain multiple metrics)
sort_metric = ['AUC','Dice','Accuracy','Precision','Specificity','Recall']
top_n = 5

compare_results_dict = {}
for r in results_dict:
    for b in results_dict:
        #if not r==b and r.split('_')[-1]==b.split('_')[-1]:
        if not r==b:
            # Getting only the numeric columns
            column_dtypes = results_dict[r]['results'].dtypes
            num_dtypes = list((column_dtypes[column_dtypes==float]).index)

            # Making sure the two dataframes are in the same order
            if results_dict[r]['results']['Image_Names'].equals(results_dict[b]['results']['Image_Names']):

                num_columns_r = results_dict[r]['results'][results_dict[r]['results'].columns.intersection(num_dtypes)]
                num_columns_b = results_dict[b]['results'][results_dict[r]['results'].columns.intersection(num_dtypes)]
                compare_performance = num_columns_r-num_columns_b

                if f'{r}_vs_{b}' not in compare_results_dict and f'{b}_vs_{r}' not in compare_results_dict:
                    print(f'{r}_vs_{b}')
                    # Saving dataframe with metrics differences
                    non_numeric_columns = results_dict[r]['results'][results_dict[r]['results'].columns.symmetric_difference(num_dtypes)]
                    compare_performance = pd.concat([compare_performance,non_numeric_columns],axis=1,ignore_index=True)
                    compare_performance.columns = num_dtypes+list(results_dict[r]['results'].columns.symmetric_difference(num_dtypes))
                    compare_results_dict[r+'_vs_'+b] = compare_performance

                    base_save_path = output_dir+f'{r}_vs_{b}\\'
                    if not os.path.exists(base_save_path):
                        os.makedirs(base_save_path)

                    compare_performance.to_csv(base_save_path+f'{r}_vs_{b}.csv')

                    # Iterating through the metrics list
                    for s_m in sort_metric:
                        
                        # Sorting by sort columns (ascending = True shows improvement from r-->b)
                        sorted_comparison = compare_performance.sort_values(by=s_m,ascending=True)
                        top_n_improved = sorted_comparison.iloc[0:top_n,:]

                        # Finding the images with the worst performance for a certain metric and seeing how they changed in the next level up
                        

                        save_path = output_dir+f'{r}_vs_{b}\\Top_{top_n}_Improved\\{s_m}\\'

                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                            top_n_improved.to_csv(save_path+'Metrics_Comparison.csv')
                            
                            image_names = top_n_improved['Image_Names'].tolist()

                            for idx,i in enumerate(image_names):

                                gt_image = np.array(Image.open(gt_dir+i))
                                bf_image = np.array(Image.open(b_dir+i))
                                f_image = np.array(Image.open(f_dir+i))

                                # Results #1 for comparison
                                r_image = np.array(Image.open(results_dict[r]['dir']+f'Test_Example_{i.replace("jpg","tif")}'))[:,:,None]
                                r_image = np.repeat(r_image,3,axis=-1)
                                # Results #2 for comparison
                                b_image = np.array(Image.open(results_dict[b]['dir']+f'Test_Example_{i.replace("jpg","tif")}'))[:,:,None]
                                b_image = np.repeat(b_image,3,axis=-1)

                                separator = np.zeros((gt_image.shape[0],20,3))

                                combined_image_row = np.concatenate((bf_image,separator,f_image,separator,gt_image,separator,r_image,separator,b_image),axis=1)
                                img_fig = go.Figure(px.imshow(combined_image_row))
                                img_fig.update_xaxes(showticklabels=False)
                                img_fig.update_yaxes(showticklabels=False)
                                img_fig.update_layout(
                                    title=dict(text=i,font=dict(size=20),yref='paper'),
                                    margin=dict(b=0,l=10,r=10),
                                    height=600,
                                    width=3000
                                )

                                img_fig.write_image(save_path+f'Number_{idx+1}.png')



# Now checking specifically BF-->F-->MC improvement
types_list = ['Grayscale','RGB']
for t in types_list:

    b_to_f_improvement = compare_results_dict[f'Brightfield_{t}_vs_Fluorescence_{t}']
    f_to_mc_improvement = compare_results_dict[f'Fluorescence_{t}_vs_MultiChannel_{t}']

    # Now constraining to only test set
    b_to_f_improvement = b_to_f_improvement[b_to_f_improvement['Phase'].str.match('Test')]
    f_to_mc_improvement = f_to_mc_improvement[f_to_mc_improvement['Phase'].str.match('Test')]

    for s_m in sort_metric:
        
        # Getting only the samples that improved in each case
        b_to_f_metric = b_to_f_improvement[b_to_f_improvement[s_m]<0]
        f_to_mc_metric = f_to_mc_improvement[f_to_mc_improvement[s_m]<0]
        print(f'Number of b-f improved: {b_to_f_metric.shape[0]}, Number of f-mc improved: {f_to_mc_metric.shape[0]}')

        # Finding images which are in both dataframes
        b_to_f_names = b_to_f_metric['Image_Names'].tolist()
        f_to_mc_names = f_to_mc_metric['Image_Names'].tolist()
        print(f'Improvement found in: {len(b_to_f_names)} b-f and in: {len(f_to_mc_names)} f-mc')

        overlap_names = list(set(b_to_f_names) & set(f_to_mc_names))
        print(f'Found: {len(overlap_names)} overlaps')

        overlap_b_to_f = b_to_f_metric[b_to_f_metric['Image_Names'].isin(overlap_names)]
        overlap_f_to_mc = f_to_mc_metric[f_to_mc_metric['Image_Names'].isin(overlap_names)]

        #print(overlap_b_to_f['Image_Names'].equals(overlap_f_to_mc['Image_Names']))
        if overlap_b_to_f['Image_Names'].equals(overlap_f_to_mc['Image_Names']):

            overlap_b_to_f_num = overlap_b_to_f[overlap_b_to_f.columns.intersection(num_dtypes)]
            overlap_f_to_mc_num = overlap_f_to_mc[overlap_f_to_mc.columns.intersection(num_dtypes)]

            non_numeric_columns = overlap_b_to_f[overlap_b_to_f.columns.symmetric_difference(num_dtypes)]
            # Subtracting numeric columns of f-mc from b-f (More negative = more improvement from b-f-mc)
            diff_overlap = overlap_b_to_f_num-overlap_f_to_mc_num
            diff_overlap = pd.concat([diff_overlap,non_numeric_columns],axis=1,ignore_index=True)
            diff_overlap.columns = num_dtypes+non_numeric_columns.columns.tolist()

            sorted_diff_overlap = diff_overlap.sort_values(by=s_m,ascending=True)

            save_path = output_dir+f'Brightfield_Fluorescence_MultiChannel_{t}\\Top_{top_n}_Improved\\{s_m}\\'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

                top_n_improved = sorted_diff_overlap.iloc[0:top_n,:]
                top_n_improved.to_csv(save_path+'Metrics_Comparison.csv')

                image_names = top_n_improved['Image_Names'].tolist()

                for idx,i in enumerate(image_names):

                    gt_image = np.array(Image.open(gt_dir+i))
                    bf_image = np.array(Image.open(b_dir+i))
                    f_image = np.array(Image.open(f_dir+i))

                    b_pred_image = np.array(Image.open(results_dict[f'Brightfield_{t}']['dir']+f'Test_Example_{i.replace("jpg","tif")}'))[:,:,None]
                    b_pred_image = np.repeat(b_pred_image,3,axis=-1)
                    f_pred_image = np.array(Image.open(results_dict[f'Fluorescence_{t}']['dir']+f'Test_Example_{i.replace("jpg","tif")}'))[:,:,None]
                    f_pred_image = np.repeat(f_pred_image,3,axis=-1)
                    mc_pred_image = np.array(Image.open(results_dict[f'MultiChannel_{t}']['dir']+f'Test_Example_{i.replace("jpg","tif")}'))[:,:,None]
                    mc_pred_image = np.repeat(mc_pred_image,3,axis=-1)

                    separator = np.zeros((gt_image.shape[0],20,3))

                    combined_image_row = np.concatenate((bf_image,separator,f_image,separator,gt_image,separator,b_pred_image,separator,f_pred_image,separator,mc_pred_image),axis=1)

                    img_fig = go.Figure(px.imshow(combined_image_row))
                    img_fig.update_xaxes(showticklabels=False)
                    img_fig.update_yaxes(showticklabels=False)
                    img_fig.update_layout(
                        title=dict(text=i,font=dict(size=20),yref='paper'),
                        margin=dict(b=0,l=10,r=10),
                        height=600,
                        width=3500
                    )

                    img_fig.write_image(save_path+f'Number_{idx+1}.png')







