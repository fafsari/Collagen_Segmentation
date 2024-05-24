"""


Script to generate some figures comparing features with fibrosis scores

Aggregated by case and non-aggregated
    - Aggregations:
        - Min, Max, Med, Mean
    - 
Dimensionally reduced and non

"""

import os
import sys
import numpy as np

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from umap import UMAP


def main():

    base_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/020524_DUET_Patches/'
    slide_names = [i for i in os.listdir(base_dir) if os.path.isdir(base_dir+i)]
    print(f'Slide Names: {slide_names}')

    quant_path = '/Collagen_Quantification/Patch_Collagen_Features.csv'
    output_path = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Fibrosis_Scores_Output/'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    fibrosis_scores = base_dir+'Fibrosis Scores_FixedCores.xlsx'
    fib_df = pd.read_excel(fibrosis_scores)

    combined_features = pd.DataFrame()
    for slide in slide_names:

        # Reading in the features for this slide:
        slide_features = pd.read_csv(base_dir+slide+quant_path,index_col=0)
        slide_features['SlideName'] = [slide]*slide_features.shape[0]

        # Adding fibrosis score
        if slide in fib_df['Slide'].tolist():
            slide_features['Fibrosis Score'] = [fib_df[fib_df['Slide'].str.match(slide)]['Pathologist Fibrosis Score'].tolist()[0]]*slide_features.shape[0]
        else:
            print(f'Could not find {slide} in list')
            slide_features['Fibrosis Score'] = [np.nan]*slide_features.shape[0]
        if combined_features.empty:
            combined_features = slide_features
        else:
            combined_features = pd.concat([combined_features,slide_features],axis = 0, ignore_index = True)

    # Saving combined_features
    combined_features = combined_features.dropna(how='any')
    combined_features.to_csv(output_path+'Combined_Features.csv')
    print(f'Saved combined features to output file: {output_path+"Combined_Features.csv"}')

    # Aggregating for each slide:
    aggregated_features = combined_features.groupby(by=['SlideName']).mean()
    aggregated_features.reset_index(inplace=True)
    aggregated_features.to_csv(output_path+'Aggregated_Features.csv')
    print(f'Aggregated features has shape: {aggregated_features.shape}')


    # Generating correlation matrix
    numeric_combined_features = combined_features.select_dtypes(include = 'number')
    print(f'Numeric column names: {numeric_combined_features.columns.tolist()}')

    # Diagonal will be 1.0 as each feature will be highly correlated with itself
    # Other features will be off-diagonal and be -1-->1
    corr_matrix = numeric_combined_features.corr()
    corr_matrix.to_csv(output_path+'Feature_Correlation_NonAgg.csv')

    # Repeating for aggregated features
    numeric_aggregated_features = aggregated_features.select_dtypes(include = 'number')
    corr_agg_matrix = numeric_aggregated_features.corr()
    corr_agg_matrix.to_csv(output_path+'Feature_Correlation_Agg.csv')

    numeric_combined_features.drop(columns = ['Fibrosis Score'],inplace=True)
    numeric_aggregated_features.drop(columns = ['Fibrosis Score'],inplace=True)
    
    # Recursive feature elimination
    # Non-Agg
    scaled_nonagg_features = StandardScaler().fit_transform(numeric_combined_features.values)
    """
    linear_svc_nonagg = LinearSVC(C=0.1, penalty = 'l2',dual = False,max_iter=10000).fit(scaled_nonagg_features, combined_features['Fibrosis Score'].values)
    nonagg_model = SelectFromModel(linear_svc_nonagg,prefit=True)
    #new_features = nonagg_model.transform(numeric_combined_features.values)

    non_agg_selected_features = nonagg_model.get_feature_names_out(input_features = numeric_combined_features.columns.tolist()).tolist()
    print(f'Non-Agg selected features: {non_agg_selected_features}')
    """
    # Agg
    scaled_agg_features = StandardScaler().fit_transform(numeric_aggregated_features.values)
    """
    linear_svc_agg = LinearSVC(C=0.1, penalty = 'l2',dual = False,max_iter=10000).fit(scaled_agg_features, aggregated_features['Fibrosis Score'].values)
    agg_model = SelectFromModel(linear_svc_agg,prefit=True)
    #new_features = nonagg_model.transform(numeric_aggregated_features.values)

    agg_selected_features = agg_model.get_feature_names_out(input_features = numeric_aggregated_features.columns.tolist()).tolist()
    print(f'Agg selected features: {agg_selected_features}')
    """

    agg_fig_output_path = output_path+'Agg/'
    nonagg_fig_output_path = output_path+'NonAgg/'
    if not os.path.exists(agg_fig_output_path):
        os.makedirs(agg_fig_output_path)
    if not os.path.exists(nonagg_fig_output_path):
        os.makedirs(nonagg_fig_output_path)

    # Making ridge plots of selected features
    
    # Non-Agg:
    unique_fib_scores = sorted(np.unique(combined_features['Fibrosis Score'].tolist()).tolist())
    colors = n_colors('rgb(0,0,255)','rgb(255,0,0)',len(np.unique(combined_features['Fibrosis Score'].tolist()).tolist()),colortype='rgb')
        
    # Using scaled data instead:
    sorted_non_agg = pd.DataFrame(data= scaled_nonagg_features,columns = numeric_combined_features.columns.tolist())
    sorted_non_agg['SlideName'] = combined_features['SlideName'].tolist()
    sorted_non_agg['Fibrosis Score'] = combined_features['Fibrosis Score'].tolist()
    #sorted_non_agg = combined_features.sort_values(by=['Fibrosis Score','SlideName'],ascending = False)
    sorted_non_agg = sorted_non_agg.sort_values(by= ['Fibrosis Score','SlideName'],ascending=False)
    
    additional_features= ['Collagen Area Ratio']

    #non_agg_selected_features.extend(additional_features)
    #agg_selected_features.extend(additional_features)

    for feat in numeric_combined_features.columns.tolist():
        fig = go.Figure()

        slide_idxes = np.unique(sorted_non_agg['SlideName'].tolist(),return_index=True)[1]
        slides = [sorted_non_agg['SlideName'].tolist()[i] for i in sorted(slide_idxes)]
        for slide in slides:
            fib_val = np.unique(sorted_non_agg[sorted_non_agg['SlideName'].str.match(slide)]['Fibrosis Score']).tolist()[0]
            fig.add_trace(
                go.Violin(
                    x = sorted_non_agg[sorted_non_agg['SlideName'].str.match(slide)][feat].values,
                    line_color = colors[unique_fib_scores.index(fib_val)],
                    name = slide
                )
            )

        fig.update_traces(orientation = 'h', side = 'positive', width = 3, points = False)
        fig.update_layout(
            xaxis_showgrid=False, 
            xaxis_zeroline = False, 
            height = 900, 
            title = f'{feat} distribution across slides (color indicating fibrosis score)',
            xaxis_title = f'Scaled {feat} value',
            yaxis_title = 'Slide Name'
            )
        fig.write_image(nonagg_fig_output_path+f'{feat}.png')

    """
    for feat in numeric_combined_features.columns.tolist():
        fig = go.Figure()

        slide_idxes = np.unique(sorted_non_agg['SlideName'].tolist(),return_index=True)[1]
        slides = [sorted_non_agg['SlideName'].tolist()[i] for i in sorted(slide_idxes)]
        for slide in slides:
            fib_val = np.unique(sorted_non_agg[sorted_non_agg['SlideName'].str.match(slide)]['Fibrosis Score']).tolist()[0]
            fig.add_trace(
                go.Violin(
                    x = sorted_non_agg[sorted_non_agg['SlideName'].str.match(slide)][feat].values,
                    line_color = colors[unique_fib_scores.index(fib_val)],
                    name = slide
                )
            )

        fig.update_traces(orientation = 'h', side = 'positive', width = 3, points = False)
        fig.update_layout(xaxis_showgrid=False, xaxis_zeroline = False, height = 900, title = f'{feat} distribution across slides (color indicating fibrosis score)')
        fig.write_image(agg_fig_output_path+f'{feat}.png')
    """

    for feat in numeric_combined_features.columns.tolist():
        fig = go.Figure()

        fib_idxes = np.unique(sorted_non_agg['Fibrosis Score'].tolist(),return_index=True)[1]
        fibs = [sorted_non_agg['Fibrosis Score'].tolist()[i] for i in sorted(fib_idxes)]
        for fib in fibs:
            fig.add_trace(
                go.Violin(
                    x = sorted_non_agg[sorted_non_agg['Fibrosis Score']==fib][feat].values,
                    line_color = colors[unique_fib_scores.index(fib)],
                    name = f'Fibrosis Score: {fib}'
                )
            )

        fig.update_traces(orientation = 'h', side = 'positive', width = 3, points = False)
        fig.update_layout(
            xaxis_showgrid=False, 
            xaxis_zeroline = False, 
            height = 900, 
            title = f'{feat} distribution across fibrosis scores',
            xaxis_title = f'Scaled {feat} value',
            yaxis_title = 'Pathologist Fibrosis Score'
            )
        fig.write_image(nonagg_fig_output_path+f'{feat}_fib_score.png')

    for feat in numeric_combined_features.columns.tolist():
        fig = go.Figure()

        fib_idxes = np.unique(sorted_non_agg['Fibrosis Score'].tolist(),return_index=True)[1]
        fibs = [sorted_non_agg['Fibrosis Score'].tolist()[i] for i in sorted(fib_idxes)]
        for fib in fibs:
            fig.add_trace(
                go.Violin(
                    x = sorted_non_agg[sorted_non_agg['Fibrosis Score']==fib][feat].values,
                    line_color = colors[unique_fib_scores.index(fib)],
                    name = fib
                )
            )

        fig.update_traces(orientation = 'h', side = 'positive', width = 3, points = False)
        fig.update_layout(
            xaxis_showgrid=False,
            xaxis_zeroline = False, 
            height = 900, 
            title = f'{feat} distribution across fibrosis scores',
            xaxis_title = f'Scaled {feat} value',
            yaxis_title = 'Pathologist Fibrosis Score'
            )
        fig.write_image(agg_fig_output_path+f'{feat}_fib_score.png')

    # Generating umap
    """
    umap_reducer = UMAP()
    embedding = umap_reducer.fit_transform(scaled_nonagg_features)
    embedding_df = pd.DataFrame({
        'UMAP_X': embedding[:,0],
        'UMAP_Y': embedding[:,1],
        'SlideName': combined_features['SlideName'].tolist(),
        'Fibrosis Score': combined_features['Fibrosis Score'].tolist()
    })
    umap_fig = px.scatter(
        data_frame = embedding_df,
        x = 'UMAP_X',
        y = 'UMAP_Y',
        color = 'Fibrosis Score'
    )
    umap_fig.write_image(output_path+'UMAP_FibrosisScore_NonAgg.png')

    umap_fig = px.scatter(
        data_frame = embedding_df,
        x = 'UMAP_X',
        y = 'UMAP_Y',
        color = 'SlideName'
    )
    umap_fig.write_image(output_path+'UMAP_SlideName_NonAgg.png')
    """

if __name__=="__main__":
    main()












































