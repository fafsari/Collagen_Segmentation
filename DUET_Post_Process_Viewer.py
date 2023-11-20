"""

Visualization and post-processing for deep-DUET outputs

"""
import os
import sys

import numpy as np
from glob import glob

from dash import dcc, ctx, Dash, dash_table,Input,Output,State, no_update
import dash_bootstrap_components as dbc

from dash_extensions.enrich import html

import plotly.graph_objects as go
import plotly.express as px

import matplotlib as mpl

from PIL import Image

from skimage.morphology import remove_small_objects
from skimage import measure


def gen_layout():

    main_layout = html.Div(
        dbc.Container(
            id = 'app-content',
            fluid = True,
            children = [
                html.H1('DUET Viewer'),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Image with Overlaid Prediction'),
                            dbc.CardBody([
                                dcc.Graph(
                                    id = 'image-figure',
                                    figure = go.Figure()
                                ),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col(html.Div(dbc.Button('Previous Image',id='prev-image',n_clicks=0)),md=6),
                                    dbc.Col(html.Div(dbc.Button('Next Image',id='next-image',n_clicks=0)),md=6)
                                ],justify='center')
                            ])
                        ])
                    ],md=8, style = {'maxHeight':'100vh','overflow':'scroll'}),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Image options'),
                            dbc.CardBody([
                                dbc.Label('Output folder:',html_for = 'output-folder'),
                                dbc.Row([
                                    dbc.Col(dcc.Input(type='text',id='output-folder'),md=8),
                                    dbc.Col(html.Button('Load Output',id='load-output',n_clicks=0),md=4)
                                ],align='center'),
                                html.Div(id='output-folder-status'),
                                #html.Hr(),
                                #dbc.Row([
                                #    dbc.Col(html.Button(id='show-fluorescent',n_clicks=0),md=6),
                                #    dbc.Col(html.Button(id='show-brightfield',n_clicks=0),md=6)
                                #],align='center'),
                                html.Hr(),
                                dbc.Label('Prediction Transparency:',html_for='trans-slider'),
                                dcc.Slider(min=0,max=1,step=0.1,value=0.7,id='trans-slider'),
                                html.Hr(),
                                dbc.Label('Prediction Threshold:',html_for='pred-thresh'),
                                dcc.Slider(min=0,max=1,value=0,step=0.01,marks=None,tooltip={'placement':'top','always_visible':True},id='pred-thresh'),
                                html.Hr(),
                                dbc.Label('Minimum Size',html_for='min-size'),
                                dcc.Slider(min=0,max=100,value=0,step=1,marks=None,tooltip={'placement':'top','always_visible':True},id='min-size'),
                                html.Hr(),
                                dbc.Label('Download post-processed mask',html_for='download-mask'),
                                dbc.Row(
                                    dbc.Button('Download Mask',id='download-mask-butt',n_clicks=0,disabled=False),
                                    justify='center'
                                ),
                                dcc.Download(id='download-mask')
                            ])
                        ])
                    ],md = 4, style = {'maxHeight':'100vh','overflow':'scroll'})
                ])
            ]
        )
    )

    return main_layout

class DUETViewer:
    def __init__(self,
                 app,
                 layout
                 ):

        self.app = app
        self.app.layout = layout
        self.app.title = 'DUET Viewer'

        self.thresh_val = 0
        self.transparency = 0
        self.image_idx = None
        self.images = []
        self.current_image = None
        self.current_mask = None
        self.image_type = None
        self.output_folder = None
        self.image_folder = None
        self.max_size_slider = 1

        self.cm = mpl.colormaps['jet']

        self.add_callbacks()

        self.app.run_server(host='0.0.0.0',debug=True, use_reloader=True,port=8050)

    def add_callbacks(self):

        self.app.callback(
            [Output('image-figure','figure'),
             Output('output-folder-status','children'),
             Output('min-size','max')],
            [Input('load-output','n_clicks'),
             Input('prev-image','n_clicks'),
             Input('next-image','n_clicks'),
             Input('trans-slider','value'),
             Input('pred-thresh','value'),
             Input('min-size','value')],
             #Input('show-fluorescent','n_clicks'),
             #Input('show-brightfield','n_clicks')],
             State('output-folder','value'),
             prevent_initial_call=True
        )(self.update_image)

        self.app.callback(
            Output('download-mask','data'),
            Input('download-mask-butt','n_clicks'),
            prevent_initial_call=True
        )(self.download_mask)

    def update_image(self,load_butt,prev_butt,next_butt,trans_val,thresh_val,min_size,out_folder):
        
        print(f'triggered_id: {ctx.triggered_id}')
        self.transparency = trans_val
        self.thresh_val = thresh_val
        self.min_size = min_size

        if ctx.triggered_id=='load-output':
            # Loading a new output folder
            if not out_folder is None:
                # This folder just has to be for the model, they all have the same folder structure
                self.output_folder = out_folder+'Testing_Output'+os.sep
                self.image_folder = self.output_folder.replace(f'Results{os.sep}{self.output_folder.split(os.sep)[-3]}{os.sep}Testing_Output{os.sep}',f'B{os.sep}')
                print(f'Reading images from: {self.output_folder} and {self.image_folder}')
                self.predictions = glob(self.output_folder+'*')
                self.images = glob(self.image_folder+'*')
                print(f'Found: {len(self.images)} images')
                if len(self.images)>0:
                    self.threshold = thresh_val
                    self.transparency = trans_val
                    self.image_idx = 0
                    self.current_image = Image.open(self.images[0])
                    self.current_mask = Image.open(self.predictions[0])
                    self.current_overlaid_image = self.create_overlay(self.current_image,self.current_mask)

                    img_fig = go.Figure(
                        data = px.imshow(self.current_overlaid_image)['data'],
                        layout = {'margin':{'b':0,'l':0,'r':0}}
                    )
                    img_fig.update_layout(
                        title={
                            'text':f'{self.images[self.image_idx].split(os.sep)[-1]}, threshold: {self.thresh_val}'
                        }
            )

                    output_status = dbc.Alert('Found results!',color='success')
                else:

                    img_fig = go.Figure()
                    output_status = dbc.Alert('Error in filepath',color='warning')

            else:
                img_fig = no_update
                output_status = no_update
        else:
            """
            if ctx.triggered_id=='show-brightfield':
                self.image_folder = os.path.join(out_folder.split(os.sep)[0:-3]+['B'])
                self.images = glob(self.image_folder+'*')
                self.current_image = Image.open(self.images[self.image_idx])
            
            elif ctx.triggered_id=='show-fluorescent':
                self.image_folder = os.path.join(out_folder.split(os.sep)[0:-3]+['F'])
                self.images = glob(self.image_folder+'*')
                self.current_image = Image.open(self.images[self.image_idx])
            """
            if ctx.triggered_id=='next-image':
                if self.image_idx+1>=len(self.images):
                    self.image_idx = 0
                else:
                    self.image_idx+=1
                
                print(self.images[self.image_idx])
                print(self.predictions[self.image_idx])
                self.current_image = Image.open(self.images[self.image_idx])
                self.current_mask = Image.open(self.predictions[self.image_idx])

            elif ctx.triggered_id=='prev-image':
                if self.image_idx-1<=0:
                    self.image_idx = len(self.images)-1
                else:
                    self.image_idx -= 1
                
                self.current_image = Image.open(self.images[self.image_idx])
                self.current_mask = Image.open(self.predictions[self.image_idx])
            
            self.current_overlaid_image = self.create_overlay(self.current_image,self.current_mask)

            img_fig = go.Figure(
                data = px.imshow(self.current_overlaid_image)['data'],
                layout = {'margin':{'b':0,'l':0,'r':0}}
            )

            img_fig.update_layout(
                title={
                    'text':f'{self.images[self.image_idx].split(os.sep)[-1]}, threshold: {self.thresh_val}'
                }
            )
            output_status = no_update

        return img_fig, output_status, self.max_size_slider

    def create_overlay(self,image,pred):
        # Creating new overlaid mask from these inputs
        print(f'Threshold: {self.thresh_val}')
        threshed_pred = np.array(pred).copy()
        thresh_mask = np.ones_like(threshed_pred)
        thresh_mask[threshed_pred<=int(255*self.thresh_val)] = 0.

        # Adding small object filtering to mask
        #thresh_mask = remove_small_objects(thresh_mask>0,self.min_size)
        labels_mask = measure.label(thresh_mask>0)
        regions = measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area,reverse=True)
        print(f'found : {len(regions)} regions')
        self.max_size_slider = len(regions)
        if self.min_size>0:
            labels_mask = measure.label(thresh_mask>0)
            regions = measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area,reverse=True)
            print(f'found : {len(regions)} regions')
            self.max_size_slider = len(regions)
            include_regions_number = self.min_size
            print(f'including: {include_regions_number}')
            if len(regions)>1:
                for rg in regions[include_regions_number:]:
                    labels_mask[rg.coords[:,0],rg.coords[:,1]] = 0
            
            thresh_mask = thresh_mask*(labels_mask>0)

        threshed_pred_rgb = 255*self.cm(np.uint8(threshed_pred))[:,:,0:3]
        threshed_pred_rgb[thresh_mask==0,:] = 0.

        zero_mask = np.where(thresh_mask==0,0,int(self.transparency*255))
        pred_4d = np.concatenate((threshed_pred_rgb,zero_mask[:,:,None]),axis=-1)
        pred_rgba = Image.fromarray(np.uint8(pred_4d),'RGBA')

        img_4d = image.convert('RGBA')
        img_4d.paste(pred_rgba, mask=pred_rgba)

        return img_4d

    def download_mask(self,download_click):

        # Getting current_mask: 
        processed_mask = np.array(self.current_mask).copy()
        processed_mask[processed_mask<=255*self.thresh_val] = 0.0

        if self.min_size>0:
            labels_mask = measure.label(processed_mask>0)
            regions = measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area,reverse=True)
            print(f'found : {len(regions)} regions')
            self.max_size_slider = len(regions)
            include_regions_number = self.min_size
            print(f'including: {include_regions_number}')
            if len(regions)>1:
                for rg in regions[include_regions_number:]:
                    labels_mask[rg.coords[:,0],rg.coords[:,1]] = 0
            
            processed_mask = processed_mask*(labels_mask>0)

        processed_mask = np.repeat(processed_mask[:,:,None],repeats=3,axis=-1)
        print(f'shape of processed_mask: {np.shape(processed_mask)}')
        processed_mask = Image.fromarray(processed_mask)

        image_name = self.images[self.image_idx].split(os.sep)[-1].replace('.png','_processed.png')
        processed_mask.save(f'./{image_name}')

        return dcc.send_file(f'./{image_name}')






def main():

    app_layout = gen_layout()
    stylesheets = [
        dbc.themes.LUX,
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ]
    app = Dash(__name__,external_stylesheets=stylesheets)

    duet_app = DUETViewer(app,app_layout)






if __name__ == '__main__':
    
    main()






