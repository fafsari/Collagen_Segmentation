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
from io import BytesIO, StringIO

from skimage.morphology import remove_small_objects
from skimage import measure
import textwrap
from base64 import b64decode


def gen_layout():

    upload_style = {
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'fontSize':8
        }

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
                                )
                            ])
                        ])
                    ],md=8, style = {'maxHeight':'100vh','overflow':'scroll'}),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Image options'),
                            dbc.CardBody([
                                dbc.Label('Output folder:',html_for = 'output-folder'),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Loading(
                                            dcc.Upload(
                                                id='upload-image',
                                                children = html.Div([
                                                    'Drag and Drop or ',
                                                    html.A('Select Image File')
                                                ]),
                                                style = upload_style,
                                                multiple=False
                                            )
                                        ),
                                        html.Div(id='image-upload-status')
                                    ],md=6),
                                    dbc.Col([
                                        dcc.Loading(
                                            dcc.Upload(
                                                id = 'upload-mask',
                                                children = html.Div([
                                                    'Drag and Drop or ',
                                                    html.A('Select Mask File')
                                                ]),
                                                style = upload_style,
                                                multiple = False
                                            )
                                        ),
                                        html.Div(id='upload-mask-status')
                                    ],md=6)
                                ],align='center'),
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
                                    dbc.Button('Download New Mask',id='download-mask-butt',n_clicks=0,disabled=False),
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

        self.app.run_server(host='0.0.0.0',debug=False, use_reloader=False,port=8050)

    def add_callbacks(self):

        self.app.callback(
            [Output('image-figure','figure'),
             Output('image-upload-status','children'),
             Output('upload-mask-status','children'),
             Output('min-size','max')],
            [Input('upload-image','contents'),
             Input('upload-mask','contents'),
             Input('trans-slider','value'),
             Input('pred-thresh','value'),
             Input('min-size','value')],
             prevent_initial_call=True
        )(self.update_image)

        self.app.callback(
            Output('download-mask','data'),
            Input('download-mask-butt','n_clicks'),
            prevent_initial_call=True
        )(self.download_mask)

    def load_file(self,image_contents):

        #try:
        # Reading in an uploaded image file using PIL and BytesIO
        # Crucial detail for reading bytes object
        #https://github.com/plotly/dash-canvas/blob/master/dash_canvas/utils/io_utils.py?source=post_page-----929c72330716--------------------------------
        new_image = Image.open(BytesIO(b64decode(image_contents[22:])))
        output_status = dbc.Alert('Success!',color='success')
        #except:
        #   new_image = None
        #    output_status = dbc.Alert('Uh Oh! There was an error reading your upload!',color='warning')

        return new_image, output_status

    def update_image(self,image_upload, mask_upload, trans_val,thresh_val,min_size):
        
        print(f'triggered_id: {ctx.triggered_id}')
        self.transparency = trans_val
        self.thresh_val = thresh_val
        self.min_size = min_size
        mask_status = no_update
        image_status = no_update

        if ctx.triggered_id=='upload-image':

            self.current_image, image_status = self.load_file(image_upload)
            mask_status = no_update

        elif ctx.triggered_id=='upload-mask':
            
            self.current_mask, mask_status = self.load_file(mask_upload)
            image_status = no_update

        # Generating new overlay based on provided inputs
        self.current_overlaid_image = self.create_overlay()
        img_fig = go.Figure(
            data = px.imshow(self.current_overlaid_image)['data'],
            layout = {'margin':{'b':0,'l':0,'r':0}}
        )
        img_fig.update_layout(
            title = {
                'text':'<br>'.join(
                    textwrap.wrap(
                    f'Current image with: prediction threshold: {self.thresh_val} and minimum size: {self.min_size}',
                    width = 50
                    )
                )
            }
        )

        return img_fig, image_status, mask_status, self.max_size_slider

    def create_overlay(self):
        # Creating new overlaid mask from these inputs
        print(f'Threshold: {self.thresh_val}')
        print(f'self.current_mask is None: {self.current_mask is None}')
        print(f'self.current_image is None: {self.current_image is None}')
        if not self.current_mask is None and not self.current_image is None:
            threshed_pred = np.array(self.current_mask).copy()
            thresh_mask = np.ones_like(threshed_pred)
            thresh_mask[threshed_pred<=int(255*self.thresh_val)] = 0.

            # Adding small object filtering to mask
            #thresh_mask = remove_small_objects(thresh_mask>0,self.min_size)
            labels_mask = measure.label(thresh_mask>0)
            regions = measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area,reverse=True)
            self.max_size_slider = len(regions)
            if self.min_size>0:
                labels_mask = measure.label(thresh_mask>0)
                regions = measure.regionprops(labels_mask)
                regions.sort(key=lambda x: x.area,reverse=True)
                self.max_size_slider = len(regions)
                include_regions_number = self.min_size
                if len(regions)>1:
                    for rg in regions[include_regions_number:]:
                        labels_mask[rg.coords[:,0],rg.coords[:,1]] = 0
                
                thresh_mask = thresh_mask*(labels_mask>0)

            threshed_pred_rgb = 255*self.cm(np.uint8(threshed_pred))[:,:,0:3]
            threshed_pred_rgb[thresh_mask==0,:] = 0.

            zero_mask = np.where(thresh_mask==0,0,int(self.transparency*255))
            pred_4d = np.concatenate((threshed_pred_rgb,zero_mask[:,:,None]),axis=-1)
            pred_rgba = Image.fromarray(np.uint8(pred_4d),'RGBA')

            img_4d = self.current_image.convert('RGBA')
            img_4d.paste(pred_rgba, mask=pred_rgba)
        elif not self.current_image is None:
            img_4d = self.current_image

        elif not self.current_mask is None:
            img_4d = self.current_mask
        
        else:
            img_4d = Image.fromarray(np.uint8(np.zeros(512)))
            
        return img_4d

    def download_mask(self,download_click):

        # Getting current_mask: 
        processed_mask = np.array(self.current_mask).copy()
        processed_mask[processed_mask<=255*self.thresh_val] = 0.0

        if self.min_size>0:
            labels_mask = measure.label(processed_mask>0)
            regions = measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area,reverse=True)
            self.max_size_slider = len(regions)
            include_regions_number = self.min_size
            if len(regions)>1:
                for rg in regions[include_regions_number:]:
                    labels_mask[rg.coords[:,0],rg.coords[:,1]] = 0
            
            processed_mask = processed_mask*(labels_mask>0)

        processed_mask = np.repeat(processed_mask[:,:,None],repeats=3,axis=-1)
        processed_mask = Image.fromarray(processed_mask)

        image_name = 'Processed_Collagen_Mask.tif'
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






