"""

Stitching and downsampling patches and predictions

"""

import os
import sys
import numpy as np

from skimage.transform import resize
from skimage.filters import gaussian, threshold_otsu

from PIL import Image
from tqdm import tqdm


def get_patch_coordinates(patch_name):
    """
    Getting Y and X coordinates from patch names for reconstruction
    """
    coords = patch_name.split('.sci ')[-1].replace('_prediction.tif','')
    try:
        x_coord = int(float(coords.split(' ')[-1].replace('X','').lstrip('0')))
    except ValueError:
        x_coord = 0

    try:
        y_coord = int(float(coords.split(' ')[0].replace('Y','').lstrip('0')))
    except ValueError:
        y_coord = 0

    return y_coord, x_coord


def main():
    
    base_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/020524_DUET_Patches/'
    slides = os.listdir(base_dir)
    #slides = ['24H Part 1']
    print(f'Found: {len(slides)} slides')

    b_dir = '/B/'
    f_dir = '/F/'
    pred_dir = '/Results/Ensemble_RGB/Testing_Output/'
    out_dir = '/Stitched_and_Downsampled/'

    patch_overlap = 150
    downsample = 10

    with tqdm(slides,total=len(slides)) as pbar:
        for slide_idx,slide in enumerate(slides):
            
            pbar.set_description(f'Working on: {slide}, {slide_idx}/{len(slides)}')

            slide_pred_dir = base_dir+slide+pred_dir
            slide_b_dir = base_dir+slide+b_dir
            slide_f_dir = base_dir+slide+f_dir
            slide_out_dir = base_dir+slide+out_dir

            try:
                if not os.path.exists(slide_out_dir):
                    os.makedirs(slide_out_dir)

                # Checking if there's anything in each pred patch
                checked_names = []
                for p in os.listdir(slide_pred_dir):
                    patch_img = np.array(Image.open(slide_pred_dir+p))
                    patch_shape = np.shape(patch_img)

                    if np.sum(patch_img)>0:
                        checked_names.append(p)
                
                print(f'Patches with collagen: {len(checked_names)}')
                x_coords = []
                y_coords = []
                for c in checked_names:
                    y_coord, x_coord = get_patch_coordinates(c)
                    x_coords.append(x_coord)
                    y_coords.append(y_coord)
                
                # Normalizing patch coordinates to be from min-->max
                y_coords = [i-min(y_coords) for i in y_coords]
                x_coords = [i-min(x_coords) for i in x_coords]

                # Determining max width and height of stitched image (number of patches)
                max_width = max(x_coords)+1
                max_height = max(y_coords)+1
                #print(f'patch height: {max_height}, patch width: {max_width}')

                pixel_width = (max_width * patch_shape[1]) - ((max_width-1) * patch_overlap)
                pixel_height = (max_height * patch_shape[0]) - ((max_height-1) * patch_overlap)
                #print(f'pixel height: {pixel_height}, pixel width: {pixel_width}')

                # Initializing downsampled mask:
                stitched_downsampled_mask = np.zeros((int(pixel_height/downsample),int(pixel_width/downsample)),dtype=np.uint8)
                stitched_downsampled_bf = np.repeat(np.zeros_like(stitched_downsampled_mask)[:,:,None],repeats=3,axis=-1)
                stitched_downsampled_f = np.zeros_like(stitched_downsampled_bf)

                """
                if os.path.exists(slide_out_dir+f'{slide}_Stitched_BF.tif'):
                    stitch_inputs = False
                else:
                    stitch_inputs = True
                """
                stitch_inputs = True

                #print(f'downsampled mask size: {stitched_downsampled_mask.shape}')
                for y,x,name in zip(y_coords,x_coords,checked_names):
                    checked_patch = np.array(Image.open(slide_pred_dir+name))[0:-(patch_overlap+1),0:-(patch_overlap+1)]
                    resized_patch = resize(checked_patch,output_shape = [int(checked_patch.shape[0]/downsample),int(checked_patch.shape[1]/downsample)])

                    # Where should this patch go?
                    y_start = int(y*resized_patch.shape[0])
                    x_start = int(x*resized_patch.shape[1])
                    #print(f'y: {y}, y_start: {y_start}, y_end: {y_start+resized_patch.shape[0]}')
                    #print(f'x: {x}, x_start: {x_start}, x_end: {x_start+resized_patch.shape[1]}')

                    stitched_downsampled_mask[y_start:int(y_start+resized_patch.shape[0]),x_start:int(x_start+resized_patch.shape[1])] += np.uint8(255*resized_patch)
                    
                    if stitch_inputs:
                        checked_bf = np.array(Image.open(slide_b_dir+name.replace('_prediction.tif','.jpg')))[0:-(patch_overlap+1),0:-(patch_overlap+1),:]
                        resized_bf = resize(checked_bf,output_shape = [int(checked_patch.shape[0]/downsample),int(checked_patch.shape[1]/downsample),3])
                        checked_f = np.array(Image.open(slide_f_dir+name.replace('_prediction.tif','.jpg')))[0:-(patch_overlap+1),0:-(patch_overlap+1),:]
                        resized_f = resize(checked_f,output_shape = [int(checked_patch.shape[0]/downsample),int(checked_patch.shape[1]/downsample),3])

                        stitched_downsampled_bf[y_start:int(y_start+resized_patch.shape[0]),x_start:int(x_start+resized_patch.shape[1]),:] += np.uint8(255*resized_bf)
                        stitched_downsampled_f[y_start:int(y_start+resized_patch.shape[0]),x_start:int(x_start+resized_patch.shape[1]),:] += np.uint8(255*resized_f)


                Image.fromarray(stitched_downsampled_mask).save(slide_out_dir+f'{slide}_Stitched_Output.tif')
                
                if stitch_inputs:
                    Image.fromarray(stitched_downsampled_bf).save(slide_out_dir+f'{slide}_Stitched_BF.tif')
                    Image.fromarray(stitched_downsampled_f).save(slide_out_dir+f'{slide}_Stitched_F.tif')

                    # Constructing virtual trichrome
                    # Gaussian smoothing
                    #smoothed_mask = np.uint8(gaussian(stitched_downsampled_mask,sigma = 1,preserve_range=True))
                    #Image.fromarray(smoothed_mask).save(slide_out_dir+f'{slide}_smoothed.tif')

                    # Thresholding
                    #thresh_mask = np.uint8(smoothed_mask > 1.25*threshold_otsu(smoothed_mask))[:,:,None]
                    #thresh_mask = np.repeat(thresh_mask,repeats=3,axis=-1)
                    #Image.fromarray(255*thresh_mask).save(slide_out_dir+f'{slide}_thresh_mask.tif')

                    # Multiplying to green channel of DUET image
                    #threshed_f = stitched_downsampled_f[:,:,1] * thresh_mask[:,:,0]
                    #new_blue = np.repeat(threshed_f.copy()[:,:,None],repeats=3,axis=-1)
                    #new_blue[:,:,0] = np.zeros_like(stitched_downsampled_mask)
                    #new_blue[:,:,1] = np.zeros_like(stitched_downsampled_mask)

                    #Image.fromarray(new_blue).save(slide_out_dir+f'{slide}_new_blue.tif')

                    #virtual_trichrome = stitched_downsampled_bf.copy()
                    #virtual_trichrome = np.where(thresh_mask>0,new_blue,virtual_trichrome)

                    #Image.fromarray(virtual_trichrome).save(slide_out_dir+f'{slide}_Virtual_Tri.tif')

            except NotADirectoryError:
                print('not a directory')

            pbar.update(1)

if __name__ == '__main__':
    print('executing main')
    main()








