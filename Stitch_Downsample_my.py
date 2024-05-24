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

from scipy.ndimage import distance_transform_edt
from matplotlib import cm as colormap


def get_patch_coordinates(patch_name):
    """
    Getting Y and X coordinates from patch names for reconstruction
    """
    coords = (os.path.basename(patch_name).replace('.tif','')).split(' ')
    
    try:
        _, y_coord, x_coord = coords[-1].split('_')
    except ValueError:
        x_coord = 0

    # try:
    #     y_coord = int(float(coords.split(' ')[0].replace('Y','').lstrip('0')))
    # except ValueError:
    #     y_coord = 0

    return int(y_coord), int(x_coord)

def get_distance_transform_image(output_array):
    """
    Apply distance transform, normalize, and apply colormap to output array
    """
    distance_transform_rgb = np.zeros((output_array.shape[0],output_array.shape[1],3))
    dist_output = distance_transform_edt(output_array>25)

    # Normalizing and applying colormap
    norm_dist = (dist_output-np.min(dist_output))/(np.max(dist_output))
    rgb_dist = np.uint8(255*colormap.jet(norm_dist)[:,:,0:3])
    
    rgb_dist[norm_dist==np.min(norm_dist),:] = 0 

    return rgb_dist, np.uint8(dist_output)


def main():
    
    # base_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/020524_DUET_Patches/'
    base_dir = "/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH"
    
    b_dir = '/B/'
    f_dir = '/F/'
    slide_pred_dir = '/blue/pinaki.sarder/f.afsari/4-DUET/Data/Results/Ensemble_Attn_RGB/UCD-PATH/Testing_Output/'
    out_dir = '/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH/Stitched_and_Downsampled/'
    
    # slides = os.listdir(base_dir)
    slides = [os.path.basename(slide_name).replace('.jpg', '') for slide_name in os.listdir(base_dir+b_dir)]
    # slides = [i for i in slides if os.path.isdir(base_dir+i)]
    #slides = ['20H']
    print(f'Found: {len(slides)} slides')

    
    patch_overlap = 150
    downsample = 200

    with tqdm(slides,total=len(slides)) as pbar:
        for slide_idx,slide in enumerate(slides):
            
            pbar.set_description(f'Working on: {slide}, {slide_idx}/{len(slides)}')

            # slide_pred_dir = base_dir+slide+pred_dir
            slide_b_dir = base_dir+slide+b_dir
            slide_f_dir = base_dir+slide+f_dir
            slide_out_dir = out_dir+slide

            try:
                if not os.path.exists(slide_out_dir):
                    os.makedirs(slide_out_dir)

                # Checking if there's anything in each pred patch
                checked_names = [p for p in os.listdir(slide_pred_dir) if slide in p]
                patch_shape = (512, 512)
                # checked_names = []
                # for p in os.listdir(slide_pred_dir):
                    
                #     if slide in p:                        
                #         # patch_img_path = slide_pred_dir+p
                        
                #         patch_img = np.array(Image.open(p))
                #         patch_shape = np.shape(patch_img)

                #         if np.sum(patch_img)>0:
                #             checked_names.append(p)
                
                #print(f'Patches with collagen: {len(checked_names)}')
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
                        name = name.replace('Test_Example_', '')
                        # checked_bf = np.array(Image.open(slide_b_dir+name.replace('_prediction.tif','.jpg')))[0:-(patch_overlap+1),0:-(patch_overlap+1),:]
                        checked_bf = np.array(Image.open(base_dir+"/Patches"+b_dir+name.replace(".tif", ".jpg")))[0:-(patch_overlap+1),0:-(patch_overlap+1),:]
                        resized_bf = resize(checked_bf,output_shape = [int(checked_patch.shape[0]/downsample),int(checked_patch.shape[1]/downsample),3])
                        # checked_f = np.array(Image.open(slide_f_dir+name.replace('_prediction.tif','.jpg')))[0:-(patch_overlap+1),0:-(patch_overlap+1),:]
                        checked_f = np.array(Image.open(base_dir+"/Patches"+f_dir+name.replace(".tif", ".jpg")))[0:-(patch_overlap+1),0:-(patch_overlap+1),:]
                        resized_f = resize(checked_f,output_shape = [int(checked_patch.shape[0]/downsample),int(checked_patch.shape[1]/downsample),3])

                        stitched_downsampled_bf[y_start:int(y_start+resized_patch.shape[0]),x_start:int(x_start+resized_patch.shape[1]),:] += np.uint8(255*resized_bf)
                        stitched_downsampled_f[y_start:int(y_start+resized_patch.shape[0]),x_start:int(x_start+resized_patch.shape[1]),:] += np.uint8(255*resized_f)
                    
                    pbar.update(1/(len(checked_names)))

                Image.fromarray(stitched_downsampled_mask).save(slide_out_dir+f'{slide}_Stitched_Output.tif')
                
                rgb_distance_transform, grayscale_distance_transform = get_distance_transform_image(stitched_downsampled_mask)
                Image.fromarray(rgb_distance_transform).save(slide_out_dir+f'{slide}_Stitched_Output_DT.tif')
                Image.fromarray(grayscale_distance_transform).save(slide_out_dir+f'{slide}_Stitched_Output_grayscale_DT.tif')

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

            #pbar.update(1)

if __name__ == '__main__':
    print('executing main')
    main()








