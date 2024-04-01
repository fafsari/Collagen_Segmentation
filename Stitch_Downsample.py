"""

Stitching and downsampling patches and predictions

"""

import os
import sys
import numpy as np

from skimage.transform import resize

from PIL import Image


def main():
    
    base_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/020524_DUET_Patches/'
    slides = os.listdir(base_dir)
    print(f'Found: {len(slides)} slides')

    b_dir = '/B/'
    f_dir = '/F/'
    pred_dir = '/Results/Ensemble_RGB/Testing_Output/'
    out_dir = '/Stitched_and_Downsampled/'

    patch_overlap = 150
    downsample = 10

    for slide in slides:
        print(f'Working on {slide}')

        slide_pred_dir = base_dir+slide+pred_dir
        slide_b_dir = base_dir+slide+b_dir
        slide_f_dir = base_dir+slide+f_dir
        slide_out_dir = base_dir+slide+out_dir

        try:
            # Checking if there's anything in each pred patch
            checked_names = []
            for p in os.listdir(slide_pred_dir):

                patch_img = np.array(Image.open(slide_pred_dir+p))

                if np.sum(patch_img)>0:

                    checked_names.append(p)
            
            print(f'Patches with collagen: {len(checked_names)}')
            

        except NotADirectoryError:
            print('not a directory')



if __name__ == '__main__':
    print('executing main')
    main()








