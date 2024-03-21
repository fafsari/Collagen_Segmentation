"""

Creating empty patches for reconstructing tiled slides

When Willy made the slide tiles (with pixel-matched DUET images) a lot of them were background.
When running predictions I skipped the patches that were mostly background. This goes through and creates empty (all black)
pseudo-prediction patches that can be used to fill in the blanks.

"""

import os
import numpy as np

from PIL import Image

from tqdm import tqdm

base_dir = '/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/020524_DUET_Patches/'
bf_dir = 'B/'
output_dir = 'Results/Ensemble_RGB/Testing_Output/'

# Each slide has three folders. B = Brightfield images, F = Fluorescent images, and Results = collagen predictions
image_folders = os.listdir(base_dir)

for i in tqdm(image_folders):
    # Getting contents of the B directory
    bf_images = os.listdir(base_dir + i + os.sep + bf_dir)

    # Getting contents of the output_dir
    output_images = os.listdir(base_dir + i + os.sep + output_dir)

    # Getting the difference between these two lists (predictions have a different file extension and _prediction at the end)
    missing_images = [i for i in bf_images if i.replace('.jpg','_prediction.tif') not in output_images]

    # Iterating through missing images and creating pseudo-prediction
    for m in missing_images:

        # Reading in the bf image to get the size of the patch
        non_missing = Image.open(base_dir + i + os.sep + bf_dir + m)
        width, height = non_missing.size

        pseudo_pred = Image.fromarray(np.zeros((height,width),dtype=np.uint8))

        new_name = m.replace('.jpg','_prediction.tif')
        pseudo_pred.save(base_dir + i + os.sep + output_dir + new_name)









