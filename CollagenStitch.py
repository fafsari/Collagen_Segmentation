from PIL import Image
import numpy as np
import os
import re

def extract_coordinates(patch_name):
    
    """
    Getting Y and X coordinates from patch names for reconstruction
    """
    coords = patch_name.replace('.tif','').split(' ')
    
    try:
        _, x_coord, y_coord = coords[-1].split('_')
    except ValueError:
        x_coord = 0

    return int(x_coord), int(y_coord)
   
def stitch_image(patch_files, original_size):
    # Create a blank canvas with the original image size
    stitched_image = Image.new('RGB', original_size)
    
    for patch_file in patch_files:
        # Open the patch image
        patch = Image.open(patch_file)
        
        # print(type(os.path.basename(patch_file)), os.path.basename(patch_file))
        
        # Extract row and column start coordinates from the file name
        coords = extract_coordinates(os.path.basename(patch_file))
        if coords:
            row_start, col_start = coords
            # Place the patch on the canvas at the correct location
            stitched_image.paste(patch, (col_start, row_start))
    
    return stitched_image

# Base path to the folder containing subfolders
base_path = "/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH"

# Directory containing the patches
patches_dir = "/blue/pinaki.sarder/f.afsari/4-DUET/Data/Results/Ensemble_Attn_RGB/UCD-PATH/Testing_Output"

# List of original image file paths
original_image_files = [os.path.join(base_path, subfolder, f) 
                        for subfolder in ["B"] 
                        for f in os.listdir(os.path.join(base_path, subfolder)) 
                        if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

# Loop over each original image file
for original_image_file in original_image_files:
    # Read the original image to get its size
    original_image = Image.open(original_image_file)
    original_size = original_image.size
    
    # Base name of the original image
    base_name = os.path.splitext(os.path.basename(original_image_file))[0]
    
    # Gather all patch files related to the current image
    patch_files = [os.path.join(patches_dir, f) 
                   for f in os.listdir(patches_dir) 
                   if f.startswith(f"Test_Example_{base_name}") and f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    # Stitch the patches together
    stitched_image = stitch_image(patch_files, original_size)
    
    stitched_folder = f"{base_path}/Stitched_Preds"
    if not os.path.isdir(stitched_folder):
        os.makedirs(stitched_folder, exist_ok=True)
        
    # Save the stitched image
    stitched_image.save(f"{stitched_folder}/{base_name}_stitched.tif")

    # Optionally display the stitched image
    stitched_image.show()
