from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import gc
from skimage.transform import resize

def extract_coordinates(patch_name):
    """
    Getting Y and X coordinates from patch names for reconstruction
    """
    coords = patch_name.replace('.tif','').split(' ')
    try:
        _, y_coord, x_coord = coords[-1].split('_')
    except ValueError:
        x_coord = 0

    return int(y_coord), int(x_coord)

def normalize_coordinates(coords):
    """
    Normalize coordinates to start from zero.
    """
    min_coord = min(coords)
    return [coord - min_coord for coord in coords]

def stitch_image(patch_files, original_size, patch_size=(512, 512), patch_overlap=128, downsample=2):
    # Extract coordinates from patch filenames
    coordinates = [extract_coordinates(os.path.basename(f)) for f in patch_files]
    y_coords, x_coords = zip(*coordinates)

    # Normalize coordinates
    y_coords = normalize_coordinates(y_coords)
    x_coords = normalize_coordinates(x_coords)

    # Determine the dimensions of the stitched image
    max_width = max(x_coords) + 1
    max_height = max(y_coords) + 1
    pixel_width = (max_width * patch_size[1]) - ((max_width - 1) * patch_overlap)
    pixel_height = (max_height * patch_size[0]) - ((max_height - 1) * patch_overlap)
    stitched_downsampled_image = np.zeros((int(pixel_height / downsample), int(pixel_width / downsample), 3), dtype=np.uint8)

    # Stitch patches together
    for y, x, patch_file in zip(y_coords, x_coords, patch_files):
        patch = np.array(Image.open(patch_file))[:-patch_overlap, :-patch_overlap]
        resized_patch = resize(patch, output_shape=(int(patch.shape[0] / downsample), int(patch.shape[1] / downsample), 3), anti_aliasing=True)

        y_start = int(y * resized_patch.shape[0])
        x_start = int(x * resized_patch.shape[1])

        stitched_downsampled_image[y_start:y_start + resized_patch.shape[0], x_start:x_start + resized_patch.shape[1]] = np.uint8(255 * resized_patch)

    return Image.fromarray(stitched_downsampled_image)

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
    
    # Free up memory used by the original image
    del original_image
    gc.collect()
    
    # Base name of the original image
    base_name = os.path.splitext(os.path.basename(original_image_file))[0]
    
    # Gather all patch files related to the current image
    patch_files = [os.path.join(patches_dir, f) 
                   for f in os.listdir(patches_dir) 
                   if f.startswith(f"Test_Example_{base_name}") and f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    # Stitch the patches together
    stitched_image = stitch_image(patch_files, original_size, patch_size=(512, 512), patch_overlap=128, downsample=2)
    
    stitched_folder = f"{base_path}/Stitched_Preds"
    if not os.path.isdir(stitched_folder):
        os.makedirs(stitched_folder, exist_ok=True)
        
    # Save the stitched image
    stitched_image.save(f"{stitched_folder}/{base_name}_stitched.tif")

    # Optionally display the stitched image
    plt.imshow(stitched_image)
    plt.show    
    
    # Free up memory used by the stitched image
    del stitched_image
    gc.collect()
