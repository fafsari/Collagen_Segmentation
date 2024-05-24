from PIL import Image
import numpy as np
import os
import gc
import numpy as np
from math import floor
import matplotlib.pyplot as plt


def is_blank_image(img):
    
    image_array = np.array(img)

    # Check if all pixels are 0 or all 255
    return np.all(image_array == 0) or np.all(image_array == 255)

def stitch_image_patches(base_name, patches_dir, y_coords, x_coords, original_image_size, patch_size):
    """
    Stitches image patches together into a big matrix.

    Args:
        image_patches (list of numpy arrays): List of image patches.
        y_coords (list of int): List of y-coordinates for each patch.
        x_coords (list of int): List of x-coordinates for each patch.
        original_image_size (tuple): Original image size (height, width).
        patch_size (tuple): Patch size (height, width).

    Returns:
        numpy array: The stitched image matrix.
    """
    # Initialize an empty canvas for stitching
    stitched_image = np.zeros((original_image_size[1], original_image_size[0]), dtype=np.float32)
    patch_counts = np.zeros((original_image_size[1], original_image_size[0]), dtype=np.float32)
    
    
    extension = '.'+base_name.split('.')[-1]
    
    i = 0
    # Iterate through each patch and place it in the stitched image
    for x in x_coords:
        for y in y_coords:
            
            # patch_name = f"Test_Example_{base_name.split(extension)[0]}_{y}_{x}{extension}"
            patch_name = f"Test_Example_{base_name.replace('.jpg', '.tif')}_{y}_{x}.tif"
            # print(f"patch_name is stitching: {patch_name}")
            patch = Image.open(os.path.join(patches_dir, patch_name))
            patch = np.array(patch)  # Convert to numpy array
            
            if patch.shape != tuple(patch_size):
                print(f"Warning: Patch {patch_name} size mismatch. Expected {patch_size}, got {patch.shape}.")
                continue
            
            try:
                stitched_image[y:y+patch_size[1], x:x+patch_size[0]] += patch
                patch_counts[y:y+patch_size[1], x:x+patch_size[0]] += 1
            except Exception as e:
                print(f"Error: {e}")
                i += 1
    
    print(i)    
    # Compute the average by dividing by the number of patches contributing to each pixel
    stitched_image /= np.maximum(patch_counts, 1)

    # Convert back to uint8 (if needed)
    stitched_image = np.clip(stitched_image, 0, 255).astype(np.uint8)

    return stitched_image


# Base path to the folder containing subfolders F and B images
base_path = "/orange/pinaki.sarder/f.afsari/Farzad_Fibrosis/Kidney Biopsies 05-21-24"

case_ID = "10H"

# Directory containing the predicted patches (the output_dir of the test phase)
patches_dir = f"{base_path}/Prediction Masks/{case_ID}/Results/Ensemble_MIT_RGB/Testing_Output"
# patches_dir = base_path+"/Patches/C"
output_dir = f"{base_path}/Prediction Masks/{case_ID}/Results/Ensemble_MIT_RGB/Pred" 


# List of image file paths 
original_image_files = [os.path.join(base_path, f"DUET Scan Images/{case_ID} F", f) 
                        for f in os.listdir(os.path.join(base_path, f"DUET Scan Images/{case_ID} F")) 
                        if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

print(f" Number of images to stitch: {len(original_image_files)}")

original_image_size = (3384, 2708)

patch_size = (512, 512)

# Overlap percentage, hardcoded patch size
patch_batch = 0.25
# Correction for downsampled (10X as opposed to 20X) data
downsample_level = 0.5
stride = [int(patch_size[0]*(1-patch_batch)*downsample_level), int(patch_size[1]*(1-patch_batch)*downsample_level)]
# print("stride:", stride)

# Calculating and storing patch coordinates for each image and reading those regions at training time :/
n_patches = [1+floor((original_image_size[0]-patch_size[0])/stride[0]), 1+floor((original_image_size[1]-patch_size[1])/stride[1])]
start_coords = [0,0]

x_coords = [int(start_coords[0]+(i*stride[0])) for i in range(0,n_patches[0])]
y_coords = [int(start_coords[1]+(i*stride[1])) for i in range(0,n_patches[1])]
x_coords.append(int(original_image_size[0]-patch_size[0]))
y_coords.append(int(original_image_size[1]-patch_size[1]))

# create figure 
fig = plt.figure(figsize=(15, 10)) 
i = 1
# Loop over each original image file
for original_image_file in original_image_files:
        
    # Base name of the original image
    base_name = os.path.basename(original_image_file)
    print(f"Stitching {base_name}.....")
    extension = '.'+base_name.split('.')[-1]
    
    img = Image.open(original_image_file)
    if is_blank_image(img):
        print(f"{base_name} is ignored, blank!")        
        continue
    
    
    # Gather all patch files related to the current image
    # image_patches = [os.path.join(patches_dir, f) for f in os.listdir(patches_dir) if base_name.split(extension)[0] in f]
    # image_patches = [os.path.join(patches_dir, f) for f in os.listdir(patches_dir) if base_name.split('.jpg')[0] in f]
    # print(len(image_patches))
    
    # Stitch the patches
    stitched_image = stitch_image_patches(base_name, patches_dir, y_coords, x_coords, original_image_size, patch_size)
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Save the stitched image
    stitched_image = Image.fromarray(stitched_image)
    stitched_image.save(os.path.join(output_dir, base_name))
    
    print("Image patches stitched successfully!")
    
    # Optionally display the stitched image    
    # fig.add_subplot(1, 2, i)         
    plt.figure(figsize=(15,10))
    plt.imshow(stitched_image)
    plt.show()
    i += 1
    
    # Free up memory used by the stitched image
    del stitched_image
    gc.collect()

