from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from math import floor


def is_blank_image(img):
    
    image_array = np.array(img)

    # Check if all pixels are 0 or all 255
    return np.all(image_array == 0) or np.all(image_array == 255)


def patch_image(image_files, patch_dirs, case_ID, image_size=(512, 512)):
    
    # Open all images
    imgs = [Image.open(image_files[i]) for i in range(len(image_files))]
    
    img = imgs[0]
    # Patch size and overlap
    patch_size = (int(image_size[0]) , int(image_size[1])) 
    # Overlap percentage, hardcoded patch size
    # patch_size = [image_size[0], image_size[1]]
    patch_batch = 0.25
    # Correction for downsampled (10X as opposed to 20X) data
    downsample_level = 0.5
    stride = [int(patch_size[0]*(1-patch_batch)*downsample_level), int(patch_size[1]*(1-patch_batch)*downsample_level)]

    # Calculating and storing patch coordinates for each image and reading those regions at training time :/
    n_patches = [1+floor((np.shape(img)[0]-patch_size[0])/stride[0]), 1+floor((np.shape(img)[1]-patch_size[1])/stride[1])]
    start_coords = [0,0]

    row_starts = [int(start_coords[0]+(i*stride[0])) for i in range(0,n_patches[0])]
    col_starts = [int(start_coords[1]+(i*stride[1])) for i in range(0,n_patches[1])]
    row_starts.append(int(np.shape(img)[0]-patch_size[0]))
    col_starts.append(int(np.shape(img)[1]-patch_size[1]))

    # Create patches and save
    for img, image_name in zip(imgs, image_files):
        
        if is_blank_image(img):
            print(f"{image_name} is ignored, blank!")
            continue
                
        base_name = os.path.basename(image_name)
        extension = os.path.splitext(image_name)[-1]
        # print(base_name.replace(extension, ''), '-------------', os.path.splitext(image_name)[-1])
        
        if base_name.startswith(f"{case_ID}.sci"):
            patch_dir = patch_dirs[0]
        else:
            patch_dir = patch_dirs[1]

        if not os.path.isdir(patch_dir):
            os.makedirs(patch_dir, exist_ok=True)
        
        # print(os.path.join(patch_dir, base_name))
        
        for r_s in row_starts:
            for c_s in col_starts:
                # Define the region for cropping
                box = (c_s, r_s, c_s + patch_size[0], r_s + patch_size[1])
                
                # Crop and create a new patch
                patch = img.crop(box)
                
                # Create a new file name for the patch
                # base_name = os.path.splitext(image_name)[0]                
                patch_name = f"{base_name.replace(extension, '')}_{r_s}_{c_s}{extension}"
                # print(os.path.join(patch_dir, patch_name))
                # Save the patch
                patch.save(os.path.join(patch_dir, patch_name))

# Base path to the folder containing subfolders
# base_path = "/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH"
base_path = "/orange/pinaki.sarder/f.afsari/Farzad_Fibrosis/Kidney Biopsies 05-21-24/DUET Scan Images"
patch_path = "/orange/pinaki.sarder/f.afsari/Farzad_Fibrosis/Kidney Biopsies 05-21-24/Patches"

case_ID = "32H"
# Names of the subfolders
subfolders = [f"{case_ID} B", f"{case_ID} F"]


# List of image file paths from all subfolders
B_image_files = [os.path.join(base_path, subfolders[0], f) 
                 for f in os.listdir(os.path.join(base_path, subfolders[0])) 
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))
                ]


for b_image in B_image_files: #zip(all_image_files[0], all_image_files[1], all_image_files[2]):
    
    image_basename = os.path.basename(b_image)
    f_basename = image_basename.replace('.sci', '_LargeGlobalFlatfield.tif')
    f_image = os.path.join(base_path, subfolders[1], f_basename)
    # m_image = os.path.join(base_path, "M", image_basename).replace('.jpg', '.tif')
    
    c_images = [b_image, f_image]#, m_image]
    p_dirs   = [os.path.join(patch_path, subfolders[0]), os.path.join(patch_path, subfolders[1])]
    
    patch_image(c_images, p_dirs, case_ID)
    
