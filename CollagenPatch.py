from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from math import floor

def is_blank_image(img):
    
    image_array = np.array(img)

    # Check if all pixels are 0 or all 255
    return np.all(image_array == 0) or np.all(image_array == 255)

def patch_image(image_files, patch_dirs, image_size=(512, 512)):
    
    # Open both Florecent and Brightfield images
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
    for i, (img, image_name) in enumerate(zip(imgs, image_files)):
        
        # skip over all blank (all 0/255) images
        if is_blank_image(img):
            continue
        
        # base_name = os.path.basename(image_name)        
        # if "sci" in base_name:
        #     patch_dir = patch_dirs[0]
        # else:
        #     patch_dir = patch_dirs[1]
        
        patch_dir = patch_dirs[i]

        if not os.path.isdir(patch_dir):
            os.makedirs(patch_dir, exist_ok=True)
        
        # print(f"{os.path.join(patch_dir, base_name)}_i_j")
        
        # Create a new file name for the patch        
        base_name = image_name.split('/')[-1]
        extension = base_name.split('.')[-1]
        base_name = base_name.replace(f'.{extension}', '')
        # if '.jpg' in base_name:
        #     base_name = base_name.replace('.jpg', '')
        #     extension = '.jpg'
        # elif '.tif' in base_name:
        #     base_name = base_name.replace('.tif', '')
        #     extension = '.tif'
        # print(patch_dir, base_name, extension, sep='\n')
        for r_s in row_starts:
            for c_s in col_starts:
                # Define the region for cropping
                box = (c_s, r_s, c_s + patch_size[0], r_s + patch_size[1])
                
                # Crop and create a new patch
                patch = img.crop(box)
                
                patch_name = f"{base_name}_{r_s}_{c_s}.{extension}"                                    
                # print(os.path.join(patch_dir, patch_name))
                
                # Save the patch
                patch.save(os.path.join(patch_dir, patch_name))

# Base path to the folder containing subfolders
base_path = "/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH/NormalizedF"
# base_path = "/blue/pinaki.sarder/f.afsari/4-DUET/database/Frozen_sections"

# B_image_files = [os.path.join(base_path, "B") 
#                 for f in os.listdir(os.path.join(base_path, "B")) 
#                 if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))
# ]
# Names of the subfolders
# subfolders = ["10H B-n", "10H F-n"]
subfolders = os.listdir(base_path)

subfolders_B = [f for f in subfolders if 'B' in f]
subfolders_F = [f for f in subfolders if 'F' in f]

for folder_B in subfolders_B:
    print("Working on folder:", folder_B)
    # List of image file paths from folder_B
    B_image_files = [os.path.join(base_path, folder_B, f) 
                    for f in os.listdir(os.path.join(base_path, folder_B)) 
                    if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))
                    ]
    # # List of image file paths from folder_F
    # F_image_files = [os.path.join(base_path, folder_F, f) 
    #                 for f in os.listdir(os.path.join(base_path, folder_F)) 
    #                 if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))
    #                 ]

    # # Determine the number of rows and columns for subplots
    # num_images = len(B_image_files)  
    # num_cols = 3
    # num_rows = num_images

    # Create a figure with the required number of subplots
    # plt.figure(figsize=(15, 5 * num_rows))
    i = 1

    for b_image in B_image_files: 
        
        image_basename = os.path.basename(b_image)
        extension = image_basename.split('.')[-1]
        print(extension)
        # f_basename = image_basename.replace('.sci', '_LargeGlobalFlatfield.tif')
        f_image = os.path.join(base_path, folder_B.replace('B', 'F'), image_basename)
        if os.path.isdir(os.path.join(base_path, "C")):
            c_image = os.path.join(base_path, "C", image_basename).replace(extension, 'tif')        
            all_images = [b_image, f_image, c_image]
            p_dirs   = [os.path.join(base_path, "Patches25", f) for f in ["B", "F", "C"]]
        else:
            all_images = [b_image, f_image]
            p_dirs   = [os.path.join(base_path, "Patches25", f) for f in ["B", "F"]]
        # print(p_dirs)
        # print(c_images)
        patch_image(all_images, p_dirs)
        
#     # Loop through each subplot (1, 2, 3)
#     for image_path, folder_name in zip(c_image, subfolders):
        
#         # Open the image
#         img = Image.open(image_path)
        
#         # Display the image in the corresponding subplot
#         plt.subplot(num_rows, num_cols, i)  # Define the grid and subplot index
#         plt.imshow(np.array(img))
#         # plt.title(folder_name)  # Use the image file name as the title
#         plt.title(os.path.basename(image_path))  # Use the image file name as the title
#         plt.axis("off")  # Hide axis ticks and labels
#         i += 1

# # Display all images in the figure
# plt.tight_layout()  # Ensure proper spacing between subplots
# plt.show()
