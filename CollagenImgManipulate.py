from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from CollagenPatch import patch_image
import subprocess

def main():
    
    # Base path to the folder containing subfolders
    base_path = "/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH"

    # Names of the subfolders
    subfolders = ["B", "F", "M"]


    # List of image file paths from all subfolders
    B_image_files = [os.path.join(base_path, "B", f) for f in os.listdir(os.path.join(base_path, "B")) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

    # Determine the number of rows and columns for subplots
    num_images = len(B_image_files)  
    num_cols = 3
    num_rows = num_images

    # Create a figure with the required number of subplots
    plt.figure(figsize=(15, 5 * num_rows))
    i = 1

    for b_image in B_image_files: #zip(all_image_files[0], all_image_files[1], all_image_files[2]):
        
        image_basename = os.path.basename(b_image)
        f_image = os.path.join(base_path, "F", image_basename)
        m_image = os.path.join(base_path, "C", image_basename).replace('.jpg', '.tif')
        
        c_image = [b_image, f_image, m_image]
        
        patch_image(c_image)
        
        # Path to your shell script
        script_path = './Run_Collagen_Seg.sh'

        # Run the script
        result = subprocess.run([script_path], capture_output=True, text=True, shell=True)

        # Print the output and error (if any)        
        print("Error:", result.stderr)

        
    #     # Loop through each subplot (1, 2, 3)
    #     for image_path, folder_name in zip(c_image, subfolders):
            
    #         # Open the image
    #         img = Image.open(image_path)
    #         print(img.size)
            
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
