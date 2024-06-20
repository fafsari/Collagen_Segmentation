import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral


def smooth_black_areas(image_path, output_path):
    try:
        # Load the image in grayscale
        image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image_array is None:
            raise ValueError("Could not load the image. Please check the file path.")
        
        # Define a structuring element (kernel)
        kernel = np.ones((5, 5), np.uint8)
        
        # Apply morphological closing
        closed_image = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel)
        
        # Save the resulting image
        cv2.imwrite(output_path, closed_image)
        
        print(f'Modified image saved at: {output_path}')
    except cv2.error as e:
        print(f'OpenCV error: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')


def apply_crf(image, output_probs, num_classes=2):
    """
    Apply CRF post-processing to refine segmentation boundaries.
    
    Parameters:
    - image: Input image (H, W, C) in RGB format
    - output_probs: Model output probabilities (C, H, W)
    - num_classes: Number of classes

    Returns:
    - Refined segmentation map (H, W)
    """
    H, W = image.shape[:2]
    
    # Create a DenseCRF model
    d = dcrf.DenseCRF2D(W, H, num_classes)
    
    # Create the unary potential
    unary = unary_from_softmax(output_probs)
    d.setUnaryEnergy(unary)
    
    # Add pairwise Gaussian potentials (smoothing term)
    gaussian_pairwise_energy = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    d.addPairwiseEnergy(gaussian_pairwise_energy, compat=3)
    
    # Add pairwise bilateral potentials (appearance term)
    bilateral_pairwise_energy = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13), img=image, chdim=2)
    d.addPairwiseEnergy(bilateral_pairwise_energy, compat=10)
    
    # Perform inference
    Q = d.inference(5)
    refined_output = np.argmax(Q, axis=0).reshape((H, W))
    
    return refined_output

# Example usage
# Load the image and model output
image_path = 'path_to_image'
output_probs_path = 'path_to_output_probs'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Assume output_probs is a numpy array of shape (num_classes, H, W)
output_probs = np.load(output_probs_path)

# Apply CRF post-processing
refined_segmentation = apply_crf(image, output_probs)

num_classes = 2
# Save or visualize the refined segmentation map
refined_segmentation_path = 'path_to_save_refined_segmentation.png'
cv2.imwrite(refined_segmentation_path, refined_segmentation * 255 // (num_classes - 1))  # Save as binary image for visualization

print(f'Refined segmentation map saved at: {refined_segmentation_path}')


# File paths
base_path = '/blue/pinaki.sarder/f.afsari/4-DUET/DUET UCD PATH vs CGPL/UCD-PATH/Results/Ensemble_DA_V2_G/Stitched_Pred/'
image_name = 'HE15863-1 X 46.75844 Y 43.765'
image_path = base_path+image_name+'_prediction.tif'
output_path = base_path+image_name+'_prediction_closed.tif'

# Call the function
smooth_black_areas(image_path, output_path)
print(f'Modified image saved at: {output_path}')
