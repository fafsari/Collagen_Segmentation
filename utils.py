import os
import torch

root_path = "/blue/pinaki.sarder/f.afsari/4-DUET/database"

def save_tensor(tensor, directory, tensor_name):
    """Helper function to save a tensor."""
    tensor_path = os.path.join(directory, f"{tensor_name}.pt")
    torch.save(tensor, tensor_path)

def save_train_test(dataset_train_s, dataset_valid_s, dataset_train_t, dataset_valid_t):
    
    # Create folders: Source, Target, and then subfolders train and test at each
    os.makedirs(os.path.join(root_path, "Source", "train"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "Source", "test"), exist_ok=True)
    
    os.makedirs(os.path.join(root_path, "Target", "train"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "Target", "test"), exist_ok=True)
    
    # Save source train dataset
    for x, y, image_name in dataset_train_s:
        save_tensor(x, os.path.join(root_path, "Source", "train"), image_name)
        save_tensor(y, os.path.join(root_path, "Source", "train"), f"{image_name}_mask")

    # Save source test dataset
    for x, y, image_name in dataset_valid_s:
        save_tensor(x, os.path.join(root_path, "Source", "test"), image_name)
        save_tensor(y, os.path.join(root_path, "Source", "test"), f"{image_name}_mask")

    # Save target train dataset
    for x, y, image_name in dataset_train_t:
        save_tensor(x, os.path.join(root_path, "Target", "train"), image_name)
        save_tensor(y, os.path.join(root_path, "Target", "train"), f"{image_name}_mask")

    # Save target test dataset
    for x, y, image_name in dataset_valid_t:
        save_tensor(x, os.path.join(root_path, "Target", "test"), image_name)
        save_tensor(y, os.path.join(root_path, "Target", "test"), f"{image_name}_mask")

# Example usage
# save_train_test(dataset_train_s, dataset_valid_s, dataset_train_t, dataset_valid_t)
