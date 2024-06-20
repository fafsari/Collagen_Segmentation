import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.mse_loss(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

# Define a function to compute pairwise distances in batches
def compute_pairwise_distances(x_s, batch_size=1000):
    n = x_s.size(0)
    pairwise_distances = []
    for i in range(0, n, batch_size):
        batch_x_s = x_s[i:i+batch_size]
        distances = torch.pow(batch_x_s.unsqueeze(1) - x_s.unsqueeze(0), 2).sum(2)
        pairwise_distances.append(distances)
    pairwise_distances = torch.cat(pairwise_distances)
    
    # x_s_kernel = torch.exp(-pairwise_distances / (2 * sigma**2)).mean(dim=1)
    
    return pairwise_distances

# functions to compute class conditional MMD
# def mmd_loss(x_s, x_t):
#     # Assuming x_s and x_t are torch tensors    
#     sigma = 1.0  # Adjust sigma as needed
    
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     # print("x_s.shape:", x_s.shape)      
#     # Calculate the pairwise distance
#     pairwise_distances = compute_pairwise_distances(x_s, batch_size=1000)
#     # pairwise_distances = torch.pow(x_s.unsqueeze(1) - x_s.unsqueeze(0), 2).sum(2)
#     # Calculate the kernel values
#     x_s_kernel = torch.exp(-pairwise_distances / (2 * sigma**2)).mean(dim=1)

#     # Calculate the pairwise distance
#     # pairwise_distances = torch.pow(x_t.unsqueeze(1) - x_t.unsqueeze(0), 2).sum(2)
#     pairwise_distances = compute_pairwise_distances(x_t, batch_size=1000)
#     # Calculate the kernel values
#     x_t_kernel = torch.exp(-pairwise_distances / (2 * sigma**2)).mean(dim=1)    
    
#     mmd = torch.pow(x_s_kernel.mean() - x_t_kernel.mean(), 2)
    
#     return mmd
def mmd_loss(x_s, x_t):
    return torch.abs(x_s.mean() - x_t.mean())

def class_conditional_mmd(pred_s, pred_t):
    # threshold = 0.2
    threshold_s = threshold_otsu(pred_s.detach().cpu().numpy())
    threshold_t = threshold_otsu(pred_t.detach().cpu().numpy())
        
    # Thresholding predictions
    labels_s = torch.where(pred_s >= threshold_s, torch.tensor(1), torch.tensor(0))
    labels_t = torch.where(pred_t >= threshold_t, torch.tensor(1), torch.tensor(0))
    
    # print("label_s.shape:", labels_s.shape)
    
    num_classes = 2  # Assuming binary classification after thresholding
    mmd_losses = []

    for class_idx in range(num_classes):
    
        # Create masks for class_idx
        mask_s = (labels_s == class_idx).float()
        mask_t = (labels_t == class_idx).float()
        # Apply masks to retain the same shape as pred_s and pred_t
        x_s_c = pred_s * mask_s
        x_t_c = pred_t * mask_t
                
        # print("x_s_c.shape:", x_s_c.shape)      
        # # Compute MMD
        # mmd_loss_c = mmd_loss(x_s_c, x_t_c)
        mmd_loss_c = torch.nn.functional.mse_loss(x_s_c, x_t_c)        
        mmd_losses.append(mmd_loss_c)

    # Aggregate all losses
    mean_mmd_loss = torch.stack(mmd_losses).mean()

    return mean_mmd_loss


def adversarial_loss(discriminator, source_features, target_features):
  """
  Calculates domain adversarial loss for domain adaptation.

  Args:
      discriminator: Domain discriminator network (nn.Module).
      segmentation_network: Segmentation network (nn.Module).
      source_features: Feature representations from source domain (tensor).
      target_features: Feature representations from target domain (tensor).

  Returns:
      Domain adversarial loss (tensor).
  """
  # Forward pass through discriminator
  source_preds = discriminator(source_features)
  target_preds = discriminator(target_features)

  # Define loss function (e.g., Binary Cross Entropy)
  criterion = torch.nn.BCELoss()

  # Adversarial loss for source and target domains
  source_loss = criterion(source_preds, torch.ones_like(source_preds))
  target_loss = criterion(target_preds, torch.zeros_like(target_preds))

  # Combine losses
  total_loss = source_loss + target_loss
  return total_loss

# # Example usage:
# # Assuming pred_s, pred_t are torch tensors of grayscale segmentation predictions
# # and labels_s, labels_t are torch tensors of corresponding class labels
# mmd_loss = compute_class_conditional_mmd(pred_s, pred_t, labels_s, labels_t)

# # Incorporate mmd_loss into your total loss
# total_loss = seg_loss + lambda_seg * mmd_loss
