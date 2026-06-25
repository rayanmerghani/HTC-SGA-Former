# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.ndimage import distance_transform_edt
# import numpy as np

# class ContourWeightedLoss(nn.Module):
#     """
#     Optimized loss function for DSA vessel segmentation with:
#     - Implicit boundary weighting via distance transforms
#     - Dynamic class balancing
#     - Only 2 essential hyperparameters
#     - Improved recall-boundary precision tradeoff
    
#     Args:
#         boundary_alpha (float): Controls boundary emphasis (1.0-2.0 recommended)
#         delta (float): Smoothing factor for distance transform (1e-5 recommended)
#     """
#     def __init__(self, boundary_alpha=2, delta=1e-5):
#         super().__init__()
#         self.boundary_alpha = boundary_alpha
#         self.delta = delta
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
#     def get_boundary_weights(self, target):
#         """
#         Generate boundary weights using efficient distance transforms.
#         Returns weights emphasizing both vessel boundaries and thin structures.
#         """
#         device = target.device
#         target_np = target.squeeze(1).cpu().numpy()
#         weights = np.zeros_like(target_np)
        
#         for b in range(target.size(0)):
#             mask = (target_np[b] > 0.5).astype(np.uint8)
            
#             if mask.sum() == 0:  # No vessels case
#                 weights[b] = 1.0
#                 continue
                
#             # Get distance transforms for both foreground and background
#             fg_dist = distance_transform_edt(mask)
#             bg_dist = distance_transform_edt(1 - mask)
            
#             # Combine distances to emphasize boundaries
#             boundary_map = np.exp(-(fg_dist + bg_dist)**2 / (2*self.delta**2))
            
#             # Enhance thin structures while preserving boundaries
#             weights[b] = 1.0 + (self.boundary_alpha - 1.0) * boundary_map
            
#         return torch.from_numpy(weights).float().to(device)
    
#     def forward(self, input, target):
#         # Get dynamic boundary weights
#         with torch.no_grad():
#             weights = self.get_boundary_weights(target)
            
#         # Calculate base BCE loss
#         bce_loss = self.bce(input, target)
        
#         # Apply boundary-aware weighting
#         weighted_bce = (weights * bce_loss).mean()
        
#         # Enhanced Dice with boundary emphasis
#         prob = torch.sigmoid(input)
#         prob_flat = prob.view(-1)
#         target_flat = target.view(-1)
#         weights_flat = weights.view(-1)
        
#         intersection = (prob_flat * target_flat * weights_flat).sum()
#         union = ((prob_flat + target_flat) * weights_flat).sum()
#         dice_loss = 1.0 - (2. * intersection + self.delta) / (union + self.delta)
        
#         # Dynamic loss balancing based on current error
#         with torch.no_grad():
#             recall_error = (target_flat * (1 - (prob_flat > 0.5).float())).sum()
#             boundary_error = ((weights_flat > 1.0) * (prob_flat > 0.5).float() * 
#                             (1 - target_flat)).sum()
#             balance_factor = torch.sigmoid(recall_error - boundary_error)
            
#         # Final balanced loss
#         return balance_factor * weighted_bce + (1 - balance_factor) * dice_loss
    


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scipy.ndimage import distance_transform_edt
# import numpy as np

# class ContourWeightedLoss(nn.Module):
#     """
#     Enhanced loss function for DSA vessel segmentation with:
#     - Boundary emphasis via distance transforms
#     - Class balancing (pos_weight for BCE)
#     - Configurable loss components (BCE, Dice, boundary weights)
#     - Dynamic or fixed loss balancing

#     Args:
#         boundary_alpha (float): Boundary emphasis (0.5-2.0). Higher = stricter boundaries.
#         delta (float): Smoothing for distance transform (1e-6 to 1e-4).
#         pos_weight (float): Weight for foreground pixels (>=1). Helps with class imbalance.
#         use_dynamic_balance (bool): Auto-balance BCE/Dice. If False, uses fixed_weights.
#         fixed_weights (tuple): Manual (BCE_weight, Dice_weight) if use_dynamic_balance=False.
#     """
#     def __init__(self, boundary_alpha=1.5, delta=1e-5, pos_weight=5.0, 
#                  use_dynamic_balance=True, fixed_weights=(0.7, 0.3), temp=1000):
#         super().__init__()
#         self.boundary_alpha = boundary_alpha
#         self.delta = delta
#         self.pos_weight = pos_weight
#         self.use_dynamic_balance = use_dynamic_balance
#         self.fixed_weights = fixed_weights
#         self.temp = 1000
        
#         # Initialize BCE with class balancing
#         self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([self.pos_weight]))
        
#     def get_boundary_weights(self, target):
#         """Generate boundary weights emphasizing thin structures and edges."""
#         device = target.device
#         target_np = target.squeeze(1).cpu().numpy()
#         weights = np.ones_like(target_np)  # Default = 1 (no weighting)
        
#         for b in range(target.size(0)):
#             mask = (target_np[b] > 0.5).astype(np.uint8)
#             if mask.sum() == 0:  # Skip if no vessels
#                 continue
                
#             # Distance transforms for foreground/background
#             fg_dist = distance_transform_edt(mask)
#             bg_dist = distance_transform_edt(1 - mask)
            
#             # Boundary map (alternative: linear 1/(1+dist))
#             boundary_map = np.exp(-(fg_dist + bg_dist)**2 / (2 * self.delta**2))
            
#             # Apply boundary emphasis
#             weights[b] = 1.0 + (self.boundary_alpha - 1.0) * boundary_map
            
#         return torch.from_numpy(weights).float().to(device)
    
#     def forward(self, input, target):
#         # Boundary-aware weights
#         with torch.no_grad():
#             weights = self.get_boundary_weights(target)
        
#         # Weighted BCE loss
#         bce_loss = self.bce(input, target)
#         weighted_bce = (weights * bce_loss).mean()
        
#         # Enhanced Dice loss (boundary-weighted)
#         prob = torch.sigmoid(input)
#         prob_flat = prob.view(-1)
#         target_flat = target.view(-1)
#         weights_flat = weights.view(-1)
        
#         intersection = (prob_flat * target_flat * weights_flat).sum()
#         union = ((prob_flat + target_flat) * weights_flat).sum()
#         dice_loss = 1.0 - (2. * intersection + self.delta) / (union + self.delta)
        
#         # Loss balancing
#         if self.use_dynamic_balance:
#             # Dynamic: Trade recall vs. boundary precision
#             recall_error = (target_flat * (1 - (prob_flat > 0.5).float())).sum()
#             boundary_error = ((weights_flat > 1.0) * (prob_flat > 0.5).float() * (1 - target_flat)).sum()
#             balance_factor = torch.sigmoid((recall_error - boundary_error) / self.temp)  # Temperature=1000
#             loss = balance_factor * weighted_bce + (1 - balance_factor) * dice_loss
#         else:
#             # Fixed weights (recommended for severe imbalance)
#             bce_weight, dice_weight = self.fixed_weights
#             loss = bce_weight * weighted_bce + dice_weight * dice_loss
            
#         return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

class ContourWeightedLoss(nn.Module):
    """
    Optimized version of your original loss with key improvements:
    1. Added focal loss component to handle class imbalance
    2. Enhanced boundary detection with vessel thickness awareness
    3. Adaptive boundary smoothing based on image content
    4. Better numerical stability for small vessels
    
    Args:
        boundary_alpha (float): Boundary emphasis (1.0-3.0). Higher = stricter boundaries.
        delta (float): Smoothing factor (1e-6 to 1e-4).
        pos_weight (float): Weight for foreground pixels (>=1). Helps with class imbalance.
        gamma (float): Focal parameter for hard examples (0.0-3.0, 0=no focal effect).
    """
    def __init__(self, boundary_alpha=1.5, delta=1e-3, pos_weight=0.35, gamma=1.0,
                 use_dynamic_balance=True, fixed_weights=(0.6, 0.4)):
        super().__init__()
        self.boundary_alpha = boundary_alpha
        self.delta = delta
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.use_dynamic_balance = use_dynamic_balance
        self.fixed_weights = fixed_weights
        
        # Initialize BCE with class balancing
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([self.pos_weight]))
        
    def get_boundary_weights(self, target):
        """Enhanced boundary weights with vessel thickness awareness."""
        device = target.device
        target_np = target.squeeze(1).cpu().numpy()
        weights = np.ones_like(target_np)
        
        for b in range(target.size(0)):
            # mask = (target_np[b] > 0.5).astype(np.uint8)
            mask = (target_np[b] > 1.2).astype(np.uint8)
            if mask.sum() == 0:  # Skip if no vessels
                continue
                
            # Distance transforms
            fg_dist = distance_transform_edt(mask)
            bg_dist = distance_transform_edt(1 - mask)
            
            # Vessel thickness estimation
            thickness = fg_dist + bg_dist
            thin_structure_mask = (thickness < 5)  # Focus more on thin vessels
            
            # Boundary map with adaptive smoothing
            boundary_map = np.exp(-(fg_dist + bg_dist)**2 / (2 * (self.delta + thickness*0.01)**2))
            
            # Apply boundary emphasis with thickness awareness
            weights[b] = 1.0 + (self.boundary_alpha - 1.0) * boundary_map * thin_structure_mask
            
        return torch.from_numpy(weights).float().to(device)
    
    def forward(self, input, target):
        # Boundary-aware weights
        with torch.no_grad():
            weights = self.get_boundary_weights(target)
        
        # Focal-enhanced BCE loss
        bce_loss = self.bce(input, target)
        if self.gamma > 0:
            pt = torch.exp(-bce_loss)  # prevents nans
            bce_loss = ((1 - pt) ** self.gamma) * bce_loss
        weighted_bce = (weights * bce_loss).mean()
        
        # Enhanced Dice loss with boundary focus
        prob = torch.sigmoid(input)
        prob_flat = prob.view(-1)
        target_flat = target.view(-1)
        weights_flat = weights.view(-1)
        
        intersection = (prob_flat * target_flat * weights_flat).sum()
        union = ((prob_flat + target_flat) * weights_flat).sum()
        dice_loss = 1.0 - (2. * intersection + self.delta) / (union + self.delta)
        
        # Loss balancing
        if self.use_dynamic_balance:
            # Dynamic balancing based on current errors
            recall_error = (target_flat * (1 - (prob_flat > 0.5).float())).sum()
            boundary_error = ((weights_flat > 1.0) * (prob_flat > 0.5).float() * (1 - target_flat)).sum()
            
            # More stable balancing with sigmoid
            balance_factor = torch.sigmoid((recall_error - boundary_error) / 1000)  # Smoother transition
            loss = balance_factor * weighted_bce + (1 - balance_factor) * dice_loss
        else:
            # Fixed weights (adjusted defaults)
            bce_weight, dice_weight = self.fixed_weights
            loss = bce_weight * weighted_bce + dice_weight * dice_loss
            
        return loss