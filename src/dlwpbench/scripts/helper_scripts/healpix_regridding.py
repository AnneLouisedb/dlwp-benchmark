import torch
import torch.nn.functional as F

def regrid_healpix_face(x, target_size=8):
    # [B, C, (F), H, W]
    B, C, Fx, H, W = x.shape
    x_reshaped = x.reshape(B*C*Fx, 1, H, W)
    
    # Create normalized 2D grid for the target size
    target_grid = F.affine_grid(
        torch.eye(2, 3).unsqueeze(0).repeat(B*C*Fx, 1, 1).to(x.device),
        size=(B*C*Fx, 1, target_size, target_size),
        align_corners=False
    )
    
    # Perform bilinear interpolation
    x_regridded = F.grid_sample(
        x_reshaped, 
        target_grid, 
        mode='bilinear', 
        align_corners=False
    )
    
    # Reshape back to original dimensions
    return x_regridded.reshape(B, C, Fx, target_size, target_size)
