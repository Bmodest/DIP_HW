import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        ### FILL:
        ### J_proj = ...
        # 检查 means3D[:, 2] 是否包含零值
        if torch.any(torch.abs(cam_points[:, 2])<1e-5):
            print("Warning: Found zero depth values in the point cloud.")
            print(torch.min(torch.abs(cam_points[:, 2])))

        # 缓存中间结果
        inv_z = 1 / cam_points[:, 2]
        z_squared = cam_points[:, 2] ** 2

        J_proj[:, 0, 0] = inv_z*K[0, 0]
        J_proj[:, 0, 1] = 0
        J_proj[:, 0, 2] = -cam_points[:, 0] / z_squared*K[0, 0]
        J_proj[:, 1, 0] = 0
        J_proj[:, 1, 1] = inv_z*K[1, 1]
        J_proj[:, 1, 2] = -cam_points[:, 1] / z_squared*K[1, 1]
        
        if torch.any(torch.isnan(J_proj)):
            print("Warning: Found NaN values in the Jacobian matrix.")
        # Transform covariance to camera space
        ### FILL: Aplly world to camera rotation to the 3d covariance matrix
        ### covs_cam = ...  # (N, 3, 3)
        covs_cam = R @ covs3d @ R.T
        
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        if torch.any(torch.det(covs2D).unsqueeze(-1).unsqueeze(-1)<1e-10):
            print(torch.min(torch.det(covs3d).unsqueeze(-1).unsqueeze(-1)))
            print(torch.min(torch.det(R).unsqueeze(-1).unsqueeze(-1)))
            print(torch.min(torch.det(covs_cam).unsqueeze(-1).unsqueeze(-1)))
            print(torch.min(torch.det(covs_cam + 1e-4 * torch.eye(3, device=covs2D.device).unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)))
            
            print(torch.min(torch.det(covs2D).unsqueeze(-1).unsqueeze(-1)))
            print(torch.min(cam_points[:, 2]))
            print(torch.max(cam_points[:, 2]))
            print("Warning: Found abs(det)<1e-10 in the determinant.")
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute determinant for normalization
        ### FILL: compute the gaussian values
        ### gaussian = ... ## (N, H, W)
        # det = torch.det(covs2D).unsqueeze(-1).unsqueeze(-1)

        # eye_matrix = torch.eye(2, device=covs2D.device).unsqueeze(0)
        # inv_covs2D = torch.linalg.solve(covs2D, eye_matrix)
        # result =(dx.unsqueeze(3)@inv_covs2D.reshape(N, 1, 1, 2, 2)@dx.unsqueeze(4))
        # result = result.squeeze(-1).squeeze(-1)
        # print(det.shape)
        # print(result.shape)
        # gaussian= torch.exp(-result/2)/2/torch.pi/torch.sqrt(det)

        # 计算 Cholesky 分解
        try:
            L = torch.linalg.cholesky(covs2D)
        except RuntimeError as e:
            raise ValueError("Cholesky decomposition failed. Check the input covariance matrices.") from e
        # 计算行列式 (N, 1, 1)
        det = torch.det(covs2D).unsqueeze(-1).unsqueeze(-1)
        # 检查行列式是否为零
        if torch.any(det <= 0):
            raise ValueError("Detected non-positive definite covariance matrix.")
        # 检查行列式是否为零
        if torch.any(det <= 0):
            raise ValueError("Detected non-positive definite covariance matrix.")

        # 计算逆矩阵
        inv_covs2D = torch.cholesky_inverse(L)

        # 计算高斯值 (N, H, W)
        result = (dx.unsqueeze(3) @ inv_covs2D.unsqueeze(1).unsqueeze(1) @ dx.unsqueeze(4))
        result = result.squeeze(-1).squeeze(-1)
        gaussian = torch.exp(-0.5 * result) / (2 * torch.pi * torch.sqrt(det))
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights
        ### FILL:
        ### weights = ... # (N, H, W)
        alphas=torch.cat([torch.zeros_like(alphas[:1, ...]), alphas], dim=0)
        T=torch.cumprod(1-alphas[:-1, ...],dim=0)
        weights = T*alphas[1:, ...]
        if torch.any(torch.isnan(weights)):
            print("Warning: Found NaN values in the weights.")
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered