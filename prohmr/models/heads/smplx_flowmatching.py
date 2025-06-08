import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np

class ResidualWrapper(nn.Module):
    """Helper for residual connections where dims match."""
    def forward(self, x):
        return x  # Identity - actual residual is handled in main forward


class SMPLXFlowMatching(nn.Module):
    """Speed-optimized improved flow matching."""
    
    def __init__(self, cfg, context_feats_dim=2048):
        super().__init__()
        self.cfg = cfg
        self.pose_dim = 132  # 22 joints * 6D
        
        # Reasonably improved network (not too deep)
        self.vector_field = FastConditionalMLP(
            input_dim=self.pose_dim,
            context_dim=context_feats_dim,
            hidden_dims=[1024, 1024, 512],  # 3 layers instead of 4
            output_dim=self.pose_dim
        )
        
        # Additional head for betas and camera
        from .fc_head_smplx import FCHeadSMPLX
        self.fc_head = FCHeadSMPLX(cfg, context_feats_dim)
        
    def forward(self, feats: torch.Tensor, num_samples: Optional[int] = None, 
                z: Optional[torch.Tensor] = None) -> Tuple:
        """Sample poses using improved but fast flow matching."""
        batch_size = feats.shape[0]
        device = feats.device
        
        if z is None:
            if num_samples is None:
                num_samples = 1
            z = torch.randn(batch_size, num_samples, self.pose_dim, device=device)
        else:
            num_samples = z.shape[1]
            
        # Use improved Euler (slightly more steps but much faster than RK4)
        samples = self.sample_ode_improved_euler(
            z.reshape(-1, self.pose_dim), 
            feats.repeat_interleave(num_samples, dim=0)
        )
        samples = samples.reshape(batch_size, num_samples, self.pose_dim)
        
        # Fast 6D to rotation conversion
        pred_pose_6d = samples.clone()
        pred_pose = self._6d_to_rotmat_fast(samples)
        
        pred_smpl_params = {
            'global_orient': pred_pose[:, :, [0]],
            'body_pose': pred_pose[:, :, 1:]
        }
        
        # Get betas and camera
        pred_betas, pred_cam = self.fc_head(pred_smpl_params, feats)
        pred_smpl_params['betas'] = pred_betas
        
        # Compute log probability
        log_prob = self.compute_log_prob_fast(samples, feats)
        
        return pred_smpl_params, pred_cam, log_prob, z, pred_pose_6d
    
    def sample_ode_improved_euler(self, z0, context, num_steps=75):
        """
        Improved Euler method - better than basic Euler, much faster than RK4.
        Uses predictor-corrector approach.
        """
        dt = 1.0 / num_steps
        zt = z0
        
        for i in range(num_steps):
            t = i * dt
            t_batch = torch.full((zt.shape[0], 1), t, device=zt.device)
            
            # Predictor step (standard Euler)
            vt = self.vector_field(zt, t_batch, context)
            z_pred = zt + vt * dt
            
            # Corrector step (average of slopes)
            t_next = t_batch + dt
            vt_next = self.vector_field(z_pred, t_next, context)
            zt = zt + 0.5 * dt * (vt + vt_next)  # Average of slopes
            
        return zt
    
    def log_prob(self, smpl_params: Dict, feats: torch.Tensor) -> Tuple:
        """Improved flow matching loss with optimizations."""
        batch_size = feats.shape[0]
        num_samples = smpl_params['global_orient'].shape[1]
        
        # Concatenate pose parameters
        x1 = torch.cat([
            smpl_params['global_orient'].reshape(batch_size, num_samples, -1),
            smpl_params['body_pose'].reshape(batch_size, num_samples, -1)
        ], dim=-1)
        
        # Better time sampling (but keep it simple)
        t = torch.rand(batch_size, num_samples, 1, device=x1.device)
        # Slight bias toward endpoints (where flow matters most)
        t = t ** 0.8  # Simple power law, much faster than sigmoid
        
        # Sample base noise
        z0 = torch.randn_like(x1)
        
        # Linear interpolation (keep it simple)
        xt = t * x1 + (1 - t) * z0
        ut = x1 - z0
        
        # Small noise for stability (reduced)
        if self.training:
            xt = xt + 0.005 * torch.randn_like(xt)  # Less noise, faster
        
        # Predict vector field
        xt_flat = xt.reshape(-1, self.pose_dim)
        t_flat = t.reshape(-1, 1)
        context_flat = feats.repeat_interleave(num_samples, dim=0)
        
        vt = self.vector_field(xt_flat, t_flat, context_flat)
        vt = vt.reshape(batch_size, num_samples, self.pose_dim)
        
        # Flow matching loss with good scaling
        mse_loss = F.mse_loss(vt, ut, reduction='none').mean(dim=-1)
        loss = -mse_loss * 15.0  # Good scaling factor
        
        return loss, xt
    
    def compute_log_prob_fast(self, samples, context):
        """Fast log probability approximation."""
        batch_shape = samples.shape[:-1]
        # Simple Gaussian approximation - very fast
        log_prob = -0.5 * samples.pow(2).mean(dim=-1)
        log_prob = log_prob.reshape(*batch_shape)
        return log_prob
    
    def _6d_to_rotmat_fast(self, pose_6d):
        """Optimized 6D to rotation matrix conversion."""
        batch_size, num_samples, _ = pose_6d.shape
        
        # Reshape efficiently
        pose_6d = pose_6d.reshape(-1, 22, 6)
        
        # Vectorized Gram-Schmidt for all joints at once
        a1 = pose_6d[:, :, :3]  # [B*N, 22, 3]
        a2 = pose_6d[:, :, 3:6]  # [B*N, 22, 3]
        
        # Normalize first vector
        b1 = F.normalize(a1, dim=-1)
        
        # Gram-Schmidt orthogonalization
        dot_product = (b1 * a2).sum(dim=-1, keepdim=True)  # [B*N, 22, 1]
        b2 = a2 - dot_product * b1
        b2 = F.normalize(b2, dim=-1)
        
        # Cross product for third vector
        b3 = torch.cross(b1, b2, dim=-1)
        
        # Stack to rotation matrices
        rot_mats = torch.stack([b1, b2, b3], dim=-1)  # [B*N, 22, 3, 3]
        
        # Reshape back
        rot_mats = rot_mats.reshape(batch_size, num_samples, 22, 3, 3)
        
        return rot_mats


# For the residual connections in the MLP
class FastConditionalMLP(nn.Module):
    """Improved MLP but optimized for speed."""
    def __init__(self, input_dim, context_dim, hidden_dims, output_dim):
        super().__init__()
        # Simple time embedding
        self.time_proj = nn.Linear(1, hidden_dims[0] // 4)
        # Context projection  
        self.context_proj = nn.Linear(context_dim, hidden_dims[0])
        
        # Main layers
        self.input_layer = nn.Linear(input_dim + hidden_dims[0] // 4 + hidden_dims[0], hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.SiLU()
            ))
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x, t, context):
        # Fast projections
        t_emb = F.silu(self.time_proj(t))
        context_emb = self.context_proj(context)
        
        # Initial layer
        h = F.silu(self.input_layer(torch.cat([x, t_emb, context_emb], dim=-1)))
        
        # Hidden layers with residuals where possible
        for i, layer in enumerate(self.hidden_layers):
            h_new = layer(h)
            # Only add residual if dimensions match (avoid reshape overhead)
            if h.shape[-1] == h_new.shape[-1]:
                h = h + h_new
            else:
                h = h_new
                
        return self.output_layer(h)