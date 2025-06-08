import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from yacs.config import CfgNode

from prohmr.utils.geometry import rot6d_to_rotmat
from .fc_head_smplx import FCHeadSMPLX


class MLPDenoiser(ModelMixin, ConfigMixin):
    """Simple MLP denoiser for low-dimensional diffusion"""
    
    @register_to_config
    def __init__(
        self,
        input_dim: int = 132,  # SMPL-X pose parameters
        conditioning_dim: int = 2048,  # ConvNeXt features
        hidden_dims: list = [1024, 512, 512, 1024],
        time_embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.cond_proj = nn.Linear(conditioning_dim, hidden_dims[0])
        self.time_proj = nn.Linear(time_embed_dim, hidden_dims[0])
        
        # Main network
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
        
        self.net = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims[-1], input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            sample: [batch_size, input_dim] - noisy SMPL-X parameters
            timestep: [batch_size] or scalar - diffusion timestep
            conditioning: [batch_size, conditioning_dim] - ConvNeXt features
        """
        batch_size = sample.shape[0]
        device = sample.device
        
        # Ensure timestep is the right shape and device
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        timestep = timestep.to(device)
        
        # Time embedding
        t_emb = self.time_mlp(timestep.float().unsqueeze(-1) / 1000)
        
        # Project inputs
        h = self.input_proj(sample)
        h = h + self.cond_proj(conditioning)
        h = h + self.time_proj(t_emb)
        
        # Main network
        h = self.net(h)
        
        # Output
        output = self.output_proj(h)
        
        if not return_dict:
            return (output,)
        
        return output


class SMPLXDiffusion(nn.Module):
    """SMPL-X parameter prediction using diffusion models"""
    
    def __init__(self, cfg: CfgNode, context_feats_dim=None):
        super(SMPLXDiffusion, self).__init__()
        
        self.cfg = cfg
        self.npose = 6 * (21 + 1)  # 132 for body_pose + global_orient
        
        # Initialize diffusion components
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        
        # For deterministic inference
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )
        
        # Denoiser model
        self.denoiser = MLPDenoiser(
            input_dim=self.npose,
            conditioning_dim=context_feats_dim,
            hidden_dims=[1024, 512, 512, 1024],
            dropout=0.1,
        )
        
        # Use the same FC head for betas and camera
        self.fc_head = FCHeadSMPLX(cfg, context_feats_dim)
    
    def training_forward(
        self, 
        smpl_params: Dict, 
        feats: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass during training - compute diffusion loss"""
        
        batch_size = feats.shape[0]
        device = feats.device
        
        # Prepare ground truth pose in 6D representation
        body_pose = smpl_params['body_pose']
        global_orient = smpl_params['global_orient']
        
        # Convert to 6D if needed (assuming input is rotation matrices)
        if body_pose.dim() == 4:  # [bs, 21, 3, 3]
            body_pose_6d = body_pose[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, -1)
        else:  # Already in some other format
            # You might need to adjust this based on your data format
            body_pose_6d = body_pose.reshape(batch_size, -1)
        
        if global_orient.dim() == 4:  # [bs, 1, 3, 3]
            global_orient_6d = global_orient[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, -1)
        else:
            global_orient_6d = global_orient.reshape(batch_size, -1)
        
        # Concatenate pose parameters
        gt_pose = torch.cat([global_orient_6d, body_pose_6d], dim=-1)  # [bs, 132]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # Add noise to the pose
        noise = torch.randn_like(gt_pose)
        noisy_pose = self.noise_scheduler.add_noise(gt_pose, noise, timesteps)
        
        # Predict the noise
        noise_pred = self.denoiser(
            noisy_pose,
            timesteps,
            feats,
        )
        
        # Compute loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return {"diffusion_loss": loss}
    
    def forward(
        self, 
        feats: torch.Tensor, 
        num_inference_steps: int = 50
    ) -> Tuple[Dict, torch.Tensor]:
        """Inference - generate SMPL-X parameters"""
        
        batch_size = feats.shape[0]
        device = feats.device
        
        # Start from random noise
        pose = torch.randn(batch_size, self.npose, device=device)
        
        # Set timesteps for inference
        self.inference_scheduler.set_timesteps(num_inference_steps)
        
        # Move timesteps to correct device
        timesteps = self.inference_scheduler.timesteps.to(device)
        
        # Denoising loop
        for t in timesteps:
            # Predict noise
            noise_pred = self.denoiser(
                pose,
                t,
                feats,
            )
            
            # Compute previous sample
            pose = self.inference_scheduler.step(
                noise_pred, t, pose
            ).prev_sample
        
        # Split pose into components
        global_orient_6d = pose[:, :6].reshape(batch_size, 1, 6)
        body_pose_6d = pose[:, 6:].reshape(batch_size, 21, 6)
        
        # Keep track of 6D representation
        pred_pose_6d = pose.unsqueeze(1)  # [bs, 1, 132]
        
        # Convert 6D to rotation matrices
        global_orient = rot6d_to_rotmat(global_orient_6d.reshape(-1, 6)).reshape(batch_size, 1, 3, 3)
        body_pose = rot6d_to_rotmat(body_pose_6d.reshape(-1, 6)).reshape(batch_size, 21, 3, 3)
        
        # Prepare output
        pred_smpl_params = {
            'global_orient': global_orient.unsqueeze(1),  # [bs, 1, 1, 3, 3]
            'body_pose': body_pose.unsqueeze(1),  # [bs, 1, 21, 3, 3]
        }
        
        # Get betas and camera from FC head
        pred_betas, pred_cam = self.fc_head(pred_smpl_params, feats)
        pred_smpl_params['betas'] = pred_betas
        
        return pred_smpl_params, pred_cam, pred_pose_6d