import torch
import torch.nn as nn

from typing import Any, Dict, Tuple

from yacs.config import CfgNode

import smplx

from .backbones.convnext_depth import ConvNeXtDepthBackbone
from prohmr.utils.geometry import aa_to_rotmat
from prohmr.utils.konia_transform import rotation_matrix_to_angle_axis

from .heads import SMPLXDiffusion
from .discriminatorSmplx import DiscriminatorSmplx
from .losses import Keypoint3DLoss
from ..utils.renderer import *



class ProHMRDepthEgobodyDiffusion(nn.Module):

    def __init__(self, cfg: CfgNode, device=None, writer=None, logger=None, with_global_3d_loss=False):
        """
        Setup ProHMR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super(ProHMRDepthEgobodyDiffusion, self).__init__()

        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.logger = logger

        self.with_global_3d_loss = with_global_3d_loss

        self.backbone = ConvNeXtDepthBackbone(
            model_name='convnext_small.fb_in22k_ft_in1k',
            out_features=2048,
            pretrained=True
        ).to(self.device)

        # Create Diffusion head
        context_feats_dim = cfg.MODEL.FLOW.CONTEXT_FEATURES
        self.diffusion = SMPLXDiffusion(cfg, context_feats_dim=context_feats_dim).to(self.device)

        # Create discriminator
        self.discriminator = DiscriminatorSmplx().to(self.device)

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.v2v_loss = nn.L1Loss(reduction='none')
        self.smpl_parameter_loss = nn.MSELoss(reduction='none')
      
        self.smplx = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', ext='npz').to(self.device)

        self.smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', ext='npz').to(self.device)
        self.smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', ext='npz').to(self.device)

        # Cache for default parameters (will be created on first use)
        self._default_params_cache = {}


    def init_optimizers(self):
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        self.optimizer = torch.optim.AdamW(params=list(self.backbone.parameters()) + list(self.diffusion.parameters()),
                                     lr=self.cfg.TRAIN.LR,
                                     weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        self.optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                           lr=self.cfg.TRAIN.LR,
                                           weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        # return optimizer, optimizer_disc


    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        x = batch['img'].unsqueeze(1)
        batch_size = x.shape[0]
        
        # Compute conditioning features
        conditioning_feats = self.backbone(x)
        
        if train:
            # During training, we need GT parameters for diffusion loss
            gt_smpl_params = batch['smpl_params']
            is_axis_angle = batch['smpl_params_is_axis_angle']
            
            # Prepare GT in proper format
            smpl_params_for_diffusion = {}
            
            # Handle body pose
            if is_axis_angle['body_pose'].all():
                body_pose_mat = aa_to_rotmat(gt_smpl_params['body_pose'].reshape(-1, 3)).reshape(batch_size, 21, 3, 3)
            else:
                body_pose_mat = gt_smpl_params['body_pose']
            smpl_params_for_diffusion['body_pose'] = body_pose_mat
            
            # Handle global orient
            if is_axis_angle['global_orient'].all():
                global_orient_mat = aa_to_rotmat(gt_smpl_params['global_orient'].reshape(-1, 3)).reshape(batch_size, 1, 3, 3)
            else:
                global_orient_mat = gt_smpl_params['global_orient']
            smpl_params_for_diffusion['global_orient'] = global_orient_mat
            
            # Compute diffusion loss
            diffusion_output = self.diffusion.training_forward(smpl_params_for_diffusion, conditioning_feats)
            diffusion_loss = diffusion_output['diffusion_loss']
        else:
            diffusion_loss = None

        # Generate predictions
        num_inference_steps = getattr(self.cfg.MODEL, 'DIFFUSION_STEPS', 50)
        pred_smpl_params, pred_cam, pred_pose_6d = self.diffusion(
            conditioning_feats, 
            num_inference_steps=num_inference_steps
        )
        
        # Store outputs
        output = {}
        output['pred_cam'] = pred_cam  # [bs, 1, 3]
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
        output['conditioning_feats'] = conditioning_feats
        output['pred_pose_6d'] = pred_pose_6d
        if train:
            output['diffusion_loss'] = diffusion_loss
        
        # Compute SMPL-X outputs (simplified without num_samples)
        pred_smpl_params['global_orient'] = rotation_matrix_to_angle_axis(
            pred_smpl_params['global_orient'].reshape(-1, 3, 3)
        ).reshape(batch_size, -1, 3)
        pred_smpl_params['body_pose'] = rotation_matrix_to_angle_axis(
            pred_smpl_params['body_pose'].reshape(-1, 3, 3)
        ).reshape(batch_size, -1, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        
        pred_smpl_params = self._fill_missing_smplx_params(pred_smpl_params, batch_size)
        
        smplx_output = self.smplx(**{k: v.float() for k,v in pred_smpl_params.items()})
        
        output['pred_keypoints_3d'] = smplx_output.joints.unsqueeze(1)  # [bs, 1, 127, 3]
        output['pred_vertices'] = smplx_output.vertices.unsqueeze(1)    # [bs, 1, 10475, 3]
        output['pred_keypoints_3d_global'] = output['pred_keypoints_3d'] + output['pred_cam'].unsqueeze(-2)
        
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_smpl_params = output['pred_smpl_params']
        pred_pose_6d = output['pred_pose_6d']
        pred_keypoints_3d_global = output['pred_keypoints_3d_global'][:, 0, 0:22]  # Remove sample dim
        pred_keypoints_3d = output['pred_keypoints_3d'][:, 0, 0:22]
        pred_vertices = output['pred_vertices'][:, 0]  # [bs, 10475, 3]
        
        batch_size = batch['img'].shape[0]
        
        # Ground truth
        gt_keypoints_3d_global = batch['keypoints_3d'][:, 0:22]
        gt_smpl_params = batch['smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']
        gt_gender = batch['gender']
        
        # 3D keypoint losses (no need for repeat since single prediction)
        loss_keypoints_3d_mode = self.keypoint_3d_loss(
            pred_keypoints_3d_global.unsqueeze(1), 
            gt_keypoints_3d_global.unsqueeze(1), 
            pelvis_id=0, 
            pelvis_align=True
        ).mean()
        
        loss_keypoints_3d_full_mode = self.keypoint_3d_loss(
            pred_keypoints_3d_global.unsqueeze(1), 
            gt_keypoints_3d_global.unsqueeze(1), 
            pelvis_align=False
        ).mean()
        
        # Vertex loss
        gt_smpl_params = self._fill_missing_smplx_params(gt_smpl_params, batch_size)
        gt_smpl_output = self.smplx_male(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices = gt_smpl_output.vertices
        gt_joints = gt_smpl_output.joints
        
        # Handle gender
        gt_smpl_output_female = self.smplx_female(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices[gt_gender == 1] = gt_smpl_output_female.vertices[gt_gender == 1]
        gt_joints[gt_gender == 1] = gt_smpl_output_female.joints[gt_gender == 1]
        
        gt_pelvis = gt_joints[:, [0], :].clone()
        loss_v2v_mode = self.v2v_loss(
            pred_vertices - pred_keypoints_3d[:, [0], :].clone(), 
            gt_vertices - gt_pelvis
        ).mean()
        
        # SMPL parameter losses
        loss_smpl_params_mode = {}
        for k, pred in pred_smpl_params.items():
            if k != 'transl':
                gt = gt_smpl_params[k]
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                loss_smpl_params_mode[k] = self.smpl_parameter_loss(pred, gt).mean()
        
        # Orthonormal loss
        pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
        loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - 
                        torch.eye(2, device=pred_pose_6d.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2)
        loss_pose_6d_mode = loss_pose_6d.mean()

        
        loss = (self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d_mode +
                self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode +
                self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_FULL_MODE'] * loss_keypoints_3d_full_mode * self.with_global_3d_loss +
                self.cfg.LOSS_WEIGHTS['V2V_MODE'] * loss_v2v_mode +
                sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k+'_MODE').upper()] 
                    for k in loss_smpl_params_mode]))
        
        # Add diffusion loss if training
        if train and 'diffusion_loss' in output:
            diffusion_weight = self.cfg.LOSS_WEIGHTS.get('DIFFUSION', 0.5)
            loss = loss + diffusion_weight * output['diffusion_loss']
        
        losses = dict(
            loss=loss.detach(),
            loss_pose_6d_mode=loss_pose_6d_mode.detach(),
            loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach(),
            loss_keypoints_3d_full_mode=loss_keypoints_3d_full_mode.detach(),
            loss_v2v_mode=loss_v2v_mode.detach(),
        )

        if train and 'diffusion_loss' in output:
            losses['loss_diffusion'] = output['diffusion_loss'].detach()
        
        for k, v in loss_smpl_params_mode.items():
            losses['loss_' + k + '_mode'] = v.detach()
        
        output['losses'] = losses
        return loss



    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        # batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']  # [bs, 69]
        gt_betas = batch['betas']
        batch_size = gt_body_pose.shape[0]
        # n_sample = body_pose.shape[0] // batch_size

        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)  # [bs, 23, 3, 3]
        if gt_rotmat.shape[1] > 21:  # If GT has more than 21 joints (SMPL format)
            gt_rotmat = gt_rotmat[:, :21, :, :]  # Take only first 21 joints for SMPL-X discriminator
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())  # [bs*n_samples, 25]
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / disc_fake_out.shape[0]
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)  # [bs, 25]
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / disc_real_out.shape[0]
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        # self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, batch: Dict, mocap_batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        ### read input data
        # batch = joint_batch['img']   # [64, 3, 224, 224]
        # mocap_batch = joint_batch['mocap']
        # optimizer, optimizer_disc = self.optimizers(use_pl_optimizer=True)
        batch_size = batch['img'].shape[0]

        self.backbone.train()
        self.diffusion.train()
        # self.backbone.eval()
        # self.flow.eval()
        ### G forward step
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        num_samples = pred_smpl_params['body_pose'].shape[1]
        ### compute G loss
        loss = self.compute_loss(batch, output, train=True)
        
        # # ### G adv loss
        disc_out = self.discriminator(
            pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1, 3, 3),  
            pred_smpl_params['betas'].reshape(batch_size * num_samples, -1)
        )
        loss_adv = ((disc_out - 1.0) ** 2).sum() / (batch_size * num_samples)
        # #
        # # ### G backward
        
        loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
        self.optimizer.zero_grad()
        # self.manual_backward(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) 
        self.optimizer.step()

        # # import pdb; pdb.set_trace()
        # ### D forward, backward
        loss_disc = self.training_step_discriminator(
            mocap_batch, 
            pred_smpl_params['body_pose'].reshape(batch_size * num_samples, -1, 3, 3),
            pred_smpl_params['betas'].reshape(batch_size * num_samples, -1), 
            self.optimizer_disc
        )
        #
        output['losses']['loss_gen'] = loss_adv
        output['losses']['loss_disc'] = loss_disc

        # if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
        #     self.tensorboard_logging(batch, output, self.global_step, train=True)

        return output

    def validation_step(self, batch: Dict) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """

        self.backbone.eval()
        self.diffusion.eval()
        

        output = self.forward_step(batch, train=False)

        loss = self.compute_loss(batch, output, train=False)
        return output

    def _fill_missing_smplx_params(self, params_dict, batch_size):
        """
        Add missing SMPL-X parameters using cached defaults.
        """
        # Get device from existing parameters
        device = next(iter(params_dict.values())).device
        
        # Get default parameters for this batch size and device
        defaults = self._get_default_smplx_params(batch_size, device)
        
        # Only add parameters that don't exist
        for param_name, default_value in defaults.items():
            if param_name not in params_dict:
                params_dict[param_name] = default_value
        
        return params_dict

    def _get_default_smplx_params(self, batch_size, device):
        """
        Get default SMPL-X parameters, creating and caching them on first use.
        This ensures they're always on the correct device.
        """
        cache_key = f"{batch_size}_{device}"
        
        if cache_key not in self._default_params_cache:
            # Create default parameters on the correct device
            self._default_params_cache[cache_key] = {
                'jaw_pose': torch.zeros(batch_size, 3, device=device),
                'leye_pose': torch.zeros(batch_size, 3, device=device),
                'reye_pose': torch.zeros(batch_size, 3, device=device),
                'left_hand_pose': torch.zeros(batch_size, 6, device=device),
                'right_hand_pose': torch.zeros(batch_size, 6, device=device),
                'expression': torch.zeros(batch_size, 10, device=device)
            }
        
        return self._default_params_cache[cache_key]