"""
Script used for evaluating the 3D pose errors of ProHMR (mode + minimum).

Example usage:
python eval_regression.py --checkpoint=/path/to/checkpoint --dataset=3DPW-TEST

Running the above will compute the Reconstruction Error for the mode as well as the minimum error for the test set of 3DPW.
"""
import os

import torch
import argparse
from tqdm import tqdm
from prohmr.configs import get_config, prohmr_config, dataset_config
# from prohmr.models import ProHMR
from prohmr.utils import Evaluator, recursive_to
# from prohmr.datasets import create_dataset
import smplx
import numpy as np
import PIL.Image as pil_img
# import open3d as o3d
import json
import copy
import pickle as pkl
import random
import PIL.Image as pil_img
import pyrender

from prohmr.models import ProHMRDepthEgobody
# from prohmr.utils.other_utils import coord_transf, coord_multiple_transf, coord_transf_holo_yz_reverse
from prohmr.utils.pose_utils import reconstruction_error
from prohmr.utils.renderer import *
from prohmr.utils.konia_transform import rotation_matrix_to_angle_axis

from prohmr.datasets.image_dataset_depth_egobody import ImageDatasetDepthEgoBody

import matplotlib.pyplot as plt
import trimesh

# def visualize_first_sample_simple(step, batch, pred_vertices_global_mode, gt_vertices, 
#                                   smplx_neutral, r, material, camera, light, camera_pose, 
#                                   img_names, args):
#     """
#     Simple visualization: show original image, GT mesh, predicted mesh, and error heatmap separately.
    
#     Args:
#         step: Current batch step
#         batch: Input batch data containing original images
#         pred_vertices_global_mode: Predicted vertices in global coordinates [bs, 10475, 3]
#         gt_vertices: Ground truth vertices [bs, 10475, 3]
#         smplx_neutral: SMPL-X model
#         r: pyrender OffscreenRenderer
#         material: pyrender material
#         camera: pyrender camera
#         light: pyrender light
#         camera_pose: camera pose matrix
#         img_names: list of image names
#         args: command line arguments
#     """
        
#     import os
#     import matplotlib.pyplot as plt
    
#     # Create output directory
#     vis_dir = f"./eval_result/visualizations_{os.path.basename(args.checkpoint).replace('.pt', '')}"
#     os.makedirs(vis_dir, exist_ok=True)
    
#     # Get first sample data
#     sample_idx = 0
    
#     # Extract vertices for first sample
#     pred_verts = pred_vertices_global_mode[sample_idx].detach().cpu().numpy()
#     gt_verts = gt_vertices[sample_idx].detach().cpu().numpy()
#     faces = smplx_neutral.faces
    
#     # Load original image
#     input_img = batch['img'][sample_idx].detach().cpu().numpy()
#     if input_img.shape[0] == 3:  # RGB format (C, H, W) -> (H, W, C)
#         input_img = np.transpose(input_img, (1, 2, 0))
#     elif input_img.shape[0] == 1:  # Grayscale
#         input_img = input_img.squeeze(0)
    
#     # Normalize to 0-255 if needed
#     if input_img.max() <= 1.0:
#         input_img = (input_img * 255).astype(np.uint8)
    
#     # Convert grayscale to RGB for display
#     if len(input_img.shape) == 2:
#         input_img = np.stack([input_img] * 3, axis=-1)
    
#     # Create materials
#     gt_material = pyrender.MetallicRoughnessMaterial(
#         metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(0.0, 1.0, 0.0, 1.0))  # Green
#     pred_material = pyrender.MetallicRoughnessMaterial(
#         metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 0.0, 0.0, 1.0))  # Red
    
#     # 1. Render Ground Truth mesh
#     scene_gt = pyrender.Scene()
#     gt_mesh = trimesh.Trimesh(vertices=gt_verts, faces=faces)
#     gt_pyrender_mesh = pyrender.Mesh.from_trimesh(gt_mesh, material=gt_material)
#     scene_gt.add(gt_pyrender_mesh)
#     scene_gt.add(camera, pose=camera_pose)
#     scene_gt.add(light, pose=camera_pose)
    
#     gt_render, _ = r.render(scene_gt)
    
#     # 2. Render Predicted mesh
#     scene_pred = pyrender.Scene()
#     pred_mesh = trimesh.Trimesh(vertices=pred_verts, faces=faces)
#     pred_pyrender_mesh = pyrender.Mesh.from_trimesh(pred_mesh, material=pred_material)
#     scene_pred.add(pred_pyrender_mesh)
#     scene_pred.add(camera, pose=camera_pose)
#     scene_pred.add(light, pose=camera_pose)
    
#     pred_render, _ = r.render(scene_pred)
    
#     # 3. Create error visualization
#     vertex_errors = np.linalg.norm(pred_verts - gt_verts, axis=1)
#     max_error = np.percentile(vertex_errors, 95)  # Use 95th percentile to avoid outliers
#     normalized_errors = np.clip(vertex_errors / max_error, 0, 1)
    
#     # Create error colormap (blue = low error, red = high error)
#     error_colors = plt.cm.viridis(normalized_errors)
    
#     # Create error mesh with vertex colors
#     error_mesh = trimesh.Trimesh(vertices=pred_verts, faces=faces, vertex_colors=error_colors)
    
#     # Render error mesh
#     scene_error = pyrender.Scene()
#     error_pyrender_mesh = pyrender.Mesh.from_trimesh(error_mesh, smooth=True)
#     scene_error.add(error_pyrender_mesh)
#     scene_error.add(camera, pose=camera_pose)
#     scene_error.add(light, pose=camera_pose)
    
#     error_render, _ = r.render(scene_error)
    
#     # Create final visualization with all four images
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
#     # Original image
#     axes[0].imshow(input_img)
#     axes[0].set_title('Original Image')
#     axes[0].axis('off')
    
#     # Ground truth mesh
#     axes[1].imshow(gt_render)
#     axes[1].set_title('Ground Truth\n(Green Mesh)')
#     axes[1].axis('off')
    
#     # Prediction mesh
#     axes[2].imshow(pred_render)
#     axes[2].set_title('Prediction\n(Red Mesh)')
#     axes[2].axis('off')
    
#     # Error visualization
#     axes[3].imshow(error_render)
#     axes[3].set_title(f'Error Heatmap\nMean: {vertex_errors.mean()*1000:.1f}mm')
#     axes[3].axis('off')
    
#     # Add colorbar for error visualization
#     sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max_error*1000))
#     cbar = plt.colorbar(sm, ax=axes[3], fraction=0.046, pad=0.04)
#     cbar.set_label('Error (mm)')
    
#     plt.tight_layout()
    
#     # Save individual images
#     clean_name = img_names[sample_idx].replace('/', '_').replace('\\', '_')
    
#     pil_img.fromarray(input_img).save(os.path.join(vis_dir, f"{clean_name}_original.png"))
#     pil_img.fromarray(gt_render).save(os.path.join(vis_dir, f"{clean_name}_gt_mesh.png"))
#     pil_img.fromarray(pred_render).save(os.path.join(vis_dir, f"{clean_name}_pred_mesh.png"))
#     pil_img.fromarray(error_render).save(os.path.join(vis_dir, f"{clean_name}_error_heatmap.png"))
    
#     # Save combined figure
#     plt.savefig(os.path.join(vis_dir, f"{clean_name}_comparison.png"), 
#                 dpi=150, bbox_inches='tight')
#     plt.close()
    
#     # Print statistics
#     print(f"\n=== Visualization saved for sample {sample_idx} ===")
#     print(f"Image name: {img_names[sample_idx]}")
#     print(f"Mean vertex error: {vertex_errors.mean() * 1000:.2f} mm")
#     print(f"Max vertex error: {vertex_errors.max() * 1000:.2f} mm")
#     print(f"Saved to: {vis_dir}")

cmap= plt.get_cmap('turbo')  # viridis
color_map = cmap.colors
color_map = np.asarray(color_map)

# python eval_regression_depth_egobody.py --checkpoint ./data/checkpoint/depth/best_model.pt --dataset_root /vlg-nfs/scratch/xialyu/EgoGen/EgoGen/experiments/hmregogen/data/egobody_release



parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--dataset_root', type=str, default='/vlg-nfs/szhang/egobody_release')
parser.add_argument('--checkpoint', type=str, default='try_egogen_new_data/92990/best_model.pt')  # runs_try/90505/best_model.pt data/checkpoint.pt
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')
parser.add_argument('--num_samples', type=int, default=2, help='Number of test samples to draw')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=100, help='How often to log results')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument('--shuffle', default='False', type=lambda x: x.lower() in ['true', '1'])  # todo


args = parser.parse_args()

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
fixseed(args.seed)

# Load model config
if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)



# Update number of test samples drawn to the desired value
model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
model_cfg.freeze()


# Setup model
model = ProHMRDepthEgobody(cfg=model_cfg, device=device)
weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
# model.load_state_dict(weights['state_dict'])
weights_copy = {}
weights_copy['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] != 'smplx' and k.split('.')[0] != 'smplx_male' and k.split('.')[0] != 'smplx_female'}
model.load_state_dict(weights_copy['state_dict'], strict=False)
model.eval()
print(args.checkpoint)



test_dataset = ImageDatasetDepthEgoBody(cfg=model_cfg, train=False, device=device, img_dir=args.dataset_root,
                                       dataset_file=os.path.join(args.dataset_root, 'smplx_spin_holo_depth_npz/egocapture_test_smplx_split_known.npz'),
                                    #    dataset_file = "./data/smplx_spin_npz/egocapture_test_smplx_depth_top5.npz",
                                       spacing=1, split='test')
dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)


smplx_neutral = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', batch_size=args.batch_size * args.num_samples, ext='npz').to(device)
smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', batch_size=args.batch_size, ext='npz').to(device)
smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', batch_size=args.batch_size, ext='npz').to(device)


# body_conversion = np.load('data/smplx_to_smpl.npz')
# smplx_mainbody_vert_id = body_conversion['SMPL_X_ids']

# eval_sample_n = len(dataset) // args.eval_freq
g_mpjpe = np.zeros(len(test_dataset))
mpjpe = np.zeros(len(test_dataset))
pa_mpjpe = np.zeros(len(test_dataset))
g_v2v = np.zeros(len(test_dataset))
v2v = np.zeros(len(test_dataset))  # translation/pelv-aligned
pa_v2v = np.zeros(len(test_dataset))  # procrustes aligned

######## todo uncomment if want to render, not compatible with open3d visualization
#common
# H, W = 350, 388
# # camera_center = np.array([160, 144])
# camera_pose = np.eye(4)
# camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
# camera = pyrender.camera.IntrinsicsCamera(
#     fx=200, fy=200,
#     cx=int(W/2), cy=int(H/2))
# light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
# base_color = (1.0, 193/255, 193/255, 1.0)
# material = pyrender.MetallicRoughnessMaterial(
#     metallicFactor=0.0,
#     alphaMode='OPAQUE',
#     baseColorFactor=base_color
#     )
# r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H, point_size=1.0)

img_name_list = []

# Go over the images in the dataset.
for step, batch in enumerate(tqdm(dataloader)):
    if 1:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        gt_pose = {}
        gt_pose['global_orient'] = batch['smpl_params']['global_orient'].to(device)
        gt_pose['transl'] = batch['smpl_params']['transl'].to(device)
        gt_pose['body_pose'] = batch['smpl_params']['body_pose'].to(device)
        gt_pose['betas'] = batch['smpl_params']['betas'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = batch['img'].shape[0]
        img_names = batch['imgname']
        img_name_list.extend(img_names)

        pred_betas = out['pred_smpl_params']['betas']  #  [bs, num_sample, 10]
        pred_body_pose = out['pred_smpl_params']['body_pose']  # [bs, num_sample, 23, 3, 3]
        pred_global_orient = out['pred_smpl_params']['global_orient']  # [bs, num_sample, 1, 3, 3]
        pred_body_pose = rotation_matrix_to_angle_axis(pred_body_pose.reshape(-1, 3, 3)).reshape(curr_batch_size, args.num_samples, -1, 3)
        pred_global_orient = rotation_matrix_to_angle_axis(pred_global_orient.reshape(-1, 3, 3)).reshape(curr_batch_size, args.num_samples, -1, 3)
        pred_transl = out['pred_cam']  #  [bs, num_sample, 3]


        ##############
        if curr_batch_size != args.batch_size:
            smplx_neutral = smplx.create('data/smplx_model', model_type='smplx', gender='neutral', batch_size=curr_batch_size * args.num_samples, ext='npz').to(device)
        pred_output = smplx_neutral(betas=pred_betas.reshape(-1, 10), body_pose=pred_body_pose.reshape(-1, 21, 3),
                                    global_orient=pred_global_orient.reshape(-1, 1, 3))
        pred_vertices = pred_output.vertices.reshape(curr_batch_size, -1, 10475, 3)
        pred_keypoints_3d = pred_output.joints.reshape(curr_batch_size, -1, 127, 3)[:, :, :22, :]  # [bs, n_sample, 22, 3]
        pred_vertices_mode = pred_vertices[:, 0]
        pred_keypoints_3d_mode = pred_keypoints_3d[:, 0]  # [bs, 22, 3]
        pred_keypoints_3d_global_mode = pred_keypoints_3d_mode + pred_transl[:, 0].unsqueeze(-2)
        pred_vertices_global_mode = pred_vertices_mode + pred_transl[:, 0].unsqueeze(-2)

        ###### single mode with z_0
        pred_pelvis_mode = pred_keypoints_3d_mode[:, [0], :].clone()
        pred_keypoints_3d_mode_align = pred_keypoints_3d_mode - pred_pelvis_mode
        pred_vertices_mode_align = pred_vertices_mode - pred_pelvis_mode
        pred_transl_mode = pred_transl[:, 0]


        ##### get gt body
        if curr_batch_size != args.batch_size:
            smplx_male = smplx.create('data/smplx_model', model_type='smplx', gender='male', batch_size=curr_batch_size, ext='npz').to(device)
            smplx_female = smplx.create('data/smplx_model', model_type='smplx', gender='female', batch_size=curr_batch_size, ext='npz').to(device)

        gt_body = smplx_male(**gt_pose)
        gt_joints = gt_body.joints
        gt_vertices = gt_body.vertices
        gt_body_female = smplx_female(**gt_pose)
        gt_joints_female = gt_body_female.joints
        gt_vertices_female = gt_body_female.vertices
        gt_joints[gender == 1, :, :] = gt_joints_female[gender == 1, :, :]
        gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]

        # if step >= 6 and step < 12:
        #     visualize_first_sample_simple(
        #         step, batch, pred_vertices_global_mode, gt_vertices,
        #         smplx_neutral, r, material, camera, light, camera_pose, 
        #         img_names, args
        #     )

        gt_keypoints_3d = gt_joints[:, :22, :]  # [bs, 22, 3]
        gt_pelvis = gt_keypoints_3d[:, [0], :].clone()  # [bs,1,3]
        gt_keypoints_3d_align = gt_keypoints_3d - gt_pelvis
        gt_vertices_align = gt_vertices - gt_pelvis

        ########################## only for vis
        pred_output_cano = smplx_neutral(betas=pred_betas.reshape(-1, 10), body_pose=pred_body_pose.reshape(-1, 21, 3),
                                         global_orient=torch.zeros([curr_batch_size*args.num_samples, 1, 3]).to(device))
        pred_vertices_cano = pred_output_cano.vertices.reshape(curr_batch_size, -1, 10475, 3)
        pred_keypoints_3d_cano = pred_output_cano.joints.reshape(curr_batch_size, -1, 127, 3)[:, :, :22, :]  # [bs, n_sample, 22, 3]
        pred_vertices_mode_cano = pred_vertices_cano[:, 0]
        pred_keypoints_3d_mode_cano = pred_keypoints_3d[:, 0]  # [bs, 22, 3]
        pred_keypoints_3d_global_mode_cano = pred_keypoints_3d_mode + pred_transl[:, 0].unsqueeze(-2)
        pred_vertices_global_mode_cano = pred_vertices_mode_cano + pred_transl[:, 0].unsqueeze(-2)


        gt_pose['global_orient'] = torch.zeros([curr_batch_size, 1, 3]).to(device)
        gt_body_cano = smplx_male(**gt_pose)
        gt_joints_cano = gt_body_cano.joints
        gt_vertices_cano = gt_body_cano.vertices
        gt_body_female_cano = smplx_female(**gt_pose)
        gt_joints_female_cano = gt_body_female_cano.joints
        gt_vertices_female_cano = gt_body_female_cano.vertices
        gt_joints_cano[gender == 1, :, :] = gt_joints_female_cano[gender == 1, :, :]
        gt_vertices_cano[gender == 1, :, :] = gt_vertices_female_cano[gender == 1, :, :]

        gt_keypoints_3d_cano = gt_joints_cano[:, :22, :]  # [bs, 22, 3]
        gt_pelvis_cano = gt_keypoints_3d_cano[:, [0], :].clone()  # [bs,1,3]
        gt_keypoints_3d_align_cano = gt_keypoints_3d_cano - gt_pelvis_cano
        gt_vertices_align_cano = gt_vertices_cano - gt_pelvis_cano

        # ###############################
        #
        #
        # G-MPJPE
        error_per_joint = torch.sqrt(((pred_keypoints_3d_global_mode - gt_keypoints_3d) ** 2).sum(dim=-1))
        error = error_per_joint.mean(dim=-1).detach().cpu().numpy()
        g_mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # MPJPE
        error_per_joint = torch.sqrt(((pred_keypoints_3d_mode_align - gt_keypoints_3d_align) ** 2).sum(dim=-1))
        error = error_per_joint.mean(dim=-1).detach().cpu().numpy()
        mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # PA-MPJPE
        pa_error_per_joint = reconstruction_error(pred_keypoints_3d_mode_align.cpu().detach().numpy(), gt_keypoints_3d_align.detach().cpu().numpy(), avg_joint=False)  # [bs, n_joints]
        pa_error = pa_error_per_joint.mean(axis=-1)  # [bs]
        pa_mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = pa_error

        # G-V2V
        error_per_verts = torch.sqrt(((pred_vertices_global_mode - gt_vertices) ** 2).sum(dim=-1))
        error = error_per_verts.mean(dim=-1).detach().cpu().numpy()
        g_v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # V2V
        error_per_verts = torch.sqrt(((pred_vertices_mode_align - gt_vertices_align) ** 2).sum(dim=-1))
        error = error_per_verts.mean(dim=-1).detach().cpu().numpy()
        v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # PA-V2V
        pa_error = reconstruction_error(pred_vertices_mode_align.detach().cpu().numpy(), gt_vertices_align.detach().cpu().numpy(), avg_joint=True)
        pa_v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = pa_error

        if step % args.log_freq == 0 and step > 0:
            print('G-MPJPE: ' + str(1000 * g_mpjpe[:step * args.batch_size].mean()))
            print('MPJPE: ' + str(1000 * mpjpe[:step * args.batch_size].mean()))
            print('PA-MPJPE: ' + str(1000 * pa_mpjpe[:step * args.batch_size].mean()))


print('*** Final Results ***')
print('G-MPJPE: ' + str(1000 * g_mpjpe.mean()))
print('MPJPE: ' + str(1000 * mpjpe.mean()))
print('PA-MPJPE: ' + str(1000 * pa_mpjpe.mean()))

print('G-V2V: ' + str(1000 * g_v2v.mean()))
print('V2V: ' + str(1000 * v2v.mean()))
print('PA-V2V: ' + str(1000 * pa_v2v.mean()))
#

if 1:
    name = '_'.join(args.checkpoint.split("/")[-2:])
    save_path = "./eval_result/%s_all_loss.json"%name
    print("save_path: ", save_path)
    with open(save_path, 'w') as f:
        tmp = {'v2v': v2v.tolist(), 'img_name': img_name_list}
        json.dump(tmp, f, indent=4, sort_keys=True)

if 1:
    #save the eval result
    result = {}
    result['g_mpjpe'] = str(1000 * g_mpjpe.mean())
    result['mpjpe'] = str(1000 * mpjpe.mean())
    result['pa_mpjpe'] = str(1000 * pa_mpjpe.mean())

    result['g_v2v'] = str(1000 * g_v2v.mean())
    result['v2v'] = str(1000 * v2v.mean())
    result['pa_v2v'] = str(1000 * pa_v2v.mean())

    # save the args in result
    result['args'] = args.__dict__

    # save the result
    name = '_'.join(args.checkpoint.split("/")[-2:])
    save_path = "./eval_result/%s_results.json"%name
    print("save_path: ", save_path)
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4, sort_keys=True)


