import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Unified training script wrapper')
    
    # Add common arguments
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--load_pretrained', default='False', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--load_only_backbone', default='False', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--checkpoint', type=str, default='try_egogen_new_data/76509/best_global_model.pt')
    parser.add_argument('--model_cfg', type=str, default='prohmr/configs/prohmr.yaml')
    parser.add_argument('--save_dir', type=str, default='tmp')
    parser.add_argument('--data_source', type=str, default='real')
    parser.add_argument('--train_dataset_root', type=str, default='egobody_release')
    parser.add_argument('--val_dataset_root', type=str, default='egobody_release')
    parser.add_argument('--train_dataset_file', type=str, default=None)
    parser.add_argument('--val_dataset_file', type=str, default=None)
    parser.add_argument('--mix_dataset_root', type=str)
    parser.add_argument('--mix_dataset_file', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--save_step', type=int, default=500)
    parser.add_argument('--with_global_3d_loss', default='True', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--do_augment', default='True', type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--shuffle', default='True', type=lambda x: x.lower() in ['true', '1'])

    # Add new --model argument
    parser.add_argument('--model', type=str, required=True, help='Model name to select the training script')

    return parser.parse_args()

def main():
    args = parse_args()

    # Mapping from model name to script path
    model_scripts = {
        'backbone': 'train_prohmr_depth_egobody_backbone.py',
        'diffusion': 'train_prohmr_depth_egobody_diffusion.py',
        'flowmatching': 'train_prohmr_depth_egobody_flowmatching.py',
        'baseline' : 'train_prohmr_depth_egobody_baseline.py'
        # Add more as needed
    }

    script_path = model_scripts.get(args.model)
    if not script_path:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(model_scripts.keys())}")
        sys.exit(1)

    # Build the command line for the subprocess
    cmd = [sys.executable, script_path]  # typically "python"
    for k, v in vars(args).items():
        if k != "model":  # exclude --model argument when forwarding
            if isinstance(v, bool):
                cmd += [f"--{k}", str(v)]
            elif v is not None:
                cmd += [f"--{k}", str(v)]

    print("Executing:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
