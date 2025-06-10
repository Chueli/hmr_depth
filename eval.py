import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Unified training script wrapper')
    
    # Add common arguments
    parser.add_argument('--dataset_root', type=str, default='egobody_release')
    parser.add_argument('--checkpoint', type=str, default='try_egogen_new_data/92990/best_model.pt')  # runs_try/90505/best_model.pt data/checkpoint.pt
    parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=2, help='Number of test samples to draw')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
    parser.add_argument('--log_freq', type=int, default=100, help='How often to log results')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--shuffle', default='False', type=lambda x: x.lower() in ['true', '1'])  # todo

    # Add new --model argument
    parser.add_argument('--model', type=str, required=True, help='Model name to select the training script')

    return parser.parse_args()

def main():
    args = parse_args()

    # Mapping from model name to script path
    model_scripts = {
        'backbone': 'eval_regression_depth_egobody_backbone.py',
        'diffusion': 'eval_regression_depth_egobody_diffusion.py',
        'flowmatching': 'eval_regression_depth_egobody_flowmatching.py',
        'baseline' : 'eval_regression_depth_egobody_baseline.py'
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