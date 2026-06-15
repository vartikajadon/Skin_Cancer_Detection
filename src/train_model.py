import argparse
import logging
from pathlib import Path
import sys

# Ensure src directory is in sys.path for imports
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import the existing training entrypoint
from train_efficientnet import main as efficientnet_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper to launch EfficientNet training with cloud GPU support.")
    parser.add_argument('--config', type=str, default='configs/augmentation_config.json', help='Path to augmentation config')
    parser.add_argument('--epochs_phase1', type=int, default=12, help='Number of epochs for feature‑extraction phase')
    parser.add_argument('--epochs_phase2', type=int, default=13, help='Number of epochs for fine‑tuning phase')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--use_gpu', action='store_true', help='Force TensorFlow to use GPU if available')
    return parser.parse_args()

def main():
    args = parse_args()
    # If user explicitly wants GPU, set TF environment variable before importing TensorFlow inside train_efficientnet
    if args.use_gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # expose first GPU
        logger.info('GPU usage forced via CUDA_VISIBLE_DEVICES=0')
    else:
        logger.info('Running training in default mode (GPU will be used automatically if present).')

    # Delegating to the original training script – it already reads the config and handles real vs mock training.
    efficientnet_main()

if __name__ == '__main__':
    main()
