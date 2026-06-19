# src/utils.py
import argparse
import os
import random
from matplotlib import pyplot as plt
import torch
import numpy as np
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


default_config = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "CLIP_config.yaml"
)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default=default_config,
    help="Path to config.yaml"
)

args = parser.parse_args()


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    
def save_training_log(epochs_list, r1_list, pos_neg_gap_list, OUTPUT_DIR):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, r1_list, marker='o', label='R@1 avg')
    plt.plot(epochs_list, pos_neg_gap_list, marker='s', label='pos_neg_gap')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    plt.savefig(OUTPUT_DIR + "CTCLIP_training_metrics.png", dpi=300)
    plt.close()