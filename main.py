import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse
from data_handling.data_reader import *
from model import UNet
from data_handling.dataset import DatasetHandler
from test import TestModel

matplotlib.use('agg')

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Test a UNet model on Cityscapes dataset')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--dataset_path', type=str, required=True, help='Root path to the Cityscapes dataset')
args = parser.parse_args()

# Dataset paths
DATASET_ROOT_PATH = args.dataset_path
DATASET_MASK_PATH = os.path.join(DATASET_ROOT_PATH, "gtFine/train")
DATASET_IMAGES_PATH = os.path.join(DATASET_ROOT_PATH, "images/train")

# Generate dataframes
train_df, val_df, test_df, dataset_df = generate_dataframes(DATASET_IMAGES_PATH, DATASET_MASK_PATH)

# Dataset handler
datahandler = DatasetHandler(input_dataframe=test_df, batch_size=4, num_workers=0, prefetch_factor=0, shuffle=True)

# Model path
model_path = args.model_path

# Create results directory if it doesn't exist
if not os.path.isdir("./results"):
    os.mkdir("./results")

# Load the model
n_class = 20
model = UNet(n_channels=3, n_classes=n_class)
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("GPU is available")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    remove_prefix = 'module.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    if use_gpu:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
else:
    print(f"No model found at {model_path}")
    exit(1)

# Test the model
test_model = TestModel(model, datahandler.dataloader, use_gpu, ignore_index=0, num_classes=20)
dice, iou, _ = test_model()

# Save the results
results = {
    "Dice Score": dice,
    "mIOU": iou
}
results_df = pd.DataFrame(results, index=[0])
results_df.to_csv(f"./results/metrics.csv", index=False)
print(f"Metrics saved at ./results/metrics.csv")
