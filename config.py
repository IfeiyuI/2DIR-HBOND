import torch
import numpy as np

# Set device (use CUDA if available)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# Dataset paths
dataset_dir_train = "./data/train"
dataset_dir_test = "./data/test"
images_dir = "images"
label_dir = "label"
images_base_dir = "images_base"
label_base_dir = "label_base"

# Training parameters
batch_size = 8
learn_rate = 1e-4
epochs = 80

# Compute Pearson correlation (PyTorch)
def pearson(a, b):
    a_mean = torch.mean(a)
    b_mean = torch.mean(b)
    num = torch.sum((a - a_mean) * (b - b_mean))
    denom = torch.sqrt(torch.sum((a - a_mean)**2)) * torch.sqrt(torch.sum((b - b_mean)**2))
    return num / denom

# Compute Pearson correlation (NumPy)
def pearson_np(a, b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    num = np.sum((a - a_mean) * (b - b_mean))
    denom = np.sqrt(np.sum((a - a_mean)**2)) * np.sqrt(np.sum((b - b_mean)**2))
    return num / denom
