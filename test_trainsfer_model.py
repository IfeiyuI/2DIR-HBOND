import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import tqdm

from model_sasa import H_S_CustomSwinTransformerForRegression
from dataload_h_sasa import Mydata
from config_h_sasa import device, batch_size, dataset_dir_test

plt.switch_backend("agg")

# Protein identifier
protein_class = "Protein"

# Load model
model = H_S_CustomSwinTransformerForRegression()
model.load_state_dict(torch.load(f'./swin_pth/swin_trans_h_sasa/swin_{protein_class}_h_sasa.pth', map_location=device))
model.eval()

# Set up result saving directory
result_dir = f"./plt_result_H_SASA_SWIN/swin_trans_h_sasa/plt_result_{protein_class}_base"
os.makedirs(result_dir, exist_ok=True)


def test_model(model, loader, device):
    model.to(device)
    model.eval()
    preds, trues, ids = torch.empty(0), torch.empty(0), torch.empty(0)
    with torch.no_grad():
        for x, y, id in tqdm.tqdm(loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = torch.cat((preds, output.cpu()), dim=0)
            trues = torch.cat((trues, y.cpu()), dim=0)
            ids = torch.cat((ids, id.cpu()), dim=0)
    return trues, preds, ids


def compute_metrics(true, pred):
    mae = nn.L1Loss()(true, pred).item()
    r = pearsonr(true.numpy(), pred.numpy())[0]
    return mae, r


def plot_scatter(true, pred, label, mae, r):
    plt.figure(figsize=(8, 6))
    plt.scatter(true, pred, color='blue', alpha=0.5, edgecolor='k')
    plt.plot([-5, 130], [-5, 130], 'r--', linewidth=2)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{label} | MAE: {mae:.3f}, Pearson: {r:.3f}")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, f"{label}_scatter.png"))
    plt.clf()


def plot_time_series(ids, true, pred, label):
    plt.figure(figsize=(10, 5))
    plt.plot(ids, true, '--', linewidth=0.7, label="True")
    plt.plot(ids, pred, '--', linewidth=0.7, label="Pred")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title(f"Time Series | {label}")
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{label}_timeseries.png"))
    plt.clf()


def save_results(ids, trues, preds, labels):
    data = {'ID': ids.numpy()}
    for i, label in enumerate(labels):
        data[f'true_{label}'] = trues[:, i].numpy()
        data[f'pred_{label}'] = preds[:, i].numpy()
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(result_dir, f"results_{protein_class}.xlsx"), index=False)


if __name__ == "__main__":
    test_data = Mydata(dataset_dir_test, f"images_{protein_class}", f"label_{protein_class}", train_data=False, return_ids=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    true_data, pred_data, id_data = test_model(model, test_loader, device)

    # Sort by ID
    idx = torch.argsort(id_data)
    true_data = true_data[idx]
    pred_data = pred_data[idx]
    id_data = id_data[idx]

    # Plot and metrics for selected outputs
    output_labels = ["H", "Ho"]
    for i, label in enumerate(output_labels):
        mae, r = compute_metrics(true_data[:, i], pred_data[:, i])
        plot_scatter(true_data[:, i], pred_data[:, i], label, mae, r)
        plot_time_series(id_data, true_data[:, i], pred_data[:, i], label)

    save_results(id_data, true_data, pred_data, output_labels)