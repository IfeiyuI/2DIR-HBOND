import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.stats import pearsonr
import tqdm

from model_sasa import H_S_CustomSwinTransformerForRegression
from dataload_h_sasa import Mydata
from config_h_sasa import pearson, device, dataset_dir_test, images_base_dir, label_base_dir, batch_size

plt.switch_backend("agg")

# Load model and weights
model = H_S_CustomSwinTransformerForRegression()
model.load_state_dict(torch.load('./swin_pth/SWIN_base_H_second_swin_more.pth', map_location=device))
model.eval()


def inverse_normalize_labels(normalized, labels_min, labels_max):
    return normalized * (labels_max - labels_min) + labels_min


def test(model, dataloader, device):
    model = model.to(device)
    labels_min = torch.tensor(dataloader.dataset.labels_min, dtype=torch.float32)
    labels_max = torch.tensor(dataloader.dataset.labels_max, dtype=torch.float32)

    ids, trues, preds = torch.empty(0), torch.empty(0), torch.empty(0)
    model.eval()
    loop = tqdm.tqdm(dataloader, total=len(dataloader))

    with torch.no_grad():
        for x, target, id in loop:
            x = x.to(device)
            target = target.to(device)
            output = model(x)

            output = inverse_normalize_labels(output.cpu(), labels_min, labels_max)
            target = inverse_normalize_labels(target.cpu(), labels_min, labels_max)

            trues = torch.cat((trues, target), dim=0)
            preds = torch.cat((preds, output), dim=0)
            ids = torch.cat((ids, id.cpu()), dim=0)

            # Logging (optional)
            loop.set_postfix({
                'MAE_H': nn.L1Loss()(trues[:, 0], preds[:, 0]).item(),
                'P_H': pearson(trues[:, 0], preds[:, 0]).item()
            })

    df = pd.DataFrame({
        "id": ids.numpy(),
        "true_H": trues[:, 0].numpy(), "pred_H": preds[:, 0].numpy(),
        "true_Ho": trues[:, 1].numpy(), "pred_Ho": preds[:, 1].numpy(),
        "true_L": trues[:, 2].numpy(), "pred_L": preds[:, 2].numpy(),
        "true_pI": trues[:, 3].numpy(), "pred_pI": preds[:, 3].numpy(),
    })
    df.to_excel("./plt_result_H_SASA_SWIN/swin_self/result_h_sasa_swin_self.xlsx", index=False)
    return df


def sort_by_id(df):
    return df.sort_values(by="id")


def plot_scatter(true_vals, pred_vals, label):
    plt.figure()
    r = pearsonr(true_vals, pred_vals)[0]
    mae = np.mean(np.abs(true_vals - pred_vals))
    plt.scatter(true_vals, pred_vals, color='b', alpha=0.5)
    plt.plot([-5, 130], [-5, 130], 'r--', label='y = x')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{label} | Pearson: {r:.2f}, MAE: {mae:.2f}')
    plt.legend()
    plt.savefig(f"./plt_result_H_SASA_SWIN/swin_self/{label}_scatter.png")
    plt.close()


def plot_time_series(ids, trues, preds, label):
    plt.figure()
    plt.plot(ids, trues, 'x--', linewidth=0.5, markersize=1, label='MD')
    plt.plot(ids, preds, 'o--', linewidth=0.5, markersize=1, label='ML')
    plt.xlabel('ID')
    plt.ylabel('Value')
    plt.title(f'Time Series of {label}')
    plt.legend()
    plt.savefig(f"./plt_result_H_SASA_SWIN/swin_self/{label}_time_series.png")
    plt.close()


if __name__ == "__main__":
    test_set = Mydata(dataset_dir_test, images_base_dir, label_base_dir, return_ids=True, train_data=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    results = test(model, test_loader, device)
    results = sort_by_id(results)

    ids = results['id'].values
    for i, label in enumerate(["H", "Ho", "L", "pI"]):
        true_vals = results[f"true_{label}"].values
        pred_vals = results[f"pred_{label}"].values
        plot_scatter(true_vals, pred_vals, label)
        plot_time_series(ids, true_vals, pred_vals, label)