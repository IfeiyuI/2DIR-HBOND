import numpy as np
import pandas as pd
from model_sasa import Efficientformer_predict_H, Vitformer, Resnet, VitformerForConvolutionRegression, \
    H_S_CustomSwinTransformerForRegression
import torch
import tqdm
import torch.nn as nn
from scipy.stats import pearsonr
from dataload_h_sasa import Mydata
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

plt.switch_backend("agg")
from config_h_sasa import pearson, pearson_np, device, dataset_dir_test, \
    images_dir, label_dir, images_base_dir, label_base_dir, batch_size, dataset_dir_test
from matplotlib.widgets import Button
import cv2

class_protein = "2y1y"

# 载入指定模型
load_model = H_S_CustomSwinTransformerForRegression()
load_model.load_state_dict(torch.load(f'./swin_pth/swin_trans_h_sasa/swin+{class_protein}_h_sasa.pth', map_location=device))
load_model.eval()

# 定义保存文件的基础路径
base_directory = "./plt_result_H_SASA_SWIN/swin_trans_h_sasa"
result_directory = os.path.join(base_directory, f"plt_result_{class_protein}_base")
os.makedirs(result_directory, exist_ok=True)  # 确保结果目录存在

def test(net, test_loader, device):
    net = net.to(device)
    net.eval()

    true_data = torch.empty(0)
    pred_data = torch.empty(0)
    id_data = torch.empty(0)

    loop_test = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (x, target, id) in loop_test:
            x, target = x.to(device), target.to(device)
            output = net(x)
            output = output.cpu().data
            target = target.cpu().data
            id = id.cpu().data

            true_data = torch.cat((true_data, target), dim=0)
            pred_data = torch.cat((pred_data, output), dim=0)
            id_data = torch.cat((id_data, id), dim=0)

    return true_data, pred_data, id_data

def calculate_metrics(true_data, pred_data):
    mae_loss = nn.L1Loss()(true_data, pred_data).item()
    pearson_coef = pearsonr(true_data.numpy(), pred_data.numpy())[0]
    return mae_loss, pearson_coef

def plot(pre_data_, lab_data_, str, mae_loss, pearson_coef):
    plt.figure(figsize=(10, 6))
    plt.scatter(lab_data_, pre_data_, color='blue', alpha=0.5, edgecolor='k')
    plt.plot([-5, 130], [-5, 130], color='red', linestyle='--', linewidth=2, label='Ideal y=x')
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'{str} - Pearson: {pearson_coef:.3f}, MAE: {mae_loss:.3f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(result_directory, f"{str}_pearson_mae.png"))
    plt.clf()

def figure(id_data, pre_data, lab_data, str):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Sample Index")
    plt.ylabel("Values")
    plt.title(f"Time Series Plot about {str}")
    plt.plot(id_data, lab_data, label="MD", linestyle='--', linewidth=0.5)
    plt.plot(id_data, pre_data, label="ML", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(os.path.join(result_directory, f"{str}_figure.png"))
    plt.clf()

def save_results(true_data, pred_data, id_data, labels, result_directory, class_protein):
    # 创建一个 DataFrame
    columns = {f"true_{label}": true_data[:, i] for i, label in enumerate(labels)}
    columns.update({f"pred_{label}": pred_data[:, i] for i, label in enumerate(labels)})
    columns['ID'] = id_data
    df_result = pd.DataFrame(columns)

    # 保存 DataFrame 为 Excel 文件
    excel_path = os.path.join(result_directory, f"results_{class_protein}.xlsx")
    df_result.to_excel(excel_path, index=False)

if __name__ == "__main__":
    test_data = Mydata(dataset_dir_test, "images_" + class_protein, "label_" + class_protein, train_data=False, return_ids=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    true_data, pred_data, id_data = test(load_model, test_loader, device)

    # 根据id进行排序
    id_index = np.argsort(id_data.numpy())
    true_data = true_data[id_index]
    pred_data = pred_data[id_index]
    id_data = id_data[id_index]

    labels = ["H", "SASA_Total", "Hydrophobic", "Hydrophilic"]
    for i, label in enumerate(labels):
        mae_loss, pearson_coef = calculate_metrics(true_data[:, i], pred_data[:, i])
        plot(pred_data[:, i], true_data[:, i], label, mae_loss, pearson_coef)
        figure(id_data, pred_data[:, i], true_data[:, i], label)

    # 保存结果为 Excel 文件
    save_results(true_data, pred_data, id_data, labels, result_directory, class_protein)