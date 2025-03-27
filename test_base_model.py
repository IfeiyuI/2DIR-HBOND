import numpy as np
import pandas as pd
from model_sasa import Efficientformer_predict_H, Vitformer, Resnet, VitformerForConvolutionRegression, \
    H_S_CustomSwinTransformerForRegression, ViT
import torch
import tqdm
import torch.nn as nn
from scipy.stats import pearsonr
from dataload_h_sasa import Mydata
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# from swin_self import SwinTransformer

plt.switch_backend("agg")
from config_h_sasa import pearson, pearson_np, device, dataset_dir_test, \
    images_dir, label_dir, images_base_dir, label_base_dir, batch_size
from matplotlib.widgets import Button
import cv2

# load_model = VitformerForConvolutionRegression()
# load_model = SwinTransformer()
load_model = H_S_CustomSwinTransformerForRegression()

# load_model = Resnet()

# 载入指定模型
# load_model.load_state_dict(torch.load('./vit_pth/vit_base_H_SASA_MSE.pth', map_location=device))
# load_model.load_state_dict(torch.load('best_model.pth', map_location=device))
load_model.load_state_dict(torch.load('./swin_pth/SWIN_base_H_second_swin_more.pth', map_location=device))
load_model.eval()


def test(net, test_loader, device):
    net = net.to(device)

    # Access labels_min and labels_max from the dataset
    labels_min = torch.tensor(test_loader.dataset.labels_min, dtype=torch.float32)
    labels_max = torch.tensor(test_loader.dataset.labels_max, dtype=torch.float32)

    def inverse_normalize_labels(normalized_labels):
        """Inverse normalization to convert labels back to original scale."""
        return normalized_labels * (labels_max - labels_min) + labels_min

    id_data = torch.empty(0)
    true_data = torch.empty(0)
    pred_data = torch.empty(0)

    net.eval()
    loop_test = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for batch_idx2, (x, target, id) in loop_test:
            x = x.to(device)
            target = target.to(device)
            output = net(x)

            # Inverse normalize
            output = inverse_normalize_labels(output.cpu())
            target = inverse_normalize_labels(target.cpu())

            # Accumulate results
            true_data = torch.cat((true_data, target), dim=0)
            pred_data = torch.cat((pred_data, output), dim=0)
            id_data = torch.cat((id_data, id.cpu()), dim=0)

            # Compute losses and Pearson correlations
            test_loss_0 = nn.L1Loss()(true_data[:, 0], pred_data[:, 0]).item()
            test_loss_1 = nn.L1Loss()(true_data[:, 1], pred_data[:, 1]).item()
            test_loss_2 = nn.L1Loss()(true_data[:, 2], pred_data[:, 2]).item()
            test_loss_3 = nn.L1Loss()(true_data[:, 3], pred_data[:, 3]).item()

            pearsonr_test_0 = pearson(true_data[:, 0], pred_data[:, 0]).item()
            pearsonr_test_1 = pearson(true_data[:, 1], pred_data[:, 1]).item()
            pearsonr_test_2 = pearson(true_data[:, 2], pred_data[:, 2]).item()
            pearsonr_test_3 = pearson(true_data[:, 3], pred_data[:, 3]).item()

            loop_test.set_description(f"Epoch_test:")
            loop_test.set_postfix(hl=test_loss_0, hp=pearsonr_test_0,
                                  Hol=test_loss_1, Hop=pearsonr_test_1,
                                  Ll=test_loss_2, Lp=pearsonr_test_2,
                                  pIl=test_loss_3, pIp=pearsonr_test_3,
                                  )

        # Convert tensors to numpy arrays for saving and plotting
        id_data_np = id_data.numpy()
        true_data_np = true_data.numpy()
        pred_data_np = pred_data.numpy()

        # Prepare DataFrame
        df_result = pd.DataFrame({
            "id": id_data_np,
            "true_H": true_data_np[:, 0],
            "pred_H": pred_data_np[:, 0],
            "true_Ho": true_data_np[:, 1],
            "pred_Ho": pred_data_np[:, 1],
            "true_L": true_data_np[:, 2],
            "pred_L": pred_data_np[:, 2],
            "true_pI": true_data_np[:, 3],
            "pred_pI": pred_data_np[:, 3],
        })

        # Save results to Excel
        df_result.to_excel("./plt_result_H_SASA_SWIN/swin_self/result_h_sasa_swin_self.xlsx", index=False)

        return true_data_np, pred_data_np, id_data_np


# Rest of your code remains the same...


# 根据id进行排序
def sort_by_id(id_data, pre_data, label_data):
    id_index = np.argsort(id_data)
    id_data = id_data[id_index]
    pre_data = pre_data[id_index]
    label_data = label_data[id_index]
    return id_data, pre_data, label_data


def plot(pre_data_, lab_data_, str):
    plt.figure()
    # 计算 Pearson 系数
    pearson_coeff = pearsonr(lab_data_, pre_data_)[0]
    # 计算 MAE Loss
    mae_loss = np.mean(np.abs(lab_data_ - pre_data_))

    # Coil 的散点图
    plt.scatter(lab_data_, pre_data_, color='b', alpha=0.5)
    plt.plot([-5, 130], [-5, 130], color='red', linestyle='--', label='y=x直线')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{str} - Pearson: {pearson_coeff:.2f}, MAE: {mae_loss:.2f}')
    # 添加图例
    plt.legend(loc='upper left')
    plt.savefig(f"./plt_result_H_SASA_SWIN/swin_self/{str}_pearson_mae.png")
    plt.show()


def figure(id_data, pre_data, lab_data, str):
    plt.figure()
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Time Series Plot about {str}")

    # fig, ax = plt.subplot()
    plt.subplots_adjust(bottom=0.2)

    # 初始化图标
    l_label, = plt.plot(id_data, lab_data, label="MD", marker='x', markersize=1, linestyle='--', linewidth=0.5,
                        visible=True)
    l_predict, = plt.plot(id_data, pre_data, label="ML", marker='o', markersize=1, linestyle='--', linewidth=0.5,
                          visible=True)

    # 定义按钮行为
    def show_lab(event):
        l_label.set_visible(True if (l_label.get_visible() is False) else False)
        plt.draw()

    def show_pre(event):
        l_predict.set_visible(True if (l_predict.get_visible() is False) else False)
        plt.draw()

    axprev = plt.axes([0.78, 0.82, 0.08, 0.06])
    axnext = plt.axes([0.78, 0.75, 0.08, 0.06])

    bnext = Button(axnext, "ML", color='#FFD700', hovercolor='dimgray')
    bnext.on_clicked(show_pre)
    bprev = Button(axprev, "MD", color='dodgerblue', hovercolor='dimgray')
    bprev.on_clicked(show_lab)
    # plt.legend()

    plt.savefig(f"./plt_result_H_SASA_SWIN/swin_self/{str}_figure.png")
    # 显示图表
    plt.show()
    plt.clf()


if __name__ == "__main__":
    test_data = Mydata(dataset_dir_test, images_base_dir, label_base_dir, return_ids=True, train_data=False, source_img=False)  # 数据加载
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    lab_data, pre_data, id = test(load_model, test_loader, device=device)  # 对模型进行测试

    # 根据id进行排序
    (id, pre_data, lab_data) = sort_by_id(id, pre_data, lab_data)

    pre_data = np.array(pre_data)
    lab_data = np.array(lab_data)
    id = np.array(id)

    # 散点及皮尔逊相关系数图
    plot(pre_data[:, 0], lab_data[:, 0], str="H")
    # plot(pre_data[:, 1], lab_data[:, 1], str="To")
    plot(pre_data[:, 1], lab_data[:, 1], str="Ho")
    # plot(pre_data[:, 3], lab_data[:, 3], str="Hi")
    # plot(pre_data[:, 4], lab_data[:, 4], str="M")
    # plot(pre_data[:, 5], lab_data[:, 5], str="GC")
    plot(pre_data[:, 2], lab_data[:, 2], str="L")
    plot(pre_data[:, 3], lab_data[:, 3], str="pI")

    # 时序折线图
    figure(id, pre_data[:, 0], lab_data[:, 0], str="H")
    # figure(id, pre_data[:, 1], lab_data[:, 1], str="To")
    figure(id, pre_data[:, 1], lab_data[:, 1], str="Ho")
    # figure(id, pre_data[:, 3], lab_data[:, 3], str="Hi")
    # figure(id, pre_data[:, 4], lab_data[:, 4], str="M")
    # figure(id, pre_data[:, 5], lab_data[:, 5], str="GC")
    figure(id, pre_data[:, 2], lab_data[:, 2], str="L")
    figure(id, pre_data[:, 3], lab_data[:, 3], str="pI")




#
# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss, L1Loss
# import numpy as np
# from scipy.stats import pearsonr
# import tqdm
# import pandas as pd
# import matplotlib.pyplot as plt
# from model_sasa import SwinTransformerHybrid
# from dataload_h_sasa import Mydata
#
# num_classes = 10
#
# def test(net, test_loader, device):
#     net.to(device)
#     true_data = torch.empty(0, device=device)
#     pred_data = torch.empty(0, device=device)
#     interval_labels_list = torch.empty(0, dtype=torch.long, device=device)
#
#     net.eval()
#     ce_criterion = CrossEntropyLoss()
#     mae_criterion = L1Loss()
#     ce_loss_accum = 0
#     mae_loss_accum = 0
#     n_samples = 0
#     loop_test = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
#     with torch.no_grad():
#         for batch_idx, (images, interval_labels, hydrogen_targets) in loop_test:
#             images = images.to(device)
#             interval_labels = interval_labels.to(device)
#             hydrogen_targets = hydrogen_targets.to(device)
#
#             outputs = net(images)
#             classification_logits = outputs['classification_logits']
#             predicted_hydrogen = outputs['hydrogen_prediction'].squeeze(-1)  # Ensure it is at least 1D
#
#             # Accumulate losses
#             ce_loss = ce_criterion(classification_logits, interval_labels)
#             mae_loss = mae_criterion(predicted_hydrogen.unsqueeze(-1), hydrogen_targets.unsqueeze(-1))
#             ce_loss_accum += ce_loss.item() * images.size(0)
#             mae_loss_accum += mae_loss.item() * images.size(0)
#             n_samples += images.size(0)
#
#             # Collect data for evaluation
#             true_data = torch.cat((true_data, hydrogen_targets), dim=0)
#             pred_data = torch.cat((pred_data, predicted_hydrogen), dim=0)
#
#     # Calculate overall loss and correlation
#     ce_loss_final = ce_loss_accum / n_samples
#     mae_loss_final = mae_loss_accum / n_samples
#     pearson_corr, _ = pearsonr(pred_data.cpu().numpy(), true_data.cpu().numpy())
#
#     return true_data.cpu(), pred_data.cpu(), mae_loss_final, ce_loss_final, pearson_corr
#
# def plot_results(true_data, pred_data, mae_loss, ce_loss, pearson_corr):
#     plt.figure(figsize=(10, 5))
#     plt.scatter(true_data, pred_data, color='blue', alpha=0.5)
#     plt.plot([true_data.min(), true_data.max()], [true_data.min(), true_data.max()], color='red', linestyle='--')
#     plt.xlabel('True Values')
#     plt.ylabel('Predicted Values')
#     plt.title(f'Hydrogen Bonds - True vs Predicted\nMAE Loss: {mae_loss:.2f}, CE Loss: {ce_loss:.2f}, Pearson: {pearson_corr:.2f}')
#     plt.savefig("./plt_result/hydrogen_bonds.png")
#     plt.show()
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(true_data, color='orange', label='True Data')
#     plt.plot(pred_data, color='blue', label='Predicted Data')
#     plt.legend()
#     plt.title('Comparison of True and Predicted Values')
#     plt.savefig("./plt_result/comparison_plot.png")
#     plt.show()
#
# if __name__ == "__main__":
#     device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
#     load_model = SwinTransformerHybrid(num_classes)
#     load_model.load_state_dict(torch.load('./swin_pth/swin_base_H_SASA_best.pth', map_location=device))
#     load_model.eval()
#
#     dataset_dir_test = r"./data/test"
#     images_base_dir = r"images_base_H"
#     label_base_dir = r"label_base_H"
#     test_data = Mydata(dataset_dir_test, images_base_dir, label_base_dir, num_classes, train_data=False)
#     test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
#
#     true_data, pred_data, mae_loss, ce_loss, pearson_corr = test(load_model, test_loader, device)
#     plot_results(true_data.numpy(), pred_data.numpy(), mae_loss, ce_loss, pearson_corr)
#
#     results_df = pd.DataFrame({
#         'True Values': np.round(true_data.numpy()).astype(int),
#         'Predicted Values': np.round(pred_data.numpy()).astype(int)
#     })
#     results_df.to_excel('./results/hydrogen_bond_predictions.xlsx', index=False)
#
