# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from dataload_h_sasa import Mydata
# # from model_sasa import H_S_CustomSwinTransformerForRegression
# # import tqdm
# # # from model import vit_2dir, VisionTransformer
# # from config_h_sasa import dataset_dir_test, dataset_dir_train, epochs, device, batch_size, pearson, \
# #     images_base_dir, label_base_dir
# #
# # # 确保模型在正确的设备上
# #
# # # net = VisionTransformer(num_ss=4)
# # net = H_S_CustomSwinTransformerForRegression()
# #
# # class RMSELoss(nn.Module):
# #     def __init__(self):
# #         super(RMSELoss, self).__init__()
# #
# #     def forward(self, predicted, actual):
# #         return torch.sqrt(torch.mean((predicted - actual) ** 2))
# #
# #
# # # criterion = RMSELoss()
# #
# #
# # criterion = nn.MSELoss()
# #
# #
# # def train(net, train_loader, val_loader, epochs, optimizer, criterion, scheduler, device):
# #     # def train(net, train_loader, val_loader, epochs, optimizer, criterion, device):
# #
# #     best_l1loss = 9999
# #     best_pearson = -1
# #     best_model = None
# #     net = net.to(device)
# #
# #     for epoch in range(epochs):
# #         # 打印当前学习率
# #         current_lr = optimizer.param_groups[0]['lr']
# #         print(f"Epoch {epoch + 1}/{epochs}, Current Learning Rate: {current_lr}")
# #
# #         # net = freeze_net(net)  # 冻住stages 0， 1， 2层
# #         # net = freeze_vit_net(net)
# #         pre_data_train = torch.empty(0)
# #         lab_data_train = torch.empty(0)
# #
# #         pre_data_val = torch.empty(0)
# #         lab_data_val = torch.empty(0)
# #
# #         val_loss_sum = 0
# #
# #         # 训练
# #         loop_train = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
# #         for batch_idx, (x, target) in loop_train:
# #             # print(target.shape)
# #             # exit()
# #             net.train()
# #             # target = target.unsqueeze(1)
# #
# #             x = x.to(device)
# #             target = target.to(device)
# #             # output = net(x, z1, z2, z3)
# #             output = net(x)
# #             # print(output)
# #             # print("...............")
# #             # print(target)
# #
# #             optimizer.zero_grad()
# #             loss = criterion(output, target)
# #             loss.backward()
# #
# #             output = output.cpu().data
# #             target = target.cpu().data
# #
# #             # 计算pearson
# #             pre_data_train = torch.cat((pre_data_train, output), dim=0)
# #             lab_data_train = torch.cat((lab_data_train, target), dim=0)
# #
# #             train_loss_0 = nn.L1Loss()(lab_data_train[:, 0], pre_data_train[:, 0]).item()
# #             train_loss_1 = nn.L1Loss()(lab_data_train[:, 1], pre_data_train[:, 1]).item()
# #             train_loss_2 = nn.L1Loss()(lab_data_train[:, 2], pre_data_train[:, 2]).item()
# #             train_loss_3 = nn.L1Loss()(lab_data_train[:, 3], pre_data_train[:, 3]).item()
# #             # train_loss_4 = nn.L1Loss()(lab_data_train[:, 4], pre_data_train[:, 4]).item()
# #             # train_loss_5 = nn.L1Loss()(lab_data_train[:, 5], pre_data_train[:, 5]).item()
# #
# #             pearsonr_train_0 = pearson(lab_data_train[:, 0], pre_data_train[:, 0]).item()
# #             pearsonr_train_1 = pearson(lab_data_train[:, 1], pre_data_train[:, 1]).item()
# #             pearsonr_train_2 = pearson(lab_data_train[:, 2], pre_data_train[:, 2]).item()
# #             pearsonr_train_3 = pearson(lab_data_train[:, 3], pre_data_train[:, 3]).item()
# #             # pearsonr_train_4 = pearson(lab_data_train[:, 4], pre_data_train[:, 4]).item()
# #             # pearsonr_train_5 = pearson(lab_data_train[:, 5], pre_data_train[:, 5]).item()
# #
# #             loop_train.set_description(f"Epoch_train:[{epoch + 1}/{epochs}]")
# #             loop_train.set_postfix(hl=train_loss_0, hp=pearsonr_train_0,
# #                                    Totall=train_loss_1, Totalp=pearsonr_train_1,
# #                                    Hydrophobicl=train_loss_2, Hydrophobicp=pearsonr_train_2,
# #                                    Hydrophilicl=train_loss_3, Hydrophilicp=pearsonr_train_3,
# #                                    # βl=train_loss_4, βp=pearsonr_train_4,
# #                                    # ol=train_loss_5, op=pearsonr_train_5,
# #                                    )
# #
# #             optimizer.step()
# #
# #         scheduler.step()
# #
# #         # 验证
# #         loop_val = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
# #         with torch.no_grad():
# #             for batch_idx2, (x, target) in loop_val:
# #                 x = x.to(device)
# #                 target = target.to(device)
# #                 # output = net(x, z1, z2, z3)
# #                 output = net(x)
# #
# #                 val_loss_sum += nn.L1Loss()(output, target)
# #
# #                 output = output.cpu().data
# #                 target = target.cpu().data
# #
# #                 # 计算pearson
# #                 pre_data_val = torch.cat((pre_data_val, output), dim=0)
# #                 lab_data_val = torch.cat((lab_data_val, target), dim=0)
# #
# #                 val_loss_0 = nn.L1Loss()(lab_data_val[:, 0], pre_data_val[:, 0]).item()
# #                 val_loss_1 = nn.L1Loss()(lab_data_val[:, 1], pre_data_val[:, 1]).item()
# #                 val_loss_2 = nn.L1Loss()(lab_data_val[:, 2], pre_data_val[:, 2]).item()
# #                 val_loss_3 = nn.L1Loss()(lab_data_val[:, 3], pre_data_val[:, 3]).item()
# #                 # val_loss_4 = nn.L1Loss()(lab_data_val[:, 4], pre_data_val[:, 4]).item()
# #                 # val_loss_5 = nn.L1Loss()(lab_data_val[:, 5], pre_data_val[:, 5]).item()
# #
# #                 pearsonr_val_0 = pearson(lab_data_val[:, 0], pre_data_val[:, 0]).item()
# #                 pearsonr_val_1 = pearson(lab_data_val[:, 1], pre_data_val[:, 1]).item()
# #                 pearsonr_val_2 = pearson(lab_data_val[:, 2], pre_data_val[:, 2]).item()
# #                 pearsonr_val_3 = pearson(lab_data_val[:, 3], pre_data_val[:, 3]).item()
# #                 # pearsonr_val_4 = pearson(lab_data_val[:, 4], pre_data_val[:, 4]).item()
# #                 # pearsonr_val_5 = pearson(lab_data_val[:, 5], pre_data_val[:, 5]).item()
# #
# #                 loop_val.set_description(f"Epoch_val:")
# #                 loop_val.set_postfix(hl=val_loss_0, hp=pearsonr_val_0,
# #                                      Totall=val_loss_1, Totalp=pearsonr_val_1,
# #                                      Hydrophobicl=val_loss_2, Hydrophobicp=pearsonr_val_2,
# #                                      Hydrophilicl=val_loss_3, Hydrophilicp=pearsonr_val_3,
# #                                      # βl=val_loss_4, βp=pearsonr_val_4,
# #                                      # ol=val_loss_5, op=pearsonr_val_5,
# #                                      )
# #
# #         if best_l1loss > val_loss_sum.item():
# #             best_l1loss = val_loss_sum.item()
# #             best_model = net.state_dict()
# #             torch.save(best_model, './swin_pth/SWIN_base_H_SASA_MSE_zhengli.pth')
# #
# #
# # if __name__ == "__main__":
# #     train_data = Mydata(dataset_dir_train, images_base_dir, label_base_dir, train_data=False)
# #     test_data = Mydata(dataset_dir_test, images_base_dir, label_base_dir, train_data=False)
# #
# #     # train_data = TensorDataset(train_data.expand(3, -1))
# #     train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# #
# #     test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
# #
# #     # optimizer = optim.Adam(net.parameters(), lr=learn_rate)
# #     backbone_parameters = {param for name, param in net.named_parameters() if
# #                            "conv_layers" not in name and "fc" not in name}
# #     conv_layers_parameters = {param for name, param in net.named_parameters() if "conv_layers" in name}
# #     fc_parameters = {param for name, param in net.named_parameters() if "fc" in name}
# #
# #     # 确保参数集合之间没有重叠
# #     assert backbone_parameters.isdisjoint(conv_layers_parameters)
# #     assert backbone_parameters.isdisjoint(fc_parameters)
# #     assert conv_layers_parameters.isdisjoint(fc_parameters)
# #
# #     optimizer = optim.Adam([
# #         {'params': list(backbone_parameters), 'lr': 0.0001},
# #         {'params': list(conv_layers_parameters), 'lr': 0.001},
# #         {'params': list(fc_parameters), 'lr': 0.001}
# #     ], betas=(0.9, 0.999), weight_decay=1e-4)
# #     # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# #
# #     # 分组定义
# #     # vit_parameters = {param for name, param in net.named_parameters() if 'model' in name}
# #     # regression_parameters = {param for name, param in net.named_parameters() if 'regreesion' in name}
# #     #
# #     # # 确保参数集合之间没有重叠
# #     # assert vit_parameters.isdisjoint(regression_parameters)
# #     #
# #     # # 为不同部分设置不同的学习率
# #     # optimizer = optim.Adam([
# #     #     {'params': list(vit_parameters), 'lr': 0.00001},  # 对于预训练的 ViT, 使用较低的学习率
# #     #     {'params': list(regression_parameters), 'lr': 0.0001}  # 对于回归部分, 可以使用较高的学习率
# #     # ], betas=(0.9, 0.999), weight_decay=1e-4)
# #
# #
# #     # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)
# #     c = criterion
# #
# #     train(net, train_loader, val_loader=test_loader, epochs=epochs, optimizer=optimizer, criterion=c,
# #           scheduler=scheduler, device=device)
# #
# #     # train(net, train_loader, val_loader=test_loader, epochs=epochs, optimizer=optimizer, criterion=c, device=device)
#
#
#


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import shutil
from torch.nn import GELU
import pandas as pd
from model_sasa import H_S_CustomSwinTransformerForRegression, ViT
from dataload_h_sasa import Mydata
from config_h_sasa import dataset_dir_test, dataset_dir_train, epochs, device, learn_rate, batch_size, pearson, \
    images_base_dir, label_base_dir


def weighted_mse_loss(output, target, weights):
    mse_loss = (output - target) ** 2
    weighted_loss = mse_loss * (weights ** 2)  # 权重平方后再乘以MSE
    return weighted_loss.mean()


def train(net, train_loader, val_loader, epochs, optimizer, criterion, scheduler, device):
    high_loss_samples = {}
    net = net.to(device)
    best_l1loss = 99999
    for epoch in range(epochs):
        loop_train = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        # 将 pre_data_train 和 lab_data_train 初始化到与 net 相同的设备上
        pre_data_train = torch.empty(0, device=device)
        lab_data_train = torch.empty(0, device=device)

        pre_data_val = torch.empty(0, device=device)
        lab_data_val = torch.empty(0, device=device)

        val_loss_sum = 0
        for batch_idx, (x, target, *rest) in loop_train:  # Assuming DataLoader returns ids
            net.train()
            x = x.to(device)
            target = target.to(device)
            output = net(x)

            # 计算pearson
            pre_data_train = torch.cat((pre_data_train, output), dim=0)
            lab_data_train = torch.cat((lab_data_train, target), dim=0)

            train_loss_0 = nn.L1Loss()(lab_data_train[:, 0], pre_data_train[:, 0]).item()
            train_loss_1 = nn.L1Loss()(lab_data_train[:, 1], pre_data_train[:, 1]).item()
            train_loss_2 = nn.L1Loss()(lab_data_train[:, 2], pre_data_train[:, 2]).item()
            train_loss_3 = nn.L1Loss()(lab_data_train[:, 3], pre_data_train[:, 3]).item()
            # train_loss_4 = nn.L1Loss()(lab_data_train[:, 4], pre_data_train[:, 4]).item()
            # train_loss_5 = nn.L1Loss()(lab_data_train[:, 5], pre_data_train[:, 5]).item()
            # train_loss_6 = nn.L1Loss()(lab_data_train[:, 6], pre_data_train[:, 6]).item()
            # train_loss_7 = nn.L1Loss()(lab_data_train[:, 7], pre_data_train[:, 7]).item()

            pearsonr_train_0 = pearson(lab_data_train[:, 0], pre_data_train[:, 0]).item()
            pearsonr_train_1 = pearson(lab_data_train[:, 1], pre_data_train[:, 1]).item()
            pearsonr_train_2 = pearson(lab_data_train[:, 2], pre_data_train[:, 2]).item()
            pearsonr_train_3 = pearson(lab_data_train[:, 3], pre_data_train[:, 3]).item()
            # pearsonr_train_4 = pearson(lab_data_train[:, 4], pre_data_train[:, 4]).item()
            # pearsonr_train_5 = pearson(lab_data_train[:, 5], pre_data_train[:, 5]).item()
            # pearsonr_train_6 = pearson(lab_data_train[:, 6], pre_data_train[:, 6]).item()
            # pearsonr_train_7 = pearson(lab_data_train[:, 7], pre_data_train[:, 7]).item()

            loop_train.set_description(f"Epoch_train:")
            loop_train.set_postfix(hl=train_loss_0, hp=pearsonr_train_0,

                                   Hol=train_loss_1, Hop=pearsonr_train_1,
                                   # Hil=train_loss_2, Hip=pearsonr_train_2,

                                   # GCl=train_loss_3, GCp=pearsonr_train_3,
                                   Ll=train_loss_2, Lp=pearsonr_train_2,
                                   pIl=train_loss_3, pIp=pearsonr_train_3,
                                   )

            optimizer.zero_grad()
            # loss = weighted_mse_loss(output, target, weights)  # 使用加权损失函数
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3.0)
            optimizer.step()

        scheduler.step()

        # 验证
        loop_val = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
        with torch.no_grad():
            for batch_idx2, (x, target, *rest) in loop_val:
                x = x.to(device)
                target = target.to(device)
                output = net(x)

                # val_loss_sum += weighted_mse_loss(output, target, weights)
                val_loss_sum += nn.L1Loss()(output, target)

                # 确保 pre_data_val 和 output 在同一个设备上
                pre_data_val = pre_data_val.to(device)
                output = output.to(device)
                lab_data_val = lab_data_val.to(device)
                target = target.to(device)

                # 计算pearson
                pre_data_val = torch.cat((pre_data_val, output), dim=0)
                lab_data_val = torch.cat((lab_data_val, target), dim=0)

                val_loss_0 = nn.L1Loss()(lab_data_val[:, 0], pre_data_val[:, 0]).item()
                val_loss_1 = nn.L1Loss()(lab_data_val[:, 1], pre_data_val[:, 1]).item()
                val_loss_2 = nn.L1Loss()(lab_data_val[:, 2], pre_data_val[:, 2]).item()
                val_loss_3 = nn.L1Loss()(lab_data_val[:, 3], pre_data_val[:, 3]).item()
                # val_loss_4 = nn.L1Loss()(lab_data_val[:, 4], pre_data_val[:, 4]).item()
                # val_loss_5 = nn.L1Loss()(lab_data_val[:, 5], pre_data_val[:, 5]).item()

                pearsonr_val_0 = pearson(lab_data_val[:, 0], pre_data_val[:, 0]).item()
                pearsonr_val_1 = pearson(lab_data_val[:, 1], pre_data_val[:, 1]).item()
                pearsonr_val_2 = pearson(lab_data_val[:, 2], pre_data_val[:, 2]).item()
                pearsonr_val_3 = pearson(lab_data_val[:, 3], pre_data_val[:, 3]).item()
                # pearsonr_val_4 = pearson(lab_data_val[:, 4], pre_data_val[:, 4]).item()
                # pearsonr_val_5 = pearson(lab_data_val[:, 5], pre_data_val[:, 5]).item()

                loop_val.set_description(f"Epoch_val:")
                loop_val.set_postfix(hl=val_loss_0, hp=pearsonr_val_0,
                                     Hol=val_loss_1, Hop=pearsonr_val_1,
                                     # Hil=val_loss_2, Hip=pearsonr_val_2,
                                     # GCl=val_loss_3, GCp=pearsonr_val_3,
                                     Ll=val_loss_2, Lp=pearsonr_val_2,
                                     pIl=val_loss_3, pIp=pearsonr_val_3,
                                     )

        if best_l1loss > val_loss_sum.item():
            best_l1loss = val_loss_sum.item()
            best_model = net.state_dict()
            torch.save(best_model, './swin_pth/SWIN_base_H_second_swin_more.pth')
            print(f'Best model saved at epoch {epoch} with val_loss_sum: {val_loss_sum.item()}')
        else:
            print(f'No improvement at epoch {epoch}, val_loss_sum: {val_loss_sum.item()}, best_l1loss: {best_l1loss}')


if __name__ == "__main__":
    # 初始化数据加载器，确保它可以返回样本的ID
    train_data = Mydata(dataset_dir_train, images_base_dir, label_base_dir, return_ids=True, train_data=True)
    test_data = Mydata(dataset_dir_test, images_base_dir, label_base_dir, return_ids=True, train_data=False)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器和学习率调度器
    net = H_S_CustomSwinTransformerForRegression().to(device)

    backbone_parameters = {param for name, param in net.named_parameters() if
                           "conv_layers" not in name and "fc" not in name}
    conv_layers_parameters = {param for name, param in net.named_parameters() if "conv_layers" in name}
    fc_parameters = {param for name, param in net.named_parameters() if "fc" in name}

    # 确保参数集合之间没有重叠
    assert backbone_parameters.isdisjoint(conv_layers_parameters)
    assert backbone_parameters.isdisjoint(fc_parameters)
    assert conv_layers_parameters.isdisjoint(fc_parameters)

    optimizer = optim.Adam([
        {'params': list(backbone_parameters), 'lr': 0.00001},
        {'params': list(conv_layers_parameters), 'lr': 0.0001},
        {'params': list(fc_parameters), 'lr': 0.0001}
    ], betas=(0.9, 0.999), weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)

    # 初始化损失函数的权重
    # weights = torch.tensor([1, 1.414, 1.414, 1.732, 1, 3], dtype=torch.float32).to(device)

    # criterion 已经改为使用 weighted_mse_loss 进行计算
    # criterion = lambda output, target: weighted_mse_loss(output, target, weights)
    criterion = nn.MSELoss()
    train(net, train_loader, test_loader, epochs, optimizer, criterion, scheduler, device)

#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import tqdm
# from swin_self import SwinTransformer
# from model_sasa import H_S_CustomSwinTransformerForRegression
# from dataload_h_sasa import Mydata  # 确保导入正确的 Mydata 类
#
# # def check_data(tensor, name=""):
# #     if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
# #         raise ValueError(f"Input tensor {name} contains NaN or Inf values")
# #     print(f"{name} - min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.mean().item()}, std: {tensor.std().item()}")
#
# def train(net, train_loader, val_loader, epochs, optimizer, criterion, scheduler, device):
#     net = net.to(device)
#     best_l1loss = float('inf')
#     for epoch in range(epochs):
#         net.train()
#         loop_train = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
#
#         for batch_idx, data in loop_train:
#             if len(data) == 2:
#                 x, target = data
#             else:
#                 x, target, _ = data  # 忽略 ID 或其他返回元素
#
#             # check_data(x, "Input Data")
#             # check_data(target, "Target")
#
#             x, target = x.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = net(x)
#
#             # check_data(output, "Model Output")
#
#             loss = criterion(output, target)
#             if torch.isnan(loss) or torch.isinf(loss):
#                 raise ValueError("Loss is NaN or Inf")
#             loss.backward()
#
#             for name, param in net.named_parameters():
#                 if param.grad is not None:
#                     print(
#                         f"Gradient for {name} - min: {param.grad.min()}, max: {param.grad.max()}, mean: {param.grad.mean()}, std: {param.grad.std()}")
#                     assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains nan or inf values"
#                 else:
#                     print(f"No gradient for {name}")
#
#             # 包装在 detect_anomaly 上下文管理器中
#             with torch.autograd.detect_anomaly():
#                 outputs = net(x)
#                 # print("Outputs: ", outputs)  # 打印输出值
#                 loss = criterion(outputs, target)
#                 # print("Loss: ", loss)  # 打印损失值
#                 loss.backward()
#                 #
#                 # for name, param in net.named_parameters():
#                 #     if param.grad is not None:
#                 #         print(
#                 #             f"Gradient for {name} - min: {param.grad.min()}, max: {param.grad.max()}, mean: {param.grad.mean()}, std: {param.grad.std()}")
#                 #     else:
#                 #         print(f"No gradient for {name}")
#
#             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
#
#
#             print(f"Input data - min: {x.min()}, max: {x.max()}, mean: {x.mean()}, std: {x.std()}")
#             assert torch.isfinite(x).all(), "Input data contains nan or inf values"
#
#             # 打印梯度统计信息
#             if batch_idx == 0 or batch_idx % 300 == 0:
#                 for name, param in net.named_parameters():
#                     if param.grad is not None:
#                         grad_mean = param.grad.mean().item()
#                         grad_std = param.grad.std().item()
#                         # print(f"{name}: Gradient mean: {grad_mean}, Gradient std: {grad_std}")
#
#             optimizer.step()
#             loop_train.set_postfix(loss=loss.item())
#
#             if batch_idx % 300 == 0:
#                 print(f"Batch {batch_idx} - Labels: {target.cpu().numpy()}, Predictions: {output.cpu().detach().numpy()}")
#                 if torch.isnan(output).any():
#                     print(f"NaN detected in output at batch {batch_idx}")
#
#         net.eval()
#         val_loss = 0.0
#
#         with torch.no_grad():
#             for batch_idx, data in enumerate(val_loader):
#                 if len(data) == 2:
#                     x, target = data
#                 else:
#                     x, target, _ = data  # 忽略 ID 或其他返回元素
#
#                 # check_data(x, "Validation Input Data")
#                 # check_data(target, "Validation Target")
#
#                 x, target = x.to(device), target.to(device)
#                 output = net(x)
#
#                 # check_data(output, "Validation Model Output")
#
#                 val_loss += criterion(output, target).item()
#
#                 if batch_idx % 300 == 0:
#                     print(f"Validation Batch {batch_idx} - Labels: {target.cpu().numpy()}, Predictions: {output.cpu().detach().numpy()}")
#                     if torch.isnan(output).any():
#                         print(f"NaN detected in validation output at batch {batch_idx}")
#
#         val_loss /= len(val_loader)
#         print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}')
#
#         if val_loss < best_l1loss:
#             best_l1loss = val_loss
#             torch.save(net.state_dict(), 'best_model.pth')
#             print("Saved Best Model")
#
#         scheduler.step()
#
# if __name__ == "__main__":
#     dataset_dir_train = r"./data/train"
#     dataset_dir_test = r"./data/test"
#     images_base_dir = r"images_base_4086"
#     label_base_dir = r"label_base_4086"
#     batch_size = 4
#     learn_rate = 1e-6  # 适度增加学习率
#     epochs = 25
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     train_data = Mydata(dataset_dir_train, images_base_dir, label_base_dir, return_ids=True, train_data=True)
#     test_data = Mydata(dataset_dir_test, images_base_dir, label_base_dir, return_ids=True, train_data=False)
#
#     train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
#
#     # net = SwinTransformer()
#     net = H_S_CustomSwinTransformerForRegression()
#
#     optimizer = optim.Adam(net.parameters(), lr=learn_rate)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
#
#     criterion = nn.L1Loss()
#
#     train(net, train_loader, test_loader, epochs, optimizer, criterion, scheduler, device)
