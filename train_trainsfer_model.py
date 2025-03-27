import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from sklearn import model_selection
from torch.utils.data import DataLoader
import tqdm
from dataload_h_sasa import Mydata
from model_sasa import Efficientformer_predict_H, Vitformer, Resnet, VitformerForConvolutionRegression, \
    H_S_CustomSwinTransformerForRegression
from config_h_sasa import dataset_dir_test, dataset_dir_train, epochs, device, learn_rate, batch_size, pearson, \
    images_base_dir, label_base_dir

net = H_S_CustomSwinTransformerForRegression()
class_protein = "2y1y"

pretrained_weights = torch.load('./swin_pth/SWIN_base_H_SASA_MSE_zuizhong.pth', map_location=device)
net.load_state_dict(pretrained_weights, strict=False)

criterion = nn.MSELoss()

# criterion = nn.L1Loss()


def train(net, train_loader, val_loader, epochs, optimizer, criterion, scheduler, device):
    # def train(net, train_loader, val_loader, epochs, optimizer, criterion, device):
    #
    # print("Parameter status in the model:")
    # for name, param in net.named_parameters():
    #     print(f"{name} is {'frozen' if not param.requires_grad else 'trainable'}")

    best_l1loss = 9999
    best_pearson = -1
    best_model = None
    net = net.to(device)

    for epoch in range(epochs):
        pre_data_train = torch.empty(0)
        lab_data_train = torch.empty(0)

        pre_data_val = torch.empty(0)
        lab_data_val = torch.empty(0)

        val_loss_sum = 0

        # 训练
        loop_train = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (x, target) in loop_train:
            # print(target.shape)
            # exit()
            net.train()
            # target = target.unsqueeze(1)

            x = x.to(device)
            target = target.to(device)
            # output = net(x, z1, z2, z3)
            output = net(x)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()

            # # 在训练循环中，每100个batch输出一次各层参数状态
            # if batch_idx % 100 == 0:
            #     print(f"Batch {batch_idx}, Layer Parameters:")
            #     for name, param in net.named_parameters():
            #         if param.requires_grad:
            #             print(
            #                 f"Layer: {name} | Mean: {param.data.mean().item():.6f} | Std: {param.data.std().item():.6f}")

            output = output.cpu().data
            target = target.cpu().data

            # 计算pearson
            pre_data_train = torch.cat((pre_data_train, output), dim=0)
            lab_data_train = torch.cat((lab_data_train, target), dim=0)

            train_loss_0 = nn.L1Loss()(lab_data_train[:, 0], pre_data_train[:, 0]).item()
            train_loss_1 = nn.L1Loss()(lab_data_train[:, 1], pre_data_train[:, 1]).item()
            train_loss_2 = nn.L1Loss()(lab_data_train[:, 2], pre_data_train[:, 2]).item()
            train_loss_3 = nn.L1Loss()(lab_data_train[:, 3], pre_data_train[:, 3]).item()
            # train_loss_4 = nn.L1Loss()(lab_data_train[:, 4], pre_data_train[:, 4]).item()
            # train_loss_5 = nn.L1Loss()(lab_data_train[:, 5], pre_data_train[:, 5]).item()

            pearsonr_train_0 = pearson(lab_data_train[:, 0], pre_data_train[:, 0]).item()
            pearsonr_train_1 = pearson(lab_data_train[:, 1], pre_data_train[:, 1]).item()
            pearsonr_train_2 = pearson(lab_data_train[:, 2], pre_data_train[:, 2]).item()
            pearsonr_train_3 = pearson(lab_data_train[:, 3], pre_data_train[:, 3]).item()
            # pearsonr_train_4 = pearson(lab_data_train[:, 4], pre_data_train[:, 4]).item()
            # pearsonr_train_5 = pearson(lab_data_train[:, 5], pre_data_train[:, 5]).item()

            loop_train.set_description(f"Epoch_train:[{epoch + 1}/{epochs}]")
            loop_train.set_postfix(hl=train_loss_0, hp=pearsonr_train_0,
                                   Totall=train_loss_1, Totalp=pearsonr_train_1,
                                   Hydrophobicl=train_loss_2, Hydrophobicp=pearsonr_train_2,
                                   Hydrophilicl=train_loss_3, Hydrophilicp=pearsonr_train_3,
                                   # βl=train_loss_4, βp=pearsonr_train_4,
                                   # ol=train_loss_5, op=pearsonr_train_5,
                                   )

            optimizer.step()

        scheduler.step()

        # 验证
        loop_val = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
        with torch.no_grad():
            for batch_idx2, (x, target) in loop_val:
                x = x.to(device)
                target = target.to(device)
                # output = net(x, z1, z2, z3)
                output = net(x)

                val_loss_sum += nn.L1Loss()(output, target)

                output = output.cpu().data
                target = target.cpu().data

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
                                     Totall=val_loss_1, Totalp=pearsonr_val_1,
                                     Hydrophobicl=val_loss_2, Hydrophobicp=pearsonr_val_2,
                                     Hydrophilicl=val_loss_3, Hydrophilicp=pearsonr_val_3,
                                     # βl=val_loss_4, βp=pearsonr_val_4,
                                     # ol=val_loss_5, op=pearsonr_val_5,
                                     )

        if best_l1loss > val_loss_sum.item():
            best_l1loss = val_loss_sum.item()
            best_model = net.state_dict()
            torch.save(best_model, "./swin_pth/swin_trans_h_sasa/swin+" + class_protein + "_h_sasa.pth")
            print(f"Saved new best model with validation loss: {best_l1loss}")


if __name__ == "__main__":
    train_data = Mydata(dataset_dir_train, "images_" + class_protein, "label_" + class_protein, train_data=False)
    test_data = Mydata(dataset_dir_test, "images_" + class_protein, "label_" + class_protein, train_data=False)

    # train_data = TensorDataset(train_data.expand(3, -1))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # optimizer = optim.Adam(net.parameters(), lr=learn_rate, weight_decay=0.001)
    backbone_parameters = {param for name, param in net.named_parameters() if
                           "conv_layers" not in name and "fc" not in name}
    conv_layers_parameters = {param for name, param in net.named_parameters() if "conv_layers" in name}
    fc_parameters = {param for name, param in net.named_parameters() if "fc" in name}

    # 确保参数集合之间没有重叠
    assert backbone_parameters.isdisjoint(conv_layers_parameters)
    assert backbone_parameters.isdisjoint(fc_parameters)
    assert conv_layers_parameters.isdisjoint(fc_parameters)

    optimizer = optim.Adam([
        {'params': list(backbone_parameters), 'lr': 0.000001},
        {'params': list(conv_layers_parameters), 'lr': 0.00001},
        {'params': list(fc_parameters), 'lr': 0.00001}
    ], betas=(0.9, 0.999), weight_decay=1e-4)
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    c = criterion

    train(net, train_loader, val_loader=test_loader, epochs=epochs, optimizer=optimizer, criterion=c,
          scheduler=scheduler, device=device)

    # train(net, train_loader, val_loader=test_loader, epochs=epochs, optimizer=optimizer, criterion=c, device=device)
