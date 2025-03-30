import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import tqdm

from dataload_h_sasa import Mydata
from model_sasa import H_S_CustomSwinTransformerForRegression
from config_h_sasa import dataset_dir_test, dataset_dir_train, epochs, device, learn_rate, batch_size, pearson, images_base_dir, label_base_dir

# Load model and pre-trained weights
model = H_S_CustomSwinTransformerForRegression()
pretrained_weights = torch.load('./swin_pth/SWIN_base_H_SASA_MSE_zuizhong.pth', map_location=device)
model.load_state_dict(pretrained_weights, strict=False)

criterion = nn.MSELoss()
protein_class = "2y1y"


def train(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler, device):
    best_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        preds_train, targets_train = [], []
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)[:, :2]  # Only use first 2 outputs
            y = y[:, :2]  # Corresponding ground truth
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            preds_train.append(output.cpu().detach())
            targets_train.append(y.cpu().detach())

        preds_train = torch.cat(preds_train)
        targets_train = torch.cat(targets_train)
        log_metrics("Train", targets_train, preds_train)
        scheduler.step()

        model.eval()
        preds_val, targets_val = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for x, y in tqdm.tqdm(val_loader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                output = model(x)[:, :2]
                y = y[:, :2]
                val_loss_total += criterion(output, y).item()
                preds_val.append(output.cpu())
                targets_val.append(y.cpu())

        preds_val = torch.cat(preds_val)
        targets_val = torch.cat(targets_val)
        log_metrics("Val", targets_val, preds_val)

        if val_loss_total < best_loss:
            best_loss = val_loss_total
            torch.save(model.state_dict(), f"./swin_pth/swin_trans_h_sasa/swin_{protein_class}_h_sasa.pth")
            print(f"Saved best model with val loss: {best_loss:.4f}")


def log_metrics(stage, true, pred):
    for i, label in enumerate(["H", "Ho"]):
        mae = nn.L1Loss()(true[:, i], pred[:, i]).item()
        r = pearson(true[:, i], pred[:, i]).item()
        print(f"[{stage}] {label} - MAE: {mae:.4f}, Pearson: {r:.4f}")


if __name__ == "__main__":
    train_data = Mydata(dataset_dir_train, f"images_{protein_class}", f"label_{protein_class}", train_data=True)
    val_data = Mydata(dataset_dir_test, f"images_{protein_class}", f"label_{protein_class}", train_data=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    backbone = [p for n, p in model.named_parameters() if "conv_layers" not in n and "fc" not in n]
    conv_layers = [p for n, p in model.named_parameters() if "conv_layers" in n]
    fc_layers = [p for n, p in model.named_parameters() if "fc" in n]

    optimizer = optim.Adam([
        {'params': backbone, 'lr': 1e-6},
        {'params': conv_layers, 'lr': 1e-5},
        {'params': fc_layers, 'lr': 1e-5}
    ], betas=(0.9, 0.999), weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler, device)