import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
from model_sasa import H_S_CustomSwinTransformerForRegression
from dataload_h_sasa import Mydata
from config_h_sasa import dataset_dir_test, dataset_dir_train, epochs, device, learn_rate, batch_size, pearson, images_base_dir, label_base_dir

def train(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler, device):
    best_val_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        preds, targets = [], []

        for x, y, *_ in loop:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            preds.append(output.detach())
            targets.append(y.detach())

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        log_metrics("Train", targets, preds)

        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            for x, y, *_ in tqdm.tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output, y).item()
                val_preds.append(output)
                val_targets.append(y)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        log_metrics("Val", val_targets, val_preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './swin_pth/SWIN_base_H_best.pth')
            print(f"Saved best model at epoch {epoch} with val_loss: {val_loss:.4f}")

        scheduler.step()


def log_metrics(stage, true, pred):
    metrics = {}
    for i, label in enumerate(["H", "Ho", "L", "pI"]):
        l1 = nn.L1Loss()(true[:, i], pred[:, i]).item()
        r = pearson(true[:, i], pred[:, i]).item()
        metrics[f"{label}_MAE"] = l1
        metrics[f"{label}_Pearson"] = r
    print(f"[{stage}] Metrics:", metrics)


if __name__ == "__main__":
    train_data = Mydata(dataset_dir_train, images_base_dir, label_base_dir, return_ids=True, train_data=True)
    val_data = Mydata(dataset_dir_test, images_base_dir, label_base_dir, return_ids=True, train_data=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = H_S_CustomSwinTransformerForRegression()

    backbone = [p for n, p in model.named_parameters() if "conv_layers" not in n and "fc" not in n]
    conv_layers = [p for n, p in model.named_parameters() if "conv_layers" in n]
    fc_layers = [p for n, p in model.named_parameters() if "fc" in n]

    optimizer = optim.Adam([
        {'params': backbone, 'lr': 1e-5},
        {'params': conv_layers, 'lr': 1e-4},
        {'params': fc_layers, 'lr': 1e-4}
    ], betas=(0.9, 0.999), weight_decay=1e-4)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 0.95 ** e)
    criterion = nn.MSELoss()

    train(model, train_loader, val_loader, epochs, optimizer, criterion, scheduler, device)
