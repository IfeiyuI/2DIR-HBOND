import torch
import torch.nn as nn
from transformers import AutoModel, logging

# Suppress transformer warnings
logging.set_verbosity_error()

class HSCustomSwinTransformer(nn.Module):
    """
    Swin Transformer-based regression model.
    Uses pre-trained Swin backbone + CNN + MLP for predicting 4 properties.
    """
    def __init__(self):
        super(HSCustomSwinTransformer, self).__init__()

        # Load pre-trained Swin Transformer
        self.swin = AutoModel.from_pretrained("./swin_pretrain_model")

        # Convolutional layers for feature reduction
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(256)
        )

        # Fully connected layers for regression
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.Softplus()  # Ensure positive outputs
        )

    def forward(self, x):
        x = self.swin(x).last_hidden_state  # [B, N, C]
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]

        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x