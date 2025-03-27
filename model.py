import torch
import torch.nn as nn
import timm
from transformers import EfficientFormerModel, ViTModel, AutoModel, logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import torchvision
import torch.nn.functional as F
# from config import device
# from efficientv2 import efficientformerv2_l


class Efficientformer_predict_H(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args,**kwargs)


        self.model = timm.create_model("efficientformerv2_l")
        self.model.stem.conv1.conv = nn.Conv2d(5, 20, 3, 2, 1)

        self.model.head = nn.Identity()
        self.model.head_dist = nn.Identity()

        self.classifer = nn.Sequential(
            nn.Linear(384, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1, bias=True),
        )

    def forward(self, x):

        x = self.model(x)
        x = self.classifer(x)
        return x


class Vitformer(nn.Module):
    def __init__(self):
        super(Vitformer, self).__init__()
        self.model = AutoModel.from_pretrained("./vit_pretrain_model")
        # Disable layernorm and pooler if not needed, as placeholders
        self.model.layernorm = nn.Identity()
        self.model.pooler = nn.Identity()

        self.regreesion = nn.Sequential(
            nn.Linear(196 * 768, 1024),  # Assuming '196' if [CLS] token is removed
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 4),
            nn.Dropout(0.1),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.last_hidden_state[:, 1:, :]  # Skip [CLS] token
        # print(x.shape)
        x = x.view(x.size(0), -1)  # 展平所有特征
        x = self.regreesion(x)
        return x





class VitformerForConvolutionRegression(nn.Module): # 取消全连接，一个最大池化到1
    def __init__(self):
        super(VitformerForConvolutionRegression, self).__init__()
        # 加载预训练的ViT模型
        self.vit = AutoModel.from_pretrained("./vit_pretrain_model")

        # 添加卷积层，这里以适应14x14的特征图尺寸
        self.conv_layers = nn.Sequential(
            nn.Conv2d(768, 384, 4, 2, 1), # 14-7
            nn.ReLU(),
            nn.BatchNorm2d(384),
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3, stride=2, padding=1), # 输出尺寸变为7-4
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Conv2d(in_channels=192, out_channels=96, kernel_size=3, stride=2, padding=1), # 输出尺寸变为4-2
            nn.ReLU(),
            nn.BatchNorm2d(96),
            # nn.Conv2d(in_channels=96, out_channels=1, kernel_size=1), # 输出尺寸为7x7
            # nn.ReLU(),
            # nn.BatchNorm2d(24),
            # nn.AdaptiveMaxPool2d(1)
        )

        # 根据卷积层输出尺寸调整全连接层的输入特征数
        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * 96, 4),
            nn.Softplus()
        )


    def forward(self, x):
        # 通过ViT模型
        outputs = self.vit(x)

        # 移除分类标记，保留图像块的特征
        encoder_output = outputs.last_hidden_state[:, 1:, :]  # [batch_size, 196, hidden_dim]

        # 将196个特征重新组织为14x14的特征图，假设batch_size=1
        feature_map = encoder_output.permute(0, 2, 1).view(-1, 768, 14, 14)  # [batch_size, hidden_dim, 14, 14]

        # 应用卷积层
        conv_output = self.conv_layers(feature_map)  # [batch_size, channels, height, width]

        # 展平特征图
        conv_output_flattened = torch.flatten(conv_output, 1)

        # 应用全连接层进行回归
        regression_output = self.fc(conv_output_flattened)

        return regression_output



# class VitformerForConvolutionRegression(nn.Module): # 13k_2的model
#     def __init__(self):
#         super(VitformerForConvolutionRegression, self).__init__()
#         # 加载预训练的ViT模型
#         self.vit = AutoModel.from_pretrained("./vit_pretrain_model")
#
#         # 添加卷积层，这里以适应14x14的特征图尺寸
#         self.conv_layers = nn.Sequential(
#             nn.ConvTranspose2d(768, 384, 3, 2, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(384),
#             nn.Conv2d(in_channels=384, out_channels=192, kernel_size=4, stride=2, padding=1), # 输出尺寸变为14x14
#             nn.ReLU(),
#             nn.BatchNorm2d(192),
#             nn.Conv2d(in_channels=192, out_channels=96, kernel_size=4, stride=2, padding=1), # 输出尺寸变为7x7
#             nn.ReLU(),
#             nn.BatchNorm2d(96),
#             nn.Conv2d(in_channels=96, out_channels=24, kernel_size=1), # 输出尺寸为7x7
#             nn.ReLU(),
#             nn.BatchNorm2d(24),
#             # nn.AdaptiveMaxPool2d(1)
#         )
#
#         # 根据卷积层输出尺寸调整全连接层的输入特征数
#         self.fc = nn.Linear(7*7*24, 1)
#
#     def forward(self, x):
#         # 通过ViT模型
#         outputs = self.vit(x)
#
#         # 移除分类标记，保留图像块的特征
#         encoder_output = outputs.last_hidden_state[:, 1:, :]  # [batch_size, 196, hidden_dim]
#
#         # 将196个特征重新组织为14x14的特征图，假设batch_size=1
#         feature_map = encoder_output.permute(0, 2, 1).view(-1, 768, 14, 14)  # [batch_size, hidden_dim, 14, 14]
#
#         # 应用卷积层
#         conv_output = self.conv_layers(feature_map)  # [batch_size, channels, height, width]
#
#         # 展平特征图
#         conv_output_flattened = torch.flatten(conv_output, 1)
#
#         # 应用全连接层进行回归
#         regression_output = self.fc(conv_output_flattened)
#
#         return regression_output
#





class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.patcher(x).permute(0, 2, 1)
        x = x + self.position_embedding
        x = self.dropout(x)
        # x = x.flatten(2)  #没有nn.   是x = x.
        # x = x.transpose(1, 2) #两维度交换  permute 多维度交换
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout, num_heads, activation, num_encoders):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation, batch_first=True, norm_first=True)
        self.encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.fc = nn.Sequential(
            # nn.LayerNorm(normalized_shape=embed_dim),
            nn.Flatten(),
            nn.Linear(196 * 768,512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4),
            nn.Softplus()
        )


    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_layer(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x) # 换全连接
        # print(x.shape)
        return x



class Resnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = torchvision.models.resnet101(torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
        # self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        self.classifer = nn.Sequential(
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 6),

        )



    def forward(self, x):

        x = self.model(x)
        
        x = self.classifer(x)
        
        return x




# 算氢键的
# import torch
# import torch.nn as nn
# from transformers import AutoModel
#
# class CustomSwinTransformerForRegression(nn.Module):
#     def __init__(self):
#         super(CustomSwinTransformerForRegression, self).__init__()
#         # 加载预训练的 Swin Transformer 模型
#         self.swin_transformer = AutoModel.from_pretrained("./swin_pretrain_model")
#
#         # 定义后续的卷积层
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1024, 256, 3, 2, 1),
#             nn.LeakyReLU(negative_slope=0.1),
#             # nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.Conv2d(256, 1, 3, 2),
#             nn.LeakyReLU(negative_slope=0.1),
#             # nn.ReLU(),
#             # nn.BatchNorm2d(1),
#             nn.AdaptiveMaxPool2d(1)
#             # nn.Linear(2*2*48,1)
#         )
#         # 定义一个全连接层
#         # self.fc = nn.Linear(2 * 2 * 48, 1)
#
#     def forward(self, x):
#         outputs = self.swin_transformer(x)
#         x = outputs.last_hidden_state
#         batch_size, num_patches, hidden_size = x.shape
#         H, W = int(num_patches ** 0.5), int(num_patches ** 0.5)
#         x = x.permute(0, 2, 1).contiguous().view(batch_size, hidden_size, H, W)
#
#         # 通过卷积层处理
#         x = self.conv_layers(x)
#
#         # 展平输出
#         x = torch.flatten(x, 1)
#         # 展平输出为全连接层做准备
#         # x = x.view(x.size(0), -1)  # 展平除批次外的所有维度
#
#         # 通过全连接层得到最终的输出
#         # x = self.fc(x)
#         return x


# 算氢键和SASA的
import torch
import torch.nn as nn
from transformers import AutoModel

class H_S_CustomSwinTransformerForRegression(nn.Module):
    def __init__(self):
        super(H_S_CustomSwinTransformerForRegression, self).__init__()
        # 加载预训练的 Swin Transformer 模型
        self.swin_transformer = AutoModel.from_pretrained("./swin_pretrain_model")

        # 定义后续的卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 2, 1),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.GELU(),
            nn.BatchNorm2d(512),
            # nn.Conv2d(256, 32, 3, 1, 1),swin_sasa_1版本的model
            nn.Conv2d(512, 256, 1),
            # nn.LeakyReLU(negative_slope=0.1),
            nn.GELU(),
            nn.BatchNorm2d(256)
            # nn.AdaptiveMaxPool2d(1)
        )
        # 定义一个全连接层
        self.fc = nn.Sequential(
            nn.Linear(4*4*256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.Softplus()
        )


    def forward(self, x):
        outputs = self.swin_transformer(x)
        x = outputs.last_hidden_state
        batch_size, num_patches, hidden_size = x.shape
        H, W = int(num_patches ** 0.5), int(num_patches ** 0.5)
        x = x.permute(0, 2, 1).contiguous().view(batch_size, hidden_size, H, W)

        # 通过卷积层处理
        x = self.conv_layers(x)

        # 展平输出
        x = torch.flatten(x, 1)
        # 展平输出为全连接层做准备
        # x = x.view(x.size(0), -1)  # 展平除批次外的所有维度

        # 通过全连接层得到最终的输出
        x = self.fc(x)
        return x




# class SwinTransformerHybrid(nn.Module):
#     def __init__(self, num_classes, pretrained_model_name='./swin_pretrain_model'):
#         super(SwinTransformerHybrid, self).__init__()
#         self.swin_transformer = SwinModel.from_pretrained(pretrained_model_name)
#
#         feature_dim = self.swin_transformer.config.hidden_size
#
#         # 分类器用于氢键的数量区间
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_dim,  num_classes)
#         )
#
#         # 为每个氢键数量区间定义一个回归器
#         self.hydrogen_regressors = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_classes)])
#
#         # # 为其他三个属性定义回归器
#         # self.sasa_hydrophilic_regressor = nn.Linear(feature_dim, 1)
#         # self.sasa_hydrophobic_regressor = nn.Linear(feature_dim, 1)
#         # self.sasa_total_regressor = nn.Linear(feature_dim, 1)
#
#     def forward(self, x):
#         features = self.swin_transformer(x).last_hidden_state.mean(dim=1)
#
#         classification_logits = self.classifier(features)
#         hydrogen_predictions = [regressor(features) for regressor in self.hydrogen_regressors]
#         hydrogen_predictions = torch.stack(hydrogen_predictions, dim=1)
#         _, class_indices = classification_logits.max(dim=1)
#         hydrogen_prediction = hydrogen_predictions[torch.arange(hydrogen_predictions.size(0)), class_indices]
#
#         # sasa_hydrophilic_prediction = self.sasa_hydrophilic_regressor(features).squeeze(-1)
#         # sasa_hydrophobic_prediction = self.sasa_hydrophobic_regressor(features).squeeze(-1)
#         # sasa_total_prediction = self.sasa_total_regressor(features).squeeze(-1)
#
#         return {
#             'classification_logits': classification_logits,
#             'hydrogen_prediction': hydrogen_prediction,
#
#         }


# class Vitformer_matrix(nn.Module): # 刚度矩阵模型
#     def __init__(self):
#         super(Vitformer_matrix, self).__init__()
#         # 加载预训练的ViT模型
#         self.vit = AutoModel.from_pretrained("./vit_pretrain_model")
#
#         # 添加卷积层，这里以适应14x14的特征图尺寸
#         self.conv_layers = nn.Sequential(
#             nn.ConvTranspose2d(768, 384, 3, 2, 1, 1), # 14-28
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.BatchNorm2d(384),
#             nn.ConvTranspose2d(384, 192, 3, 2, 1, 1), # 28-56
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.BatchNorm2d(192),
#             nn.ConvTranspose2d(192, 96, 4, 1, 1),  # 56-60
#             nn.LeakyReLU(negative_slope=0.1),
#             nn.BatchNorm2d(96),
#             # nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1), # 输出尺寸不变
#             # nn.ReLU(),
#             # nn.BatchNorm2d(96),
#             nn.Conv2d(in_channels=96, out_channels=1, kernel_size=3), # 输出尺寸为60×60
#             nn.ReLU(),
#             # nn.BatchNorm2d(24),
#             # nn.AdaptiveMaxPool2d(1)
#         )
#
#     def forward(self, x):
#         # 通过ViT模型
#         outputs = self.vit(x)
#
#         # 移除分类标记，保留图像块的特征
#         encoder_output = outputs.last_hidden_state[:, 1:, :]  # [batch_size, 196, hidden_dim]
#
#         # 将196个特征重新组织为14x14的特征图，假设batch_size=1
#         feature_map = encoder_output.permute(0, 2, 1).view(-1, 768, 14, 14)  # [batch_size, hidden_dim, 14, 14]
#
#         # 应用卷积层
#         conv_output = self.conv_layers(feature_map)  # [batch_size, channels, height, width]
#
#         return conv_output



